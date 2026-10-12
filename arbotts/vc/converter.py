import os
import time
from pathlib import Path
from typing import Union, Optional
import numpy as np
import torch
import librosa
import soundfile as sf

from . import utils
from .mel_processing import spectrogram_torch
from .models import SynthesizerTrn

DEFAULT_CKPT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints_vc"


class ToneColorConverter:
    """Lightweight flow-based Voice Conversion model for timbre transfer.
    
    Transforms base Indonesian VITS speech into target speaker identity
    on CPU with ~0.1 RTF using optimized half-precision (FP16/BF16).
    """
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        use_bf16: bool = True,
    ):
        self.device = torch.device(device)
        self.use_bf16 = use_bf16 and (self.device.type == "cpu" or torch.cuda.is_available())
        self.config_path = Path(config_path or (DEFAULT_CKPT_DIR / "config.json"))
        
        # Prefer FP16 checkpoint (62 MB) if available, otherwise fallback to FP32 (125 MB)
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            fp16_ckpt = DEFAULT_CKPT_DIR / "checkpoint_fp16.pth"
            if fp16_ckpt.exists():
                self.checkpoint_path = fp16_ckpt
            else:
                self.checkpoint_path = DEFAULT_CKPT_DIR / "checkpoint.pth"

        if not self.config_path.exists() or not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Converter weights missing at {self.checkpoint_path} or config at {self.config_path}."
            )

        self.hps = utils.get_hparams_from_file(str(self.config_path))
        self.sampling_rate = self.hps.data.sampling_rate

        self.model = SynthesizerTrn(
            len(getattr(self.hps, "symbols", [])),
            self.hps.data.filter_length // 2 + 1,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        ).to(self.device)

        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)
        # Convert floating tensors to model dtype if needed
        state_dict = {
            k: v.float() if (v.is_floating_point() and self.device.type == "cpu") else v
            for k, v in ckpt["model"].items()
        }
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _get_spec(self, y: torch.Tensor) -> torch.Tensor:
        """Compute linear spectrogram matching converter filter length."""
        return spectrogram_torch(
            y,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        ).to(self.device)

    def extract_se(self, audio_input: Union[str, Path, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract a 256-dimensional speaker embedding from reference audio."""
        if isinstance(audio_input, (str, Path)):
            wav, _ = librosa.load(str(audio_input), sr=self.sampling_rate)
            wav_tensor = torch.FloatTensor(wav)
        elif isinstance(audio_input, np.ndarray):
            wav_tensor = torch.from_numpy(audio_input).float()
        elif isinstance(audio_input, torch.Tensor):
            wav_tensor = audio_input.float()
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        wav_tensor = wav_tensor.to(self.device)
        spec = self._get_spec(wav_tensor)

        with torch.no_grad():
            g = self.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
        return g.detach()

    def convert(
        self,
        source_audio: Union[str, Path, np.ndarray, torch.Tensor],
        target_se: torch.Tensor,
        source_se: Optional[torch.Tensor] = None,
        tau: float = 0.3,
    ) -> tuple[np.ndarray, float]:
        """Convert timbre of source audio to target speaker with fast FP16/BF16 inference.
        
        Returns:
            (converted_wav_np, conversion_time_sec)
        """
        if isinstance(source_audio, (str, Path)):
            wav, _ = librosa.load(str(source_audio), sr=self.sampling_rate)
            wav_tensor = torch.FloatTensor(wav)
        elif isinstance(source_audio, np.ndarray):
            wav_tensor = torch.from_numpy(source_audio).float()
        elif isinstance(source_audio, torch.Tensor):
            wav_tensor = source_audio.float()
        else:
            raise TypeError(f"Unsupported source audio type: {type(source_audio)}")

        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        wav_tensor = wav_tensor.to(self.device)
        spec = self._get_spec(wav_tensor)
        spec_len = torch.LongTensor([spec.size(-1)]).to(self.device)

        # If source_se not provided, extract it directly from the source audio
        if source_se is None:
            with torch.no_grad():
                source_se = self.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)

        t0 = time.perf_counter()
        with torch.no_grad():
            if self.use_bf16:
                autocast_device = self.device.type
                dtype = torch.bfloat16 if autocast_device == "cpu" else torch.float16
                with torch.autocast(device_type=autocast_device, dtype=dtype):
                    out_audio = self.model.voice_conversion(
                        spec,
                        spec_len,
                        sid_src=source_se.to(self.device),
                        sid_tgt=target_se.to(self.device),
                        tau=tau,
                    )[0][0, 0].cpu().float().numpy()
            else:
                out_audio = self.model.voice_conversion(
                    spec,
                    spec_len,
                    sid_src=source_se.to(self.device),
                    sid_tgt=target_se.to(self.device),
                    tau=tau,
                )[0][0, 0].cpu().numpy()
        elapsed = time.perf_counter() - t0

        return out_audio, elapsed
