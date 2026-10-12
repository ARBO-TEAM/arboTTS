import sys
import time
from pathlib import Path
import torch
import soundfile as sf
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arbotts.vc import ToneColorConverter

vc = ToneColorConverter(device="cpu")
m = vc.model

# Remove weight norm from all submodules
for name, module in m.named_modules():
    try:
        torch.nn.utils.remove_weight_norm(module)
    except Exception:
        pass

class VoiceConverterONNX(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.enc_q = model.enc_q
        self.flow = model.flow
        self.dec = model.dec
        self.zero_g = model.zero_g

    def forward(self, y, y_lengths, sid_src, sid_tgt):
        # tau=0.0 for deterministic high-fidelity conversion
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=sid_src, tau=0.0)
        z_p = self.flow(z, y_mask, g=sid_src)
        z_hat = self.flow(z_p, y_mask, g=sid_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=sid_tgt)
        return o_hat.squeeze(1)

model_onnx_wrapper = VoiceConverterONNX(m)
model_onnx_wrapper.eval()

# Dummy inputs
dummy_spec = torch.randn(1, 513, 100)
dummy_len = torch.tensor([100], dtype=torch.long)
dummy_src = torch.randn(1, 256, 1)
dummy_tgt = torch.randn(1, 256, 1)

out_onnx_path = REPO_ROOT / "checkpoints_vc" / "converter_fp32.onnx"

print(f"Exporting ToneColorConverter to {out_onnx_path.name}...")
t0 = time.time()
with torch.no_grad():
    torch.onnx.export(
        model_onnx_wrapper,
        (dummy_spec, dummy_len, dummy_src, dummy_tgt),
        str(out_onnx_path),
        input_names=["spec", "spec_lengths", "sid_src", "sid_tgt"],
        output_names=["audio"],
        dynamic_axes={
            "spec": {0: "batch", 2: "time"},
            "spec_lengths": {0: "batch"},
            "audio": {0: "batch", 1: "samples"}
        },
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
    )

print(f"FP32 ONNX exported in {time.time()-t0:.2f}s! File size: {out_onnx_path.stat().st_size / (1024*1024):.2f} MB")
