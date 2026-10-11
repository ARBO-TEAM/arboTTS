# ArboTTS

Ultra-lightweight bilingual Indonesian & English Text-to-Speech.

VITS end-to-end acoustic model trained with PyTorch on CUDA, with a unified bilingual IPA frontend (Indonesian `indo-g2p` + English CMUdict) and batch acoustic morphing for the broadcast "Arbo" persona.

## Features

- Bilingual Indonesian / English text normalization and G2P frontend
- VITS training pipeline (PyTorch + CUDA)
- Batch voice morphing (-2.0 semitones pitch/formant transform, v4 broadcast profile)
- Zero PyTorch dependency in deployment (`onnxruntime` + `numpy` only)

## Layout

```
config/           Training & model configuration (config.json)
dataset/          Pilot dataset: wavs/, file lists, preparation scripts
training/         VITS model, losses, training loop
export/           ONNX export
dist/             Packaged ONNX artifacts
arbotts/          Runtime package (frontend, inference)
tests/            Pytest suite
```

## Development

```bash
pip install -e ".[train]"
python -m pytest tests/ -v
```

## Inference requirements

Deployment needs only `onnxruntime` and `numpy` — no PyTorch.

Audio: 22,050 Hz mono 16-bit PCM WAV.
