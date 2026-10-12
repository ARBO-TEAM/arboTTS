import io
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ckpt_path = REPO_ROOT / "checkpoints_vc" / "checkpoint.pth"
out_fp16_path = REPO_ROOT / "checkpoints_vc" / "checkpoint_fp16.pth"

print("Converting checkpoint.pth to FP16 state dict...")
ckpt = torch.load(str(ckpt_path), map_location="cpu")
model_state = ckpt["model"]

half_state = {}
for k, v in model_state.items():
    if v.is_floating_point():
        half_state[k] = v.half()
    else:
        half_state[k] = v

torch.save({"model": half_state}, str(out_fp16_path))
orig_mb = ckpt_path.stat().st_size / (1024 * 1024)
new_mb = out_fp16_path.stat().st_size / (1024 * 1024)
print(f"Original FP32 size: {orig_mb:.2f} MB -> New FP16 size: {new_mb:.2f} MB (Reduction: {(1 - new_mb/orig_mb)*100:.1f}%)")
