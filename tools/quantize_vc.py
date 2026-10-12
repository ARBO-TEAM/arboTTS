import sys
import time
from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

REPO_ROOT = Path(__file__).resolve().parent.parent
fp32_onnx = REPO_ROOT / "checkpoints_vc" / "converter_fp32.onnx"
int8_onnx = REPO_ROOT / "checkpoints_vc" / "converter_int8.onnx"

print("Quantizing converter_fp32.onnx to INT8...")
t0 = time.time()
quantize_dynamic(
    model_input=str(fp32_onnx),
    model_output=str(int8_onnx),
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=True,
)
print(f"Quantization finished in {time.time()-t0:.2f}s!")
print(f"INT8 ONNX model size: {int8_onnx.stat().st_size / (1024*1024):.2f} MB")
if (REPO_ROOT / "checkpoints_vc" / "converter_int8.onnx.data").exists():
    data_size = (REPO_ROOT / "checkpoints_vc" / "converter_int8.onnx.data").stat().st_size / (1024*1024)
    print(f"INT8 data file size: {data_size:.2f} MB")
