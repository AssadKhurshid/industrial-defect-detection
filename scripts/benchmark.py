"""CPU latency of one image through the model: ONNX Runtime vs PyTorch, p50 and p95.

    python scripts/benchmark.py out/onnx/model.onnx

Times the model only (image already decoded and normalised), batch size 1, after warm-up.
PyTorch gets a memory bank of the same shape as the one in the ONNX file, filled with random
numbers: the work per image depends only on the shape, so the timing is the same.
"""

import argparse
import json
import os
import platform
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from idd.patchcore import PatchCore


def cpu_name() -> str:
    if platform.system() == "Windows":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor()


def time_ms(run, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        run()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": round(float(np.percentile(times, 50)), 1), "p95_ms": round(float(np.percentile(times, 95)), 1)}


parser = argparse.ArgumentParser()
parser.add_argument("onnx_path")
parser.add_argument("--runs", type=int, default=200)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--out", default="out/latency.json")
args = parser.parse_args()

image = np.random.default_rng(0).standard_normal((1, 3, 224, 224), dtype=np.float32)

session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
onnx_result = time_ms(lambda: session.run(None, {"image": image}), args.runs, args.warmup)

graph = onnx.load(args.onnx_path, load_external_data=False).graph
bank_size = next(t.dims[0] for t in graph.initializer if t.name == "memory_sq")
model = PatchCore()
model.memory = torch.randn(bank_size, 1536)
x = torch.from_numpy(image)
torch_result = time_ms(lambda: model.anomaly_map(x), args.runs, args.warmup)

result = {
    "cpu": cpu_name(),
    "logical_cores": os.cpu_count(),
    "torch_threads": torch.get_num_threads(),
    "onnxruntime": ort.__version__,
    "torch": torch.__version__,
    "memory_bank": bank_size,
    "runs": args.runs,
    "onnxruntime_cpu": onnx_result,
    "pytorch_cpu": torch_result,
}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
