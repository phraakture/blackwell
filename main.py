# run this script with
#   modal run main.py

from pathlib import Path

import modal

CURRENT_DIR = Path(__file__).parent
REMOTE_DIR = Path("/my_extension")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("ninja", "numpy")
    .add_local_dir(CURRENT_DIR, remote_path=REMOTE_DIR)
)
app = modal.App("sm100-matmul", image=image)


@app.function(gpu="B200")
def benchmark(shape: str):
    import torch
    import torch.utils.cpp_extension
    from triton.testing import do_bench

    torch.utils.cpp_extension.load(
        "module",
        sources=[REMOTE_DIR / "matmul_v0.cu"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
        extra_ldflags=["-lcuda"],  # for cuTensorMapEncodeTiled() used by TMA
        is_python_module=False,
    )
    matmul_v0 = torch.ops.my_matmul.matmul_v0

    M, N, K = map(int, shape.split(","))
    A = torch.randn(M, K).bfloat16().cuda()
    B = torch.randn(N, K).bfloat16().cuda().T

    torch.testing.assert_close(matmul_v0(A, B), torch.matmul(A, B))

    for name, f in [("CuBLAS", torch.matmul), ("v0", matmul_v0)]:
        latency_ms = do_bench(lambda: f(A, B), warmup=10, rep=100, return_mode="median")
        tflops = 2 * M * N * K / latency_ms / 1e9
        print(f"{name}:\t{latency_ms:.4f} ms\t{tflops:.2f} TFLOPS")


@app.local_entrypoint()
def main(shape: str = "4096,4096,4096"):
    benchmark.remote(shape)
