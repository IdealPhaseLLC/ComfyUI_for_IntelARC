import argparse
import time

import torch
import torch.nn.functional as F

from xe2_flash import xe2_flash_attention


def sync():
    torch.xpu.synchronize()


def benchmark(fn, warmup=1, iterations=3):
    for _ in range(warmup):
        fn()

    sync()

    start = time.perf_counter()

    for _ in range(iterations):
        fn()

    sync()

    return (time.perf_counter() - start) * 1000.0 / iterations


parser = argparse.ArgumentParser()

parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--seq", type=int, default=1024)
parser.add_argument("--heads", type=int, default=40)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--iters", type=int, default=3)

args = parser.parse_args()

B = args.batch
S = args.seq
H = args.heads
D = args.head_dim
C = H * D

device = "xpu"
dtype = torch.float16

print(f"B={B} H={H} S={S} D={D} C={C}")

torch.manual_seed(1234)

q = torch.randn((B, S, C), device=device, dtype=dtype)
k = torch.randn((B, S, C), device=device, dtype=dtype)
v = torch.randn((B, S, C), device=device, dtype=dtype)


def run_xe2():
    return xe2_flash_attention(q, k, v, H)


def run_torch():
    q4 = q.view(B, S, H, D).transpose(1, 2)
    k4 = k.view(B, S, H, D).transpose(1, 2)
    v4 = v.view(B, S, H, D).transpose(1, 2)

    out = F.scaled_dot_product_attention(
        q4,
        k4,
        v4,
        dropout_p=0.0,
        is_causal=False,
    )

    return out.transpose(1, 2).reshape(B, S, C)


print("Compiling / running Xe2 Triton...")
xe2 = run_xe2()
sync()

print("Running PyTorch SDPA...")
ref = run_torch()
sync()

# Full comparison for small test sizes.
if S <= 2048:
    a = xe2.float()
    b = ref.float()
else:
    # For the huge Wan shape, compare representative sections without
    # allocating another multi-hundred-MB full fp32 difference tensor.
    idx = torch.tensor(
        [0, 1, 2, S // 2, S - 3, S - 2, S - 1],
        device=device,
    )

    a = xe2.index_select(1, idx)[:, :, :1024].float()
    b = ref.index_select(1, idx)[:, :, :1024].float()

diff = (a - b).abs()

print()
print("Correctness:")
print("  max_abs :", diff.max().item())
print("  mean_abs:", diff.mean().item())

torch.testing.assert_close(
    a,
    b,
    atol=3e-2,
    rtol=3e-2,
)

print("  PASS")

xe2_ms = benchmark(run_xe2, iterations=args.iters)
torch_ms = benchmark(run_torch, iterations=args.iters)

print()
print("Performance:")
print(f"  Xe2 Triton : {xe2_ms:.3f} ms")
print(f"  PyTorch    : {torch_ms:.3f} ms")
print(f"  speedup    : {torch_ms / xe2_ms:.3f}x")