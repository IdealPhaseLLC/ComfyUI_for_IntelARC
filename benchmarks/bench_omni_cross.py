import argparse
import time

import torch
import torch.nn.functional as F
from omni_xpu_kernel import cute


def sync():
    torch.xpu.synchronize()


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) * 1000.0 / iters


parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--query", type=int, default=18816)
parser.add_argument("--kv", type=int, default=512)
parser.add_argument("--heads", type=int, default=40)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--iters", type=int, default=20)
args = parser.parse_args()

B, H, Q, KV, D = args.batch, args.heads, args.query, args.kv, args.head_dim
print(f"Cross attention: B={B} H={H} Q={Q} KV={KV} D={D}")
print("torch:", torch.__version__)
print("device:", torch.xpu.get_device_name(0))
print("CUTE D128:", cute.supports_d128_bhld())

q_blhd = torch.randn((B, Q, H, D), device="xpu", dtype=torch.float16)
k_blhd = torch.randn((B, KV, H, D), device="xpu", dtype=torch.float16)
v_blhd = torch.randn_like(k_blhd)

q = q_blhd.permute(0, 2, 1, 3)
k = k_blhd.permute(0, 2, 1, 3)
v = v_blhd.permute(0, 2, 1, 3)

print("Q shape/stride:", tuple(q.shape), q.stride())
print("K shape/stride:", tuple(k.shape), k.stride())


def run_torch():
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def run_cute():
    return cute.sdp_bhld_d128(q, k, v)


print("\nCorrectness check...")
with torch.inference_mode():
    ref = run_torch()
    test = run_cute()
    sync()

    idx = torch.tensor([0, 1, 2, Q // 2, Q - 3, Q - 2, Q - 1], device="xpu")
    a = ref.index_select(2, idx)[:, :, :, :64].float()
    b = test.index_select(2, idx)[:, :, :, :64].float()
    diff = (a - b).abs()

    print("max_abs :", diff.max().item())
    print("mean_abs:", diff.mean().item())
    torch.testing.assert_close(b, a, atol=3e-2, rtol=3e-2)
    print("PASS")

    del ref, test, a, b, diff
    torch.xpu.empty_cache()

    print("\nBenchmarking...")
    torch_ms = bench(run_torch, args.warmup, args.iters)
    cute_ms = bench(run_cute, args.warmup, args.iters)

print("\nRESULTS")
print("--------------------------------")
print(f"PyTorch / oneDNN : {torch_ms:9.3f} ms")
print(f"Intel Omni CUTE  : {cute_ms:9.3f} ms")
print(f"CUTE speedup     : {torch_ms / cute_ms:9.3f}x")
