import argparse
import time

import torch
import torch.nn.functional as F

from xe2_flash_v2 import xe2_flash_attention_v2


def sync():
    torch.xpu.synchronize()


def benchmark(fn, iterations):
    # warmup
    fn()
    sync()

    start = time.perf_counter()

    for _ in range(iterations):
        fn()

    sync()

    return (
            (time.perf_counter() - start)
            * 1000.0
            / iterations
    )


parser = argparse.ArgumentParser()

parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--seq", type=int, default=18816)
parser.add_argument("--heads", type=int, default=40)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--iters", type=int, default=1)

args = parser.parse_args()

B = args.batch
S = args.seq
H = args.heads
D = args.head_dim
C = H * D

print(
    f"B={B} H={H} S={S} D={D} C={C}"
)

torch.manual_seed(1234)

q = torch.randn(
    (B, S, C),
    device="xpu",
    dtype=torch.float16,
)

k = torch.randn_like(q)
v = torch.randn_like(q)


def run_torch():
    q4 = (
        q.view(B, S, H, D)
        .transpose(1, 2)
    )

    k4 = (
        k.view(B, S, H, D)
        .transpose(1, 2)
    )

    v4 = (
        v.view(B, S, H, D)
        .transpose(1, 2)
    )

    out = F.scaled_dot_product_attention(
        q4,
        k4,
        v4,
        dropout_p=0.0,
        is_causal=False,
    )

    return (
        out.transpose(1, 2)
        .reshape(B, S, C)
    )


print("Building PyTorch reference...")
ref = run_torch()
sync()

torch_ms = benchmark(
    run_torch,
    args.iters,
)

print()
print(f"PyTorch SDPA: {torch_ms:.3f} ms")
print()


#
# First-pass Xe2 search space.
#
# Keep this deliberately small.
#
configs = [
    ("64x64-w8-s2",   64,  64,  8, 2),
    ("128x64-w8-s2", 128,  64,  8, 2),
    ("64x128-w8-s2",  64, 128,  8, 2),
    ("128x128-w8-s2",128, 128,  8, 2),

    ("128x64-w16-s2",128,  64, 16, 2),
]


def sample(t):
    #
    # Avoid constructing another huge full-fp32 tensor.
    #
    idx = torch.tensor(
        [
            0,
            1,
            2,
            S // 2,
            S - 3,
            S - 2,
            S - 1,
            ],
        device="xpu",
    )

    return (
        t.index_select(1, idx)
        [:, :, :1024]
        .float()
    )


ref_sample = sample(ref)


print(
    "config".ljust(22),
    "ms".rjust(12),
    "vs torch".rjust(12),
    "max abs".rjust(14),
)

print("-" * 64)


for name, bm, bn, warps, stages in configs:
    try:
        def run():
            return xe2_flash_attention_v2(
                q,
                k,
                v,
                H,
                block_m=bm,
                block_n=bn,
                num_warps=warps,
                num_stages=stages,
            )

        #
        # First invocation also forces compilation.
        #
        out = run()
        sync()

        test_sample = sample(out)

        diff = (
                test_sample
                - ref_sample
        ).abs()

        max_abs = diff.max().item()

        torch.testing.assert_close(
            test_sample,
            ref_sample,
            atol=3e-2,
            rtol=3e-2,
        )

        ms = benchmark(
            run,
            args.iters,
        )

        speedup = torch_ms / ms

        print(
            name.ljust(22),
            f"{ms:10.3f}",
            f"{speedup:10.3f}x",
            f"{max_abs:12.8f}",
        )

    except Exception as e:
        print(
            name.ljust(22),
            "FAILED",
            repr(e),
        )