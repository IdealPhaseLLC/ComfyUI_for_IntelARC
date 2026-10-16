# Intel Arc Pro B50 benchmark notes

These are microbenchmarks used while evaluating Intel Omni's CUTE D128
attention path for Wan2.2 on an Arc Pro B50 16 GB.

## Validated shapes

| Workload | PyTorch/oneDNN | Omni CUTE | CUTE speedup |
| --- | ---: | ---: | ---: |
| Self, B=2 H=40 S=18816 D=128 | ~490 ms | ~422 ms | ~1.16x |
| Self, B=1 H=40 S=94080 D=128 | 6437 ms | 5311 ms | 1.21x |
| Cross, B=2 H=40 Q=18816 KV=512 D=128 | 16.09 ms | 14.76 ms | 1.09x |

The Wan2.2 local adapter patch is intentionally limited to the two validated
FP16 self-attention contracts.  Cross-attention was correct but small enough
that its whole-step benefit was negligible.

## Run

Use the built derivative image:

```bash
IMAGE=comfyui-intel-arc:omni-b2

docker run --rm --device=/dev/dri \
  -v "$PWD/benchmarks/bench_omni_attention.py:/tmp/bench.py:ro" \
  "$IMAGE" python /tmp/bench.py

docker run --rm --device=/dev/dri \
  -v "$PWD/benchmarks/bench_omni_attention.py:/tmp/bench.py:ro" \
  "$IMAGE" python /tmp/bench.py --batch 1 --seq 94080 --iters 3

docker run --rm --device=/dev/dri \
  -v "$PWD/benchmarks/bench_omni_cross.py:/tmp/bench.py:ro" \
  "$IMAGE" python /tmp/bench.py
```

Stop other GPU workloads for clean measurements.

## End-to-end Wan2.2 result

At 13 frames / 20 diffusion steps (10 high + 10 low), Q4_K_S on both experts
settled around 50.7-51.2 s/it.  The prior Q2 high / Q3_K_S low setup was about
54.0 / 59.8 s/it on the original stack.  Q4_K_S also produced visibly better
output in the tested workflow.

TorchCompileModel did not improve the warmed Omni/Q4_K_S result and used more
VRAM, so it is not part of the recommended configuration.
