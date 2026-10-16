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

The end-to-end test used 13 frames and 20 diffusion steps split 10 high-noise /
10 low-noise.

| Configuration | High noise | Low noise | Approx. 20-step diffusion time |
| --- | ---: | ---: | ---: |
| Original Q2 / Q3_K_S stack | ~54.0 s/it | ~59.8 s/it | ~18:58 |
| Omni + CUTE/Kitchen, Q4_K_S / Q4_K_S | 50.70 s/it | 51.20 s/it | ~16:59 |
| **Omni + CUTE/Kitchen, Q6_K / Q6_K** | **47.43 s/it** | **47.40 s/it** | **~15:48** |

Q6_K/Q6_K reduced the measured diffusion time by about **6.9%** versus the
Q4_K_S/Q4_K_S run while also using the higher-precision quant.

The tested Q6_K files reported:

```text
gguf qtypes: F16 (694), Q6_K (400), F32 (1)
loaded completely; 12435.49 MB usable, 11580.16 MB loaded, full load: True
```

Runtime GPU memory usage reached about **14.5 GiB of 15.9 GiB**.  That leaves
limited headroom, but both high- and low-noise passes remained fully resident
in the tested workflow.

The earlier Q4_K_S files contained `F16 (694), Q4_K (356), Q5_K (44), F32
(1)`.  This Omni build has managed XPU Kitchen routes for both `Q4_K` and
`Q6_K`, while `Q5_K` is not listed among the configured Kitchen routes.
The Q6 result is therefore consistent with the clean Q6_K quantized tensor mix
making good use of the accelerated path, though the benchmark does not isolate
quantization format as the sole cause of the speed difference.

TorchCompileModel did not improve the warmed Omni result and used additional
VRAM, so the current recommended Wan2.2 configuration on the B50 is **Omni +
CUTE/Kitchen + Q6_K high + Q6_K low, without TorchCompileModel**.
