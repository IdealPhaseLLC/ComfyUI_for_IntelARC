# ComfyUI on Intel Arc with Intel LLM Scaler Omni

This repo deploys **ComfyUI on Intel Arc/Battlemage using Intel's LLM Scaler Omni stack**.

The original hand-built native PyTorch XPU image has been retired from the main
path. Intel now maintains the difficult parts of the stack: PyTorch XPU,
oneDNN, CUTE attention, Comfy Kitchen XPU, GGUF-XPU, ComfyUI, and the matching
runtime providers. This repo stays intentionally small and adds:

- a reproducible, pinned Intel Omni source build;
- Docker Compose for a persistent ComfyUI deployment;
- a narrow Wan2.2/Arc Pro B50 CUTE routing patch validated against PyTorch SDPA;
- image validation and B50 benchmark scripts;
- notes from the original Xe2/Triton attention experiments.

The configuration was validated on an **Intel Arc Pro B50 16 GB**.

> Intel Omni `0.2.0-b2` is currently a source-build preview rather than a
> published Docker Hub image. Do not substitute `intel/llm-scaler-omni:latest`
> and expect the same stack.

## Known-good stack

The exact Intel source revision is pinned in [`omni.lock`](omni.lock):

```text
llm-scaler  49c94a32bed69fbabd824debc4b136a6d15cae6d
Omni image  0.2.0-b2
Target      bmg
PyTorch     2.14.0+xpu
oneDNN      2026.0.0
ComfyUI     0.39.0
Kitchen     0.2.37
AIMDO       0.5.5
```

The locally built Intel base image is:

```text
llm-scaler-omni:0.2.0-b2-comfyui-bmg
```

The reference image used during validation had image ID:

```text
sha256:578a322763b6f375085b43f860b9344753fea566927e3fab8209576034b7cb83
```

That ID is documentation, not a required rebuild output. The source revision
and component identities are the reproducibility contract.

## Why the local patch exists

Intel's CUTE D128 kernel is faster than the stock PyTorch/oneDNN SDPA path for
the large Wan2.2 self-attention shapes on the B50, but the upstream adapter at
the pinned revision does not admit these FP16 40-head contracts.

The derivative image enables CUTE only for the exact shapes that were
benchmarked and checked against PyTorch:

```text
B=2 H=40 S=18816 D=128 fp16
B=1 H=40 S=94080 D=128 fp16
```

The patch deliberately fails closed if the pinned Intel source revision or
expected adapter code changes. It does **not** broadly enable FP16 or arbitrary
40-head models.

See [`patches/apply_wan22_b50_cute.py`](patches/apply_wan22_b50_cute.py).

## Build

Clone this repo:

```bash
git clone https://github.com/IdealPhaseLLC/ComfyUI_for_IntelARC.git
cd ComfyUI_for_IntelARC
```

Build the exact pinned Intel Omni base image:

```bash
bash scripts/build-omni-base.sh
```

This clones Intel's `llm-scaler` repository below `.cache/`, checks out the
commit in `omni.lock`, verifies that the Omni source tree is clean, and runs
Intel's own BMG build.

Then build the small local derivative:

```bash
cp .env.example .env
docker compose build
```

The local Dockerfile does not rebuild PyTorch, ComfyUI, or the Intel runtime.
It simply derives from the pinned Omni image and applies the verified Wan2.2
B50 route.

## Persistent data

Edit `.env` if the data already lives elsewhere:

```dotenv
COMFYUI_MODELS_DIR=/volume1/docker/comfyui/models
COMFYUI_INPUT_DIR=/volume1/docker/comfyui/input
COMFYUI_OUTPUT_DIR=/volume1/docker/comfyui/output
COMFYUI_USER_DIR=/volume1/docker/comfyui/user
COMFYUI_HF_CACHE_DIR=/volume1/docker/comfyui/hf-cache
```

Models are mounted writable at `/llm/ComfyUI/models`, so ComfyUI and downloader extensions such as Civicomfy and HuggingFace use the same persistent host model tree.

The Compose service exposes `/dev/dri`, matching the validated Omni launch.
On a host with multiple Intel GPUs, verify the detected devices before assuming
`xpu:0` is the intended card.

## Validate

Before starting the long-lived service:

```bash
bash scripts/validate.sh
```

This first runs Intel's supplied image acceptance validator, then checks:

- the exact pinned llm-scaler revision;
- XPU availability and detected device names;
- presence of the local Wan2.2 B50 CUTE patch.

## Run

```bash
docker compose up -d
docker compose logs -f
```

Open:

```text
http://<host>:8188
```

The default launch is direct ComfyUI rather than Intel's DynamicVRAM launcher.
That matched the tested Wan2.2 workload and avoids weight-staging overhead when
the workflow already fits in VRAM.

Manager is enabled through ComfyUI's integrated `--enable-manager` support.

## Wan2.2 results on Arc Pro B50

The useful baseline from testing was a 13-frame Wan2.2 I2V workflow with 20
diffusion steps split 10 high-noise / 10 low-noise.

| Configuration | High noise | Low noise |
| --- | ---: | ---: |
| Original Q2 / Q3_K_S stack | ~54.0 s/it | ~59.8 s/it |
| Omni + CUTE, Q2 / Q3_K_S | ~51.2 s/it | ~54.5 s/it |
| Omni + CUTE/Kitchen, Q4_K_S / Q4_K_S | 50.70 s/it | 51.20 s/it |
| **Omni + CUTE/Kitchen, Q6_K / Q6_K** | **47.43 s/it** | **47.40 s/it** |

Q6_K is the fastest configuration tested so far for the 13-frame workload.  The 20-step diffusion portion
(10 high + 10 low) takes about **15:48** with Q6_K/Q6_K, versus about **16:59**
with Q4_K_S/Q4_K_S -- roughly a **6.9% reduction** in diffusion time while also
using the higher-precision quant.

The tested Q6_K files reported this GGUF tensor mix:

```text
F16 (694), Q6_K (400), F32 (1)
```

Each Q6_K model loaded completely, with ComfyUI reporting **11580.16 MB loaded**
and `full load: True`.  Runtime GPU memory usage was about **14.5 GiB of
15.9 GiB**, so Q6_K is a tight fit on the 16 GB B50 but remained fully resident
during the tested workflow.

For comparison, the Q4_K_S files contained a mix dominated by `Q4_K` plus
some `Q5_K` tensors.  Both `Q4_K` and `Q6_K` have managed XPU Kitchen
routes in this Omni build; the clean Q6_K tensor mix and measured timings make
Q6_K/Q6_K the current recommended configuration for short, 13-frame Wan2.2 workloads on this card.

### TorchCompileModel

TorchCompileModel was tested again after enabling Omni CUTE/Kitchen. The warmed
high-noise run was **51.04 s/it**, versus **50.70 s/it without compile**, while
using additional VRAM. It is therefore not part of the recommended setup.

## What we learned: Arc Pro B50 for image and video generation

After spending quite a bit of time testing and optimizing ComfyUI on the
Intel Arc Pro B50, the overall conclusion is positive -- with an important
distinction between **image generation** and **video diffusion**.

The B50 is an impressively versatile 16 GB GPU. It runs modern ComfyUI
workflows, SDXL, LoRAs, GGUF-quantized models, Wan2.2 image-to-video, and
Intel's optimized XPU kernels. It has also been a great platform for learning
how model quantization, VRAM residency, attention kernels, and sequence length
affect real-world generative AI performance.

### Image generation is a strong use case

Image generation works very well on the B50.

SDXL-class models produce excellent results, the 16 GB of VRAM is useful, and
performance is entirely reasonable for interactive experimentation. LoRA
workflows, model comparisons, and other image-generation tasks are a very good
fit for this card.

For someone interested in experimenting with local generative AI rather than
building a dedicated video-rendering workstation, the B50 is a surprisingly
capable and flexible GPU.

### Short video works too

Wan2.2 I2V-A14B also works.

For short sequences, the results can be genuinely impressive. In our testing,
a 13-frame Wan2.2 workflow using Q6_K models:

- fit completely in the B50's 16 GB VRAM;
- produced very good visual quality;
- maintained good subject identity and temporal consistency;
- ran at approximately **47.4 seconds per diffusion step**;
- benefited from Intel CUTE attention and XPU optimizations.

That makes short animations, animated images, GIF-style clips, and experiments
with image-to-video entirely practical. A roughly one-second animation
generated from a still image can be a very cool use of this hardware.

### Long-form video is a different workload

Where the limits become apparent is when the frame count increases.

Video diffusion does not scale linearly with the number of frames. The video
sequence becomes much larger, and self-attention becomes increasingly
expensive as sequence length grows.

For example, our observed Wan attention sequence lengths included:

| Frames | Attention sequence length |
| ---: | ---: |
| 13 | 18,816 |
| 49 | 61,152 |
| 77 | 94,080 |

The cost of self-attention grows approximately with the square of sequence
length, so longer clips become dramatically more expensive.

We experimented with increasingly aggressive GGUF quantization to determine
whether VRAM was the primary limitation.

At 49 frames, a Q3_K_S Wan model fit completely in VRAM:

```text
loaded completely; 6521.60 MB usable, 6345.78 MB loaded, full load: True
```

CUTE attention was also active.

Despite eliminating model offload, measured performance was:

```text
290.94 s/it
```

or almost **five minutes per diffusion step**.

A 20-step high-noise + low-noise workflow would therefore require roughly
**97 minutes of diffusion** for that short clip.

This was an important result because it showed that VRAM was no longer the
main problem. Even with the model fully resident on the GPU, the computational
cost of long video attention dominated performance.

### Quantization helps -- but cannot change the scaling problem

Quantization was extremely useful for exploring the limits of the card.

Some of the configurations tested included:

| Configuration | Result |
| --- | --- |
| Q6_K, 13 frames | Fully resident, ~47.4 s/it |
| Q4_K_S, longer sequences | Model begins partially offloading |
| Q3_K_M, 49 frames | Very close to fitting; ~412 MB offloaded |
| Q3_K_S, 49 frames | Fully resident, ~290.9 s/it |
| Q6_K, 77 frames | Several GB of model weights offloaded |

The experiments also showed that simply reducing model precision cannot solve
the underlying long-sequence attention cost.

Once the model fits completely in VRAM, further reductions in model size do
not remove the rapidly increasing amount of computation required by the video
sequence itself.

### Practical conclusion

The Arc Pro B50 should not be dismissed because of its video-generation
limits. Quite the opposite: getting a 16 GB Intel GPU to run these workloads
at all -- and produce high-quality results -- is pretty remarkable.

Its sweet spots are simply different.

**Great fits:**

- SDXL and other image-generation workflows;
- LoRA experimentation;
- local generative-AI development;
- learning ComfyUI and diffusion workflows;
- GGUF and quantization experimentation;
- short Wan2.2 animations;
- animated-image / GIF-style projects;
- testing Intel XPU and kernel optimizations.

**Less practical:**

- long Wan2.2 video sequences;
- multi-second high-resolution video generation;
- workloads where very long attention sequences must be processed repeatedly.

A tiny animated clip? Absolutely.

A 10-15 second AI-generated video? Technically interesting, but not a
practical workload for this card.

For those workloads, a larger and substantially faster GPU -- or a temporary
cloud GPU -- makes much more sense.

### Final takeaway

The B50 turned out to be an excellent **general-purpose generative AI
experimenting card**, rather than a dedicated video-generation card.

It generates excellent images, can generate surprisingly good short videos,
and provides enough VRAM and capability to explore models and techniques that
would otherwise require much more expensive hardware.

Perhaps most importantly, it has been a very fun platform for learning how all
of these pieces actually work.

Video diffusion just isn't its sweet spot.

## Attention microbenchmarks

See [`benchmarks/README.md`](benchmarks/README.md).

The most important result was the long Wan self-attention shape:

```text
B=1 H=40 S=94080 D=128 fp16

PyTorch / oneDNN : 6437 ms
Intel Omni CUTE  : 5311 ms
speedup          : 1.21x
```

The short Wan self-attention shape improved by roughly 16%.

## Updating Intel Omni

Do not change the base tag to `:latest` as an update strategy.

Instead:

1. select a new Intel llm-scaler commit;
2. update `omni.lock`;
3. rebuild the base;
4. let the local patch fail if Intel changed the adapter;
5. rerun `scripts/validate.sh`;
6. rerun the attention and Wan workflow benchmarks;
7. remove the local patch when Intel upstream natively supports the validated
   Wan FP16/40-head route.

This makes upstream changes deliberate rather than silently changing the
runtime underneath existing workflows.

## Legacy native-XPU work

The former all-in-one native PyTorch XPU Dockerfile is retained at:

```text
legacy/Dockerfile.native-xpu
```

The custom Xe2/Triton attention experiments are retained under:

```text
legacy/ComfyUI-Xe2Attention/
```

They were useful for understanding the workload, but Intel's CUTE kernel
ultimately outperformed the custom Triton implementation.

The complete pre-Omni repository state is also preserved on the
`native-xpu-v1` branch.

## Security

This configuration is intended for a trusted LAN. ComfyUI is not an
Internet-facing authenticated service. Do not expose port 8188 directly to the
public Internet.
