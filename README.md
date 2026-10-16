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
| **Omni + CUTE/Kitchen, Q4_K_S / Q4_K_S** | **50.70 s/it** | **51.20 s/it** |

The Q4_K_S files used a mix dominated by `Q4_K` tensors, which is an
accelerated Kitchen GGUF route in this Omni build. In the tested workflow,
Q4_K_S was both faster and visibly higher quality than the lower quants.

Runtime VRAM during the Q4_K_S low-noise pass was about **11.4 GB of 16 GB**.

### TorchCompileModel

TorchCompileModel was tested again after enabling Omni CUTE/Kitchen. The warmed
high-noise run was **51.04 s/it**, versus **50.70 s/it without compile**, while
using additional VRAM. It is therefore not part of the recommended setup.

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
