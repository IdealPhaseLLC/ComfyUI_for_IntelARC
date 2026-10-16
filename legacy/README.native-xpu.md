# ComfyUI on Intel Arc Pro B50 (native PyTorch XPU)

This project intentionally uses native PyTorch XPU support, not the retired Intel Extension for PyTorch (IPEX) route. It exposes only `/dev/dri/by-path/pci-0000:04:00.0-render`, which is the Arc Pro B50 on this host, so ComfyUI cannot accidentally select the Iris Xe iGPU.

It also enables ComfyUI-Manager and its legacy Extensions interface. The included Manager configuration permits catalogued model and custom-node installs on a LAN-only instance.

## Install

Copy this directory to a persistent location on the NAS, for example `/volume1/docker/comfyui-xpu`, or clone the repo like:

```bash
git clone https://github.com/IdealPhaseLLC/ComfyUI_for_IntelARC.git /volume1/docker/comfyui-xpu
```

then build and start it:

```bash
cd /volume1/docker/comfyui-xpu
docker compose build --pull
docker compose up -d
docker compose logs -f
```

Open `http://<NAS-LAN-IP>:8188` once the log reports that the server is running.

The first build is large because it downloads PyTorch and the ComfyUI dependencies. Models are not included. All model downloads land in the persistent `models/` bind mount and survive container rebuilds; they are deliberately excluded from Git.

## Extensions and model downloads

Open **Extensions** in ComfyUI to reach Manager. The default Model Manager catalog is limited; it is useful for common ComfyUI assets but is not a general Hugging Face or CivitAI search engine.

For direct Hugging Face model downloads, install the `ComfyUI-HuggingFace` custom node through **Extensions → Custom Nodes**, then restart ComfyUI. It can download a selected file directly into a ComfyUI model directory. For example, RealVisXL's FP16 checkpoint is:

```text
Repository: SG161222/RealVisXL_V5.0
File:       RealVisXL_V5.0_fp16.safetensors
Type:       checkpoint
```

For gated Hugging Face repositories such as FLUX.1, first accept the repository terms in the browser and configure a Hugging Face read token in the downloader. Never commit a token; `hf_token` is ignored by Git.

The Manager policy is in `user/__manager/config.ini`:

```ini
security_level = normal
network_mode = personal_cloud
```

These settings are intentionally appropriate only for a trusted LAN deployment. Do not expose this instance directly to the Internet.

## Model workflows

Flux and SDXL use different workflows:

| Model family | Loader path |
| --- | --- |
| FLUX.1 | Separate diffusion-model, dual-CLIP, VAE, and `ModelSamplingFlux` nodes |
| SDXL checkpoints (for example RealVisXL) | `Load Checkpoint` directly into `KSampler` |

Do not select an SDXL checkpoint in a Flux workflow, or a Flux diffusion model in an SDXL workflow.

## Verify XPU before starting ComfyUI

Run this once after the image builds:

```bash
docker compose run --rm --entrypoint python comfyui \
  -c "import torch; print('XPU:', torch.xpu.is_available()); print('devices:', torch.xpu.device_count()); print('name:', torch.xpu.get_device_name(0))"
```

Expected result: `XPU: True`, one device, and a name that identifies the Arc Pro B50/Battlemage GPU. If it reports `False`, capture the output of:

```bash
docker compose run --rm --entrypoint bash comfyui -lc 'ls -l /dev/dri; sycl-ls || true; python -c "import torch; print(torch.xpu.is_available())"'
```

## Day-to-day

```bash
docker compose up -d
docker compose logs -f
docker compose down
```

To update ComfyUI deliberately, rebuild with no cache:

```bash
docker compose build --pull --no-cache
docker compose up -d
```

Keep it LAN-only. ComfyUI has no built-in authentication; do not expose port 8188 directly to the Internet.
