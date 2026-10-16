#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/omni.lock"

IMAGE="${COMFYUI_IMAGE:-comfyui-intel-arc:omni-b2}"

echo "== Intel Omni acceptance =="
docker run --rm \
  --device=/dev/dri \
  "$IMAGE" \
  /llm/entrypoints/validate_comfyui_image.sh

echo
echo "== Local derivative checks =="
docker run --rm -i \
  --device=/dev/dri \
  "$IMAGE" \
  python - <<'PY'
import os
from pathlib import Path
import torch

expected_revision = "49c94a32bed69fbabd824debc4b136a6d15cae6d"
actual_revision = os.environ.get("OMNI_LLM_SCALER_SOURCE_REVISION")
assert actual_revision == expected_revision, (actual_revision, expected_revision)

p = Path("/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/adapters/attention.py")
text = p.read_text()
for needle in (
    "wan_fp16_verified = (",
    "(b == 2 and q_len == 18816)",
    "(b == 1 and q_len == 94080)",
    "or wan_fp16_verified",
):
    assert needle in text, f"missing patch marker: {needle}"

assert torch.xpu.is_available()
print("torch:", torch.__version__)
print("xpu count:", torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    print(f"xpu:{i}:", torch.xpu.get_device_name(i))
print("llm-scaler revision:", actual_revision)
print("Wan2.2 B50 CUTE patch: present")
PY
