#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/omni.lock"

SOURCE_DIR="${OMNI_SOURCE_DIR:-$ROOT_DIR/.cache/llm-scaler}"

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  mkdir -p "$(dirname "$SOURCE_DIR")"
  git clone "$LLM_SCALER_REPOSITORY" "$SOURCE_DIR"
fi

git -C "$SOURCE_DIR" fetch origin "$LLM_SCALER_COMMIT"
git -C "$SOURCE_DIR" checkout --detach "$LLM_SCALER_COMMIT"

if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=normal -- omni)" ]]; then
  echo "Refusing to build from a dirty omni tree: $SOURCE_DIR" >&2
  exit 1
fi

actual="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$actual" != "$LLM_SCALER_COMMIT" ]]; then
  echo "Revision mismatch: $actual != $LLM_SCALER_COMMIT" >&2
  exit 1
fi

echo "Building Intel Omni from $actual for Battlemage..."
(
  cd "$SOURCE_DIR/omni"
  OMNI_IMAGE_REPOSITORY=llm-scaler-omni \
  XPU_TARGET=bmg \
  bash build.sh
)

docker image inspect "$OMNI_BASE_IMAGE" >/dev/null
echo "Built $OMNI_BASE_IMAGE"
