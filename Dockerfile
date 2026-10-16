ARG OMNI_BASE_IMAGE=llm-scaler-omni:0.2.0-b2-comfyui-bmg
FROM ${OMNI_BASE_IMAGE}

# Intel owns the PyTorch/oneDNN/ComfyUI/XPU stack.  This derivative stays tiny:
# it only carries the narrow Wan2.2/B50 CUTE eligibility patch we benchmarked.
COPY patches/apply_wan22_b50_cute.py /tmp/apply_wan22_b50_cute.py

RUN python /tmp/apply_wan22_b50_cute.py \
 && rm /tmp/apply_wan22_b50_cute.py

LABEL org.opencontainers.image.source="https://github.com/IdealPhaseLLC/ComfyUI_for_IntelARC" \
      org.opencontainers.image.description="Intel Omni ComfyUI derivative with verified Wan2.2 Arc Pro B50 CUTE routing"
