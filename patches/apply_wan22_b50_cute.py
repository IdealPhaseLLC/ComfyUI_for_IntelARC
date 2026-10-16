#!/usr/bin/env python3
"""Apply the narrow Wan2.2 FP16 CUTE eligibility patch validated on Arc Pro B50.

This intentionally fails closed if the pinned upstream file changes.  The route
is enabled only for the two self-attention contracts benchmarked against
PyTorch SDPA:
  * B=2, H=40, Q=K=V=18816, D=128, fp16
  * B=1, H=40, Q=K=V=94080, D=128, fp16
"""

from pathlib import Path
import os

EXPECTED_REVISION = "49c94a32bed69fbabd824debc4b136a6d15cae6d"
TARGET = Path(
    "/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/adapters/attention.py"
)

actual_revision = os.environ.get("OMNI_LLM_SCALER_SOURCE_REVISION")
if actual_revision != EXPECTED_REVISION:
    raise SystemExit(
        "Refusing to patch unexpected llm-scaler revision: "
        f"{actual_revision!r} != {EXPECTED_REVISION!r}"
    )

text = TARGET.read_text()

if "wan_fp16_verified = (" in text:
    print("Wan2.2 B50 CUTE patch already present")
    raise SystemExit(0)

supported_batch_block = """    supported_batch_kind = (
        (b == 2 and (self_attention or cross_attention))
        or (b == 1 and cross_attention)
        or minimax_h3_attention
        or animate2_attention
        or prefix_rectangular
    )
    if not (
"""

supported_batch_replacement = """    supported_batch_kind = (
        (b == 2 and (self_attention or cross_attention))
        or (b == 1 and cross_attention)
        or minimax_h3_attention
        or animate2_attention
        or prefix_rectangular
    )

    # Local B50/Wan2.2 FP16 route.
    # These exact contracts were validated directly against PyTorch SDPA.
    wan_fp16_verified = (
        q.dtype == torch.float16
        and heads == 40
        and dim_head == 128
        and self_attention
        and (
            (b == 2 and q_len == 18816)
            or (b == 1 and q_len == 94080)
        )
    )

    if not (
"""

dtype_block = """        and (
            q.dtype == torch.bfloat16
            or (animate2_attention and q.dtype == torch.float16)
        )
"""

dtype_replacement = """        and (
            q.dtype == torch.bfloat16
            or (animate2_attention and q.dtype == torch.float16)
            or wan_fp16_verified
        )
"""

heads_gate = """        and (heads == 32 or minimax_h3_attention or animate2_attention)
"""

heads_replacement = """        and (
            heads == 32
            or minimax_h3_attention
            or animate2_attention
            or wan_fp16_verified
        )
"""

for description, needle in (
    ("supported_batch_kind block", supported_batch_block),
    ("dtype gate", dtype_block),
    ("head-count gate", heads_gate),
):
    if text.count(needle) != 1:
        raise SystemExit(
            f"Refusing to patch: expected exactly one {description}; "
            f"found {text.count(needle)}"
        )

text = text.replace(supported_batch_block, supported_batch_replacement, 1)
text = text.replace(dtype_block, dtype_replacement, 1)
text = text.replace(heads_gate, heads_replacement, 1)
TARGET.write_text(text)

print(f"Applied Wan2.2 B50 CUTE patch to {TARGET}")
