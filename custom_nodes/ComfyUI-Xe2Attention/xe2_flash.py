import math

import torch
import triton
import triton.language as tl


LOG2E = 1.4426950408889634


@triton.jit
def _xe2_flash_fwd_kernel(
        Q,
        K,
        V,
        O,

        stride_qb,
        stride_qs,
        stride_qc,

        stride_kb,
        stride_ks,
        stride_kc,

        stride_vb,
        stride_vs,
        stride_vc,

        stride_ob,
        stride_os,
        stride_oc,

        n_ctx,
        sm_scale,

        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,

        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    """
    Forward-only, non-causal FlashAttention.

    Input/output layout:
        [B, S, H * D]

    Specialized initially for:
        fp16
        D = 128
        no mask
        no GQA
        self-attention
    """

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // HEADS
    head_idx = pid_bh % HEADS

    # Intel guidance recommends 64-bit indexing for stride arithmetic.
    batch_idx = batch_idx.to(tl.int64)
    head_idx = head_idx.to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Q/K/V are [B, S, H*D].
    q_head_base = (
            Q
            + batch_idx * stride_qb
            + head_idx * HEAD_DIM * stride_qc
    )

    k_head_base = (
            K
            + batch_idx * stride_kb
            + head_idx * HEAD_DIM * stride_kc
    )

    v_head_base = (
            V
            + batch_idx * stride_vb
            + head_idx * HEAD_DIM * stride_vc
    )

    o_head_base = (
            O
            + batch_idx * stride_ob
            + head_idx * HEAD_DIM * stride_oc
    )

    q_ptrs = (
            q_head_base
            + offs_m[:, None] * stride_qs
            + offs_d[None, :] * stride_qc
    )

    q = tl.load(
        q_ptrs,
        mask=offs_m[:, None] < n_ctx,
        other=0.0,
    )

    # Online softmax state.
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)

    # fp32 accumulator
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # exp2() is generally preferable in the kernel.
    qk_scale = sm_scale * LOG2E

    for start_n in tl.range(0, n_ctx, BLOCK_N):
        n = start_n + offs_n
        valid_n = n < n_ctx

        # Load K as [D, N] so Q @ K directly produces [M, N].
        k_ptrs = (
                k_head_base
                + offs_d[:, None] * stride_kc
                + n[None, :] * stride_ks
        )

        k = tl.load(
            k_ptrs,
            mask=valid_n[None, :],
            other=0.0,
        )

        scores = tl.dot(q, k)
        scores *= qk_scale

        scores = tl.where(
            valid_n[None, :],
            scores,
            -float("inf"),
        )

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)

        p = tl.exp2(scores - m_new[:, None])

        alpha = tl.exp2(m_i - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (
                v_head_base
                + n[:, None] * stride_vs
                + offs_d[None, :] * stride_vc
        )

        v = tl.load(
            v_ptrs,
            mask=valid_n[:, None],
            other=0.0,
        )

        acc *= alpha[:, None]

        # FP16 DPAS input, FP32 accumulation.
        acc += tl.dot(p.to(tl.float16), v)

        m_i = m_new

    output = acc / l_i[:, None]

    o_ptrs = (
            o_head_base
            + offs_m[:, None] * stride_os
            + offs_d[None, :] * stride_oc
    )

    tl.store(
        o_ptrs,
        output.to(tl.float16),
        mask=offs_m[:, None] < n_ctx,
    )


def xe2_flash_attention(q, k, v, heads, scale=None):
    if q.device.type != "xpu":
        raise ValueError("Xe2 FlashAttention requires XPU")

    if q.dtype != torch.float16:
        raise ValueError(
            f"Xe2 FlashAttention currently supports fp16 only, got {q.dtype}"
        )

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Expected Q/K/V in [B,S,C] form")

    b, s, channels = q.shape

    if channels % heads != 0:
        raise ValueError("Channels must divide evenly by heads")

    head_dim = channels // heads

    if head_dim != 128:
        raise ValueError(
            f"Initial Xe2 kernel is specialized to head_dim=128, got {head_dim}"
        )

    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("Initial kernel supports self-attention only")

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)

    #
    # Important for Intel:
    #
    # axis 0 varies fastest on XPU. That means neighboring query tiles
    # for the same B/H are dispatched near each other, which gives us
    # some chance of K/V L2 reuse.
    #
    grid = (
        triton.cdiv(s, 64),
        b * heads,
    )

    _xe2_flash_fwd_kernel[grid](
        q,
        k,
        v,
        out,

        q.stride(0),
        q.stride(1),
        q.stride(2),

        k.stride(0),
        k.stride(1),
        k.stride(2),

        v.stride(0),
        v.stride(1),
        v.stride(2),

        out.stride(0),
        out.stride(1),
        out.stride(2),

        s,
        float(scale),

        HEADS=heads,
        HEAD_DIM=head_dim,

        BLOCK_M=64,
        BLOCK_N=64,

        num_warps=8,
        num_stages=2,
    )

    return out