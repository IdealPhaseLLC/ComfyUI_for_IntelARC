import math

import torch
import triton
import triton.language as tl


@triton.jit
def _xe2_flash_fwd_desc_kernel(
        Q,
        K,
        V,
        O,

        stride_qb,
        stride_qs,

        stride_kb,
        stride_ks,

        stride_vb,
        stride_vs,

        stride_ob,
        stride_os,

        n_ctx,
        sm_scale,

        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,

        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    """
    Forward-only non-causal self-attention.

    Physical Q/K/V/O layout:
        [B, S, H*D]

    Each program operates on one:
        [BLOCK_M, D]

    query tile for one batch/head.

    Designed initially for Wan:
        fp16
        H = 40
        D = 128
        mask = None
        self-attention
    """

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = (pid_bh // HEADS).to(tl.int64)
    head_idx = (pid_bh % HEADS).to(tl.int64)

    start_m = pid_m * BLOCK_M

    #
    # Treat each head as a strided 2D tensor:
    #
    #     [S, D]
    #
    # The D dimension is contiguous even though consecutive sequence
    # rows are separated by H*D elements.
    #
    q_base = Q + batch_idx * stride_qb + head_idx * HEAD_DIM
    k_base = K + batch_idx * stride_kb + head_idx * HEAD_DIM
    v_base = V + batch_idx * stride_vb + head_idx * HEAD_DIM
    o_base = O + batch_idx * stride_ob + head_idx * HEAD_DIM

    #
    # XPU-optimized 2D tensor descriptors.
    #
    # Last stride is deliberately 1.
    #
    q_desc = tl.make_tensor_descriptor(
        base=q_base,
        shape=[n_ctx, HEAD_DIM],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    k_desc = tl.make_tensor_descriptor(
        base=k_base,
        shape=[n_ctx, HEAD_DIM],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )

    v_desc = tl.make_tensor_descriptor(
        base=v_base,
        shape=[n_ctx, HEAD_DIM],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )

    o_desc = tl.make_tensor_descriptor(
        base=o_base,
        shape=[n_ctx, HEAD_DIM],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    #
    # Load Q once and retain it through the entire K/V sweep.
    #
    q = q_desc.load([start_m, 0])

    #
    # Online softmax state.
    #
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)

    acc = tl.zeros(
        [BLOCK_M, HEAD_DIM],
        dtype=tl.float32,
    )

    #
    # exp2(x * log2(e)) == exp(x)
    #
    qk_scale = sm_scale * 1.4426950408889634

    #
    # Our Wan target sequence lengths are multiples of 128:
    #
    #   18816 / 128 = 147
    #   94080 / 128 = 735
    #
    # so there is no partial K/V block in the target workloads.
    #
    for start_n in tl.range(
            0,
            n_ctx,
            BLOCK_N,
    ):
        #
        # Descriptor load gives [BLOCK_N, D].
        #
        k = k_desc.load([start_n, 0])

        #
        # [M,D] @ [D,N] -> [M,N]
        #
        scores = tl.dot(q, k.T)
        scores *= qk_scale

        #
        # Online numerically stable softmax.
        #
        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)

        p = tl.exp2(
            scores - m_new[:, None]
        )

        alpha = tl.exp2(
            m_i - m_new
        )

        l_new = (
                l_i * alpha
                + tl.sum(p, axis=1)
        )

        #
        # V is [N,D].
        #
        v = v_desc.load([start_n, 0])

        acc *= alpha[:, None]

        #
        # Let DPAS do FP16 x FP16 with FP32 accumulation.
        #
        acc += tl.dot(
            p.to(tl.float16),
            v,
        )

        m_i = m_new
        l_i = l_new

    out = acc / l_i[:, None]

    o_desc.store(
        [start_m, 0],
        out.to(tl.float16),
    )


def xe2_flash_attention_v2(
        q,
        k,
        v,
        heads,
        scale=None,
        *,
        block_m=128,
        block_n=64,
        num_warps=8,
        num_stages=2,
        grf_mode=None,
):
    if q.device.type != "xpu":
        raise ValueError("Xe2 FlashAttention v2 requires XPU")

    if q.dtype != torch.float16:
        raise ValueError(
            f"fp16 required for initial kernel; got {q.dtype}"
        )

    if q.ndim != 3:
        raise ValueError("Expected Q as [B,S,C]")

    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "v2 currently supports self-attention only"
        )

    b, s, channels = q.shape

    if channels % heads != 0:
        raise ValueError(
            "channels must divide evenly by heads"
        )

    head_dim = channels // heads

    if head_dim != 128:
        raise ValueError(
            f"v2 specializes head_dim=128; got {head_dim}"
        )

    #
    # Tensor descriptor fast path requires our innermost D dimension
    # to be physically contiguous.
    #
    if (
            q.stride(-1) != 1
            or k.stride(-1) != 1
            or v.stride(-1) != 1
    ):
        raise ValueError(
            "Q/K/V last dimension must be contiguous"
        )

    #
    # For this first tuned version, avoid partial sequence tiles.
    # Both real Wan workloads satisfy this.
    #
    if s % block_m != 0:
        raise ValueError(
            f"S={s} is not divisible by BLOCK_M={block_m}"
        )

    if s % block_n != 0:
        raise ValueError(
            f"S={s} is not divisible by BLOCK_N={block_n}"
        )

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)

    #
    # XPU dispatches grid axis 0 fastest.
    #
    # Keeping query blocks for the same B/H adjacent should improve
    # reuse of the same K/V data through L2.
    #
    grid = (
        triton.cdiv(s, block_m),
        b * heads,
    )

    launch_options = {
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    if grf_mode is not None:
        launch_options["grf_mode"] = grf_mode

    _xe2_flash_fwd_desc_kernel[grid](
        q,
        k,
        v,
        out,

        q.stride(0),
        q.stride(1),

        k.stride(0),
        k.stride(1),

        v.stride(0),
        v.stride(1),

        out.stride(0),
        out.stride(1),

        s,
        float(scale),

        HEADS=heads,
        HEAD_DIM=head_dim,

        BLOCK_M=block_m,
        BLOCK_N=block_n,

        **launch_options,
    )

    return out