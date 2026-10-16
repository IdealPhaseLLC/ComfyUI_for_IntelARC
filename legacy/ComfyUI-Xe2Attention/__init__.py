import logging

from comfy.ldm.modules.attention import optimized_attention as comfy_optimized_attention


class Xe2AttentionInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "label": (
                    "STRING",
                    {
                        "default": "WAN",
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Intel/Xe2"

    def patch(self, model, label):
        patched = model.clone()

        # Each patched model gets its own set, so HIGH and LOW can be
        # inspected independently without flooding the log.
        seen = set()

        def seq_len(t, heads):
            if t.ndim == 3:
                # Typical Wan input: [B, S, C]
                return t.shape[1]

            if t.ndim == 4:
                # Handle either [B, H, S, D] or [B, S, H, D].
                if t.shape[1] == heads:
                    return t.shape[2]
                if t.shape[2] == heads:
                    return t.shape[1]

            return None

        def inspect_attention(
                q,
                k,
                v,
                heads,
                mask=None,
                attn_precision=None,
                skip_reshape=False,
                skip_output_reshape=False,
                **kwargs,
        ):
            q_shape = tuple(q.shape)
            k_shape = tuple(k.shape)
            v_shape = tuple(v.shape)

            q_len = seq_len(q, heads)
            k_len = seq_len(k, heads)

            if q_len is not None and k_len is not None:
                kind = "SELF" if q_len == k_len else "CROSS"
            else:
                kind = "UNKNOWN"

            if skip_reshape:
                head_dim = q.shape[-1]
            elif q.shape[-1] % heads == 0:
                head_dim = q.shape[-1] // heads
            else:
                head_dim = None

            mask_shape = None if mask is None else tuple(mask.shape)

            signature = (
                kind,
                q_shape,
                k_shape,
                v_shape,
                heads,
                head_dim,
                str(q.dtype),
                str(q.device),
                mask_shape,
                skip_reshape,
                skip_output_reshape,
                kwargs.get("scale"),
                kwargs.get("enable_gqa", False),
                kwargs.get("low_precision_attention", True),
            )

            if signature not in seen:
                seen.add(signature)

                logging.info(
                    "[Xe2Attention:%s] "
                    "kind=%s "
                    "q=%s k=%s v=%s "
                    "q_tokens=%s k_tokens=%s "
                    "heads=%s head_dim=%s "
                    "dtype=%s device=%s "
                    "mask=%s "
                    "skip_reshape=%s "
                    "skip_output_reshape=%s "
                    "scale=%s "
                    "enable_gqa=%s "
                    "low_precision_attention=%s",
                    label,
                    kind,
                    q_shape,
                    k_shape,
                    v_shape,
                    q_len,
                    k_len,
                    heads,
                    head_dim,
                    q.dtype,
                    q.device,
                    mask_shape,
                    skip_reshape,
                    skip_output_reshape,
                    kwargs.get("scale"),
                    kwargs.get("enable_gqa", False),
                    kwargs.get("low_precision_attention", True),
                )

            # Important:
            # We're only observing right now. The outer Comfy attention wrapper
            # has already marked this as being inside an attention wrapper, so
            # calling the selected backend here will not recursively invoke
            # this override.
            return comfy_optimized_attention(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        patched.set_model_optimized_attention(inspect_attention)

        logging.info("[Xe2Attention:%s] inspector installed", label)

        return (patched,)


NODE_CLASS_MAPPINGS = {
    "Xe2AttentionInspector": Xe2AttentionInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Xe2AttentionInspector": "Xe2 Attention Inspector",
}