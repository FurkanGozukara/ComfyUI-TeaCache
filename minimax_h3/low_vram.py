"""MiniMax H3 low-VRAM execution path.

Peak VRAM while sampling H3 is dominated by two per-block transients over the packed
token sequence, not by the weights. A 5 second 1344x768 clip packs roughly 38k tokens,
and at that length each transformer block briefly holds:

* the fused qkv buffer, ``[S, 3 * heads * head_dim]`` (about 1.6 GB), plus the attention
  kernel's own working set (int8/fp8 copies of q and k and an fp32 accumulator), and
* the feedforward's gate/up projection, ``[S, 2 * ffn_hidden]`` (about 2.2 GB).

Three techniques, in decreasing order of how safely they apply:

1. **Buffer release.** The normed block input is freed as soon as the qkv GEMM has
   consumed it, and the fused qkv buffer is freed before ``out_proj`` allocates. Pure
   bookkeeping, always exact.
2. **Feedforward token chunking.** Rows are independent and the INT8 activation quantizer
   works per row, so chunking the feedforward over tokens is bit-identical. Verified
   against the INT8 and INT8-ConvRot kernels the shipped H3 checkpoints use, and
   end-to-end: a full generation with chunking on decoded to pixel-identical and
   audio-identical output.
3. **Attention head grouping.** Heads are mathematically independent, so per-group
   attention is the same arithmetic -- but only if the attention kernel agrees. Kernels
   pick their tiling and their quantization scales from the tensor they are handed, so a
   head group can round differently than those same heads do inside the whole tensor. It
   is about one bf16 ulp, but a diffusion sampler amplifies that into a visibly different
   (not worse) video.

   Whether it happens depends on the backend *and* the sequence length, which rules out
   deciding it up front: measured on one machine, SageAttention was bit-exact up to 8k
   tokens and not at 16k, while xformers was the other way round -- inexact below 4k,
   exact above. Since the real sequence length is not known when the node is applied, and
   probing at it would itself cost the memory we are trying to save, ``exact_output``
   simply leaves head grouping off. Turn it off to take the saving knowingly.

Measured on one real-geometry H3 block (5376 hidden, 56 heads, 14336 ffn) at 38k packed
tokens on an RTX 5090 with SageAttention: 3.84 GB peak unpatched, 3.30 GB with the exact
path (14% less) and 2.25 GB with head grouping as well (41% less). Neither is slower --
both measured around 17-19% faster than stock, because the smaller working set keeps more
of each matmul in cache.
"""

import logging
import types

import torch

import comfy.model_management as mm
import comfy.ops
import comfy.quant_ops
from comfy.ldm.modules.attention import optimized_attention

try:
    from comfy.ldm.minimax.model import _mod_scale_shift, _mod_gate
except ImportError:  # ComfyUI without MiniMax H3 support
    _mod_scale_shift = _mod_gate = None

# Same key ComfyUI-KJNodes' MiniMax low VRAM nodes use, so the two implementations
# interoperate instead of fighting when both are present in a workflow.
HEAD_GROUPS_KEY = "minimax_head_chunks"
MIN_TOKENS_KEY = "minimax_h3_low_vram_min_tokens"

def _group_sizes(heads, groups):
    return [heads // groups + (1 if i < heads % groups else 0) for i in range(groups)]


def _grouped_attention(q, k, v, heads, head_dim, groups, out, transformer_options):
    """Run attention in head groups, writing each group into its slice of out."""
    start = 0
    for size in _group_sizes(heads, groups):
        stop = start + size
        out[:, start * head_dim:stop * head_dim] = optimized_attention(
            q[:, start:stop], k[:, start:stop], v[:, start:stop], size, mask=None,
            skip_reshape=True, transformer_options=transformer_options).squeeze(0)
        start = stop
    return out


def _attention_forward(self, x, rope_freqs=None, transformer_options={}):
    """Attention.forward, restructured to drop each large buffer at its last use."""
    # The patched block forward hands x over inside a single-item list, so popping it
    # here leaves exactly one reference and the normed block input dies with the qkv GEMM.
    if isinstance(x, list):
        x = x.pop()
    s = x.shape[0]
    device, dtype = x.device, x.dtype
    qkv = self.qkv_proj(x)
    del x
    q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = mm.cast_to(self.q_norm.weight, device=device)
        kw = mm.cast_to(self.k_norm.weight, device=device)
        rot = rope_freqs.shape[-3] * 2
        if mm.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)

    groups = 1
    if isinstance(transformer_options, dict) and s > transformer_options.get(MIN_TOKENS_KEY, 0):
        groups = min(int(transformer_options.get(HEAD_GROUPS_KEY, 1)), self.heads)
    if groups <= 1:
        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True,
                                  transformer_options=transformer_options).squeeze(0)
    else:
        out = _grouped_attention(q, k, v, self.heads, self.head_dim, groups,
                                 torch.empty((s, self.heads * self.head_dim), dtype=dtype, device=device),
                                 transformer_options)
    # free the whole fused qkv buffer before out_proj allocates
    del q, k, v, qkv
    return self.out_proj(out)


# Attention overrides that hook optimized_attention (Sol-Attn sparse attention, the
# KJNodes sage patch) still apply inside this forward rather than being bypassed by it.
_attention_forward._uses_optimized_attention = True
# Marks this forward as willing to take the block's normed input inside a list and free it.
_attention_forward._h3_accepts_list = True


def _make_mlp_forward(chunks, min_tokens):
    def forward(self, x):
        """MLP.forward over token chunks.

        Only one chunk's gate/up activation is live at a time, so the largest buffer in
        the block shrinks by the chunk count. Rows are independent and the INT8
        activation quantizer works per row, so this is bit-identical to the whole-tensor
        call. The chunks are read-only views of x; nothing is written back into it.
        """
        if chunks <= 1 or x.shape[0] <= min_tokens:
            return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")

        out = None
        offset = 0
        for chunk in torch.chunk(x, chunks, dim=0):
            y = comfy.ops.linear_input_act(self.fc2, self.fc1(chunk), "swiglu")
            if out is None:
                out = torch.empty((x.shape[0], y.shape[1]), dtype=y.dtype, device=y.device)
            out[offset:offset + y.shape[0]] = y
            offset += y.shape[0]
            del y
        return out
    return forward


def _block_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
    """DiTBlock.forward that hands its normed attention input over to attn.

    The handoff is checked per call rather than assumed: another node may own this block's
    attn.forward, either because it patched first or because it patched after us, and only
    a forward that advertises _h3_accepts_list knows what to do with the list. Anything
    else gets the plain tensor and simply misses the early release.
    """
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
    if getattr(self.attn.forward, "_h3_accepts_list", False):
        h = [h]
    x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)
    h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
    return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


class MiniMaxH3LowVRAM:
    """Peak VRAM reduction for the MiniMax H3 transformer, exact by default."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "A MiniMax H3 diffusion model."}),
                "enable_low_vram": ("BOOLEAN", {
                    "default": False,
                    "label_on": "ENABLED",
                    "label_off": "DISABLED (normal)",
                    "tooltip": "Master switch. Off passes the model straight through and changes nothing.",
                }),
            },
            "optional": {
                "exact_output": ("BOOLEAN", {
                    "default": True,
                    "label_on": "EXACT (same result, ~15% less)",
                    "label_off": "MAX SAVING (result differs, ~40% less)",
                    "tooltip": "EXACT keeps only what reproduces your normal result bit-for-bit: releasing the big attention buffers at their last use, and chunking the feedforward over tokens. Verified end-to-end - a full generation came out pixel-identical and audio-identical.\n"
                               "MAX SAVING also splits attention into head groups. Heads are mathematically independent, but a kernel picks its tiling and quantization scales from the tensor it is handed, so a head group can round about one bf16 ulp differently - and a sampler amplifies that into a different (not worse) video. Whether it happens depends on your attention backend and the sequence length, so it is offered as a choice rather than guessed at.",
                }),
                "attention_head_groups": ("INT", {"default": 8, "min": 1, "max": 56, "tooltip": "MAX SAVING only. Split each attention call into this many head groups, shrinking the kernel's internal working set by roughly this factor. Gains flatten out past 8. Ignored while exact_output is EXACT."}),
                "feedforward_chunks": ("INT", {"default": 4, "min": 1, "max": 64, "tooltip": "Split the feedforward into this many token chunks. The gate/up activation is the single largest buffer in a block and shrinks by this factor. Always bit-identical. 1 disables chunking."}),
                "min_tokens": ("INT", {"default": 4096, "min": 256, "max": 1048576, "step": 256, "tooltip": "Only split when the packed sequence is longer than this. Short sequences (small canvases, audio-only, single images) gain nothing and skip the extra launches."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "TeaCache/MiniMaxH3"
    TITLE = "MiniMax H3 Low VRAM Optimizations"
    DESCRIPTION = ("Reduces peak VRAM of the MiniMax H3 transformer so a resolution or duration that "
                   "runs out of memory can still generate. Frees the fused qkv buffer and the normed "
                   "block input at their last use and runs the feedforward in token chunks, which is "
                   "bit-identical to the normal path; optionally also splits attention into head "
                   "groups for a larger saving that can shift the result. Measured on an RTX 5090 at "
                   "38k packed tokens: 3.84 GB peak becomes 3.30 GB exact or 2.25 GB at maximum "
                   "saving, and both are slightly faster than stock. Stacks with the MiniMax H3 "
                   "Speed Optimizer. Disabled by default.")

    def apply(self, model, enable_low_vram, exact_output=True, attention_head_groups=8,
              feedforward_chunks=4, min_tokens=4096):
        if not enable_low_vram:
            return (model,)
        if _mod_scale_shift is None or _mod_gate is None:
            raise RuntimeError("This ComfyUI version does not support MiniMax H3, cannot apply MiniMax H3 Low VRAM Optimizations.")

        try:
            diffusion_model = model.get_model_object("diffusion_model")
        except AttributeError:  # not a diffusion model patcher at all
            diffusion_model = None
        if type(diffusion_model).__name__ != "MiniMaxH3Model":
            raise ValueError(f"MiniMaxH3LowVRAM requires a MiniMax H3 model, got {type(diffusion_model).__name__}. "
                             "Connect the MiniMax H3 diffusion model (e.g. minimax_h3_fl2va / ref2va).")
        blocks = getattr(diffusion_model, "blocks", None)
        if not blocks:
            raise ValueError("MiniMaxH3LowVRAM could not find the MiniMax H3 transformer blocks.")

        chunks = max(1, min(feedforward_chunks, 64))
        groups = 1 if exact_output else max(1, min(attention_head_groups, blocks[0].attn.heads))
        if groups <= 1 and chunks <= 1:
            return (model,)

        m = model.clone()
        # Read by the patched attention forward, so head grouping follows the model down
        # either branch of a chain and stays in effect under an attention override.
        transformer_options = m.model_options.setdefault("transformer_options", {})
        transformer_options[HEAD_GROUPS_KEY] = groups
        transformer_options[MIN_TOKENS_KEY] = max(0, min_tokens)

        mlp_forward = _make_mlp_forward(chunks, max(0, min_tokens))
        composed = 0
        for index, block in enumerate(blocks):
            m.add_object_patch(f"diffusion_model.blocks.{index}.forward",
                               types.MethodType(_block_forward, block))
            m.add_object_patch(f"diffusion_model.blocks.{index}.mlp.forward",
                               types.MethodType(mlp_forward, block.mlp))
            attention_key = f"diffusion_model.blocks.{index}.attn.forward"
            if attention_key in m.object_patches:
                # Another attention patch (eg the KJNodes sage patch) already owns this
                # forward; leave it in place. It picks up head grouping from
                # transformer_options, and the block-level buffer release still applies.
                composed += 1
                continue
            m.add_object_patch(attention_key, types.MethodType(_attention_forward, block.attn))

        if composed:
            logging.info(f"[MiniMaxH3LowVRAM] composing with an existing attention patch on {composed} block(s); "
                         "keeping its forward and passing head grouping through.")
        logging.info(f"[MiniMaxH3LowVRAM] enabled on {len(blocks)} blocks above {min_tokens} tokens: "
                     f"{chunks} feedforward chunk(s), "
                     + (f"{groups} attention head group(s) (maximum saving, output may differ from the "
                        "unpatched model)" if groups > 1 else
                        "attention head grouping off (exact output)"))
        return (m,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LowVRAM": MiniMaxH3LowVRAM,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
