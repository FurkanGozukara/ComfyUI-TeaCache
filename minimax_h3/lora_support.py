"""MiniMax H3 LoRA key mapping for the standard ComfyUI lora loaders.

The lightx2v Minimax-h3-Turbo distill LoRA (minimax_h3_fl2v_turbo_4step_v0.1)
ships diffusers/PEFT key names (transformer_blocks.N.attn.to_q.lora_A.default.weight)
in the original MiniMax weight layout. ComfyUI parses that adapter format natively
but has no key mapping for the MiniMax H3 architecture. Besides the fused-qkv
offsets, one layout change from the ComfyUI checkpoint conversion must be followed:
the original swiglu stores fc1 as [value; gate] (gate = second half, as in the
lightx2v reference implementation) while ComfyUI's swiglu gates the first half,
so the fc1 delta halves are swapped to match.

Wrapping comfy.lora.model_lora_keys_unet here makes the plain LoraLoader /
LoraLoaderModelOnly nodes (and therefore SwarmUI's lora handling) apply these
LoRAs to every H3 checkpoint variant, including the quantized pruned ones.

These PEFT exports carry no alpha keys, so ComfyUI would apply them at
alpha == rank while the distill's calibrated application scale is rank/8
(strength 1.0 fries the model, 0.125 is the sweet spot). convert_lora is
wrapped to inject alpha = rank/8 per module, so a user-facing strength of
1.0 lands on the calibrated scale and the strength dial stays intuitive.
"""

import torch

import comfy.lora
import comfy.lora_convert
import comfy.model_base

# H3-specific PEFT-export signature: the token refiner module path only exists on
# this architecture, and the ".default" adapter suffix marks a raw PEFT dump.
_PEFT_MARKER = "token_refiner.refiner_blocks.0.attn.to_q.lora_A.default.weight"
_TRAINED_ALPHA_RATIO = 0.125  # effective trained alpha / rank of the lightx2v turbo exports


def _inject_peft_alpha(sd):
    """Add alpha = rank/8 to alpha-less MiniMax H3 PEFT loras.

    Only fires on the H3 PEFT signature and never overrides explicit alphas,
    so every other lora format and architecture passes through untouched.
    """
    if _PEFT_MARKER not in sd:
        return sd
    if any(k.endswith(".alpha") for k in sd):
        return sd
    out = dict(sd)
    suffix = ".lora_A.default.weight"
    for k, v in sd.items():
        if k.endswith(suffix):
            out[k[:-len(suffix)] + ".alpha"] = torch.tensor(v.shape[0] * _TRAINED_ALPHA_RATIO)
    return out


def _swiglu_swap(diff):
    half = diff.shape[0] // 2
    return torch.cat((diff[half:], diff[:half]), dim=0)


def _add_minimax_h3_keys(model, sd, key_map):
    for k in sd:
        if not (k.startswith("diffusion_model.") and k.endswith(".weight")):
            continue
        base = k[len("diffusion_model."):-len(".weight")]
        # diffusers block naming used by the lightx2v/PEFT export
        diff = base.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
        if diff.startswith("blocks."):
            diff = "transformer_blocks." + diff[len("blocks."):]
        if base.endswith(".attn.qkv_proj"):
            stem = diff[:-len(".qkv_proj")]
            rows = sd[k].shape[0] // 3
            for i, name in enumerate(("to_q", "to_k", "to_v")):
                key_map["{}.{}".format(stem, name)] = (k, (0, rows * i, rows))
        elif base.endswith(".attn.out_proj"):
            key_map[diff[:-len(".out_proj")] + ".to_out.0"] = k
        elif base.endswith(".mlp.fc1"):
            key_map[diff[:-len(".mlp.fc1")] + ".ff.net.0.proj"] = (k, None, _swiglu_swap)
        elif base.endswith(".mlp.fc2"):
            key_map[diff[:-len(".mlp.fc2")] + ".ff.net.2"] = k
    return key_map


def install():
    if not getattr(comfy.lora.model_lora_keys_unet, "_minimax_h3_lora_hook", False):
        original = comfy.lora.model_lora_keys_unet

        def model_lora_keys_unet(model, key_map={}):
            key_map = original(model, key_map)
            if isinstance(model, comfy.model_base.MiniMaxH3):
                _add_minimax_h3_keys(model, model.state_dict(), key_map)
            return key_map

        model_lora_keys_unet._minimax_h3_lora_hook = True
        comfy.lora.model_lora_keys_unet = model_lora_keys_unet

    if not getattr(comfy.lora_convert.convert_lora, "_minimax_h3_lora_hook", False):
        original_convert = comfy.lora_convert.convert_lora

        def convert_lora(sd):
            return original_convert(_inject_peft_alpha(sd))

        convert_lora._minimax_h3_lora_hook = True
        comfy.lora_convert.convert_lora = convert_lora
