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
"""

import torch

import comfy.lora
import comfy.model_base


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
    if getattr(comfy.lora.model_lora_keys_unet, "_minimax_h3_lora_hook", False):
        return
    original = comfy.lora.model_lora_keys_unet

    def model_lora_keys_unet(model, key_map={}):
        key_map = original(model, key_map)
        if isinstance(model, comfy.model_base.MiniMaxH3):
            _add_minimax_h3_keys(model, model.state_dict(), key_map)
        return key_map

    model_lora_keys_unet._minimax_h3_lora_hook = True
    comfy.lora.model_lora_keys_unet = model_lora_keys_unet
