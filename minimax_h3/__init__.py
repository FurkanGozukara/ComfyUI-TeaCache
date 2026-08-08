"""MiniMax H3 speed and low-VRAM nodes (NVlabs Sana sol-engine port) with vendored sol_attn."""

from .optimizer import NODE_CLASS_MAPPINGS as SPEED_CLASS, NODE_DISPLAY_NAME_MAPPINGS as SPEED_DISPLAY
from .low_vram import NODE_CLASS_MAPPINGS as LOW_VRAM_CLASS, NODE_DISPLAY_NAME_MAPPINGS as LOW_VRAM_DISPLAY
from . import lora_support

lora_support.install()

NODE_CLASS_MAPPINGS = {**SPEED_CLASS, **LOW_VRAM_CLASS}
NODE_DISPLAY_NAME_MAPPINGS = {**SPEED_DISPLAY, **LOW_VRAM_DISPLAY}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
