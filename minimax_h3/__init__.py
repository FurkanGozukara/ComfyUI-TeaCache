"""MiniMax H3 speed optimization nodes (NVlabs Sana sol-engine port) with vendored sol_attn."""

from .optimizer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import lora_support

lora_support.install()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
