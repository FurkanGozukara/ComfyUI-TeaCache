"""MiniMax H3 live preview: an animated latent-to-RGB view of the video stream while sampling.

Every sampler step already hands ComfyUI's preview callback the model's x0 estimate. For H3
that is a nested (video, audio) latent whose video stream is ``[B, 24, T, H/16, W/16]``, and
``comfy.latent_formats.MiniMaxH3Video`` ships a 24 -> 3 linear projection for it. Applying that
projection to *every* latent frame costs one tiny matmul on tensors that are already resident
(about a megabyte of transient output for a 5 s clip, allocated between steps when the
transformer's activations are gone, so peak VRAM does not move) and one sub-millisecond copy of
a few hundred kilobytes to the CPU. Everything else -- tiling the frames into a sprite sheet,
JPEG encoding, the websocket send -- runs on a background thread, so the sampler thread never
waits on the preview. There is no extra model to load: this is deliberately not a tiny-VAE path.

The frames reach the frontend as one JPEG sprite sheet per step through a custom websocket
event; ``web/js/minimax_h3_live_preview.js`` plays them back on the ``MiniMax H3 Live Preview``
node at the latent frame rate (24 fps * 5 / 17, H3 codes 17 pixel frames per 5 latent frames).

The display node is a typed passthrough (VIDEO in the video presets, IMAGE in the image preset)
so it can sit right before the output node. It does not need to be in the model chain: the
preview is produced by wrapping ``latent_preview.prepare_callback``, and the wrapper only acts
when the sampled model uses the MiniMax H3 latent format and the running prompt contains an
enabled ``MiniMaxH3LivePreview`` node. Any other model, or a prompt without the node, goes
through the stock callback untouched.
"""

import base64
import io
import logging
import math
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import comfy.latent_formats
import latent_preview
from comfy_api.latest import io as comfy_io
from comfy_execution.utils import get_executing_context
from server import PromptServer

# ComfyUI builds without MiniMax H3 have no H3 latent format: nothing to preview, nothing to hook.
H3_LATENT_FORMAT = getattr(comfy.latent_formats, "MiniMaxH3Video", None)

NODE_ID = "MiniMaxH3LivePreview"
EVENT = "minimax_h3_live_preview"
LATENT_FPS = 24 * 5 / 17
MAX_FRAMES = 128
JPEG_QUALITY = 90


class _SheetEncoder:
    """Background encoder holding at most one pending job: a newer step replaces an unencoded
    older one, so a slow client can never queue up work behind the sampler."""

    def __init__(self):
        self.lock = threading.Condition()
        self.pending = None
        self.thread = None

    def submit(self, job):
        with self.lock:
            self.pending = job
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name="minimax_h3_live_preview", daemon=True)
                self.thread.start()
            self.lock.notify()

    def _run(self):
        while True:
            with self.lock:
                while self.pending is None:
                    self.lock.wait()
                job, self.pending = self.pending, None
            try:
                job()
            except Exception:
                logging.exception("[MiniMaxH3LivePreview] preview encode failed")


_encoder = _SheetEncoder()


def _sprite_sheet(frames):
    """[T, H, W, 3] uint8 -> (JPEG bytes, cols, rows). Roughly square sheet, row-major."""
    count, height, width, _ = frames.shape
    cols = max(1, min(count, math.ceil(math.sqrt(count * height / width))))
    rows = math.ceil(count / cols)
    sheet = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    for i in range(count):
        y, x = divmod(i, cols)
        sheet[y * height:(y + 1) * height, x * width:(x + 1) * width] = frames[i]
    buf = io.BytesIO()
    Image.fromarray(sheet).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), cols, rows


def _preview_node_ids():
    """Ids of enabled MiniMaxH3LivePreview nodes in the prompt that is executing right now."""
    server = PromptServer.instance
    context = get_executing_context()
    prompt_id = context.prompt_id if context is not None else server.last_prompt_id
    prompt = None
    for item in list(server.prompt_queue.currently_running.values()):
        if item[1] == prompt_id:
            prompt = item[2]
            break
    if prompt is None:
        return []
    ids = []
    for node_id, node in prompt.items():
        if node.get("class_type") != NODE_ID:
            continue
        # a linked switch arrives as [node, slot]; only a literal False disables the node
        if node.get("inputs", {}).get("enable_preview", True) is False:
            continue
        ids.append(node_id)
    return ids


class _LivePreview:
    """Per-sampling-run state: the projection matrix on the sampling device and step timing."""

    def __init__(self, node_ids, latent_format, sampler_node_id):
        self.node_ids = node_ids
        self.sampler_node_id = sampler_node_id
        self.factors = torch.tensor(latent_format.latent_rgb_factors).transpose(0, 1)
        self.bias = torch.tensor(latent_format.latent_rgb_factors_bias)
        self.last_time = None
        self.failed = False

    def __call__(self, step, x0, total_steps):
        if self.failed:
            return
        try:
            self._push(step, x0, total_steps)
        except Exception:
            self.failed = True
            logging.exception("[MiniMaxH3LivePreview] live preview disabled for this run")

    def _push(self, step, x0, total_steps):
        video = x0.tensors[0] if x0.is_nested else x0
        if video.ndim != 5:
            return
        video = video[0]  # [24, T, H, W]
        frame_count = video.shape[1]
        if frame_count > MAX_FRAMES:
            keep = [round(i * (frame_count - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]
            video = video[:, keep]
        if self.factors.device != video.device or self.factors.dtype != video.dtype:
            self.factors = self.factors.to(device=video.device, dtype=video.dtype)
            self.bias = self.bias.to(device=video.device, dtype=video.dtype)
        rgb = F.linear(video.movedim(0, -1), self.factors, bias=self.bias)  # [T, H, W, 3]
        rgb = rgb.add_(1.0).mul_(127.5).clamp_(0, 255)
        frames = rgb.to(device="cpu", dtype=torch.uint8).numpy()

        now = time.perf_counter()
        step_ms = None if self.last_time is None else (now - self.last_time) * 1000.0
        self.last_time = now
        payload = {
            "nodes": self.node_ids,
            "sampler": self.sampler_node_id,
            "step": step + 1,
            "total": total_steps,
            "latent_frames": frame_count,
            "shown_frames": int(frames.shape[0]),
            "video_frames": (frame_count - 2) // 5 * 17 + 5 if frame_count >= 2 else frame_count,
            "fps": LATENT_FPS,
            "w": int(frames.shape[2]),
            "h": int(frames.shape[1]),
            "step_ms": step_ms,
        }
        client_id = PromptServer.instance.client_id

        def encode_and_send():
            jpeg, cols, rows = _sprite_sheet(frames)
            payload["cols"] = cols
            payload["rows"] = rows
            payload["sheet"] = base64.b64encode(jpeg).decode("ascii")
            PromptServer.instance.send_sync(EVENT, payload, client_id)

        _encoder.submit(encode_and_send)


_stock_prepare_callback = latent_preview.prepare_callback


def prepare_callback(model, steps, x0_output_dict=None, *args, **kwargs):
    callback = _stock_prepare_callback(model, steps, x0_output_dict, *args, **kwargs)
    if not isinstance(model.model.latent_format, H3_LATENT_FORMAT):
        return callback
    node_ids = _preview_node_ids()
    if not node_ids:
        return callback
    context = get_executing_context()
    preview = _LivePreview(node_ids, model.model.latent_format,
                           context.node_id if context is not None else None)

    def live_callback(step, x0, x, total_steps):
        callback(step, x0, x, total_steps)
        preview(step, x0, total_steps)

    return live_callback


if H3_LATENT_FORMAT is not None and not getattr(latent_preview.prepare_callback, "_minimax_h3_live_preview", False):
    prepare_callback._minimax_h3_live_preview = True
    latent_preview.prepare_callback = prepare_callback


class MiniMaxH3LivePreview(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = comfy_io.MatchType.Template("passthrough")
        return comfy_io.Schema(
            node_id=NODE_ID,
            display_name="MiniMax H3 Live Preview",
            category="TeaCache/MiniMaxH3",
            description=("Shows an animated latent preview of the MiniMax H3 video stream on this node "
                         "after every sampler step. It is a 24->3 projection of latents that are already "
                         "on the GPU: no extra model, no peak VRAM change, and the JPEG encoding runs on a "
                         "background thread so sampling speed is unaffected. Connect anything (VIDEO or "
                         "IMAGE) through it right before the output node; the value passes through "
                         "unchanged. Bypassing the node or switching it off restores the stock behaviour."),
            inputs=[
                comfy_io.MatchType.Input("value", template=template,
                                         tooltip="Passed through unchanged. Route the final VIDEO (or IMAGE) through here so the node sits right before the output node."),
                comfy_io.Boolean.Input("enable_preview", default=True, label_on="ENABLED", label_off="DISABLED",
                                       tooltip="Master switch. Off passes the value straight through and sends no previews."),
            ],
            outputs=[comfy_io.MatchType.Output(template=template, display_name="value")],
        )

    @classmethod
    def execute(cls, value, enable_preview=True) -> comfy_io.NodeOutput:
        return comfy_io.NodeOutput(value)


NODE_CLASS_MAPPINGS = {NODE_ID: MiniMaxH3LivePreview} if H3_LATENT_FORMAT is not None else {}

NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "MiniMax H3 Live Preview"} if H3_LATENT_FORMAT is not None else {}
