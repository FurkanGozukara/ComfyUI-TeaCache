"""MiniMax H3 masked video edit and Time-to-Move (cut and drag) motion control.

Both map pixel-frame masks onto H3's video latent grid. The H3 video VAE (comfy/ldm/minimax/vae.py) encodes
17-frame clips; its causal encoder halves time twice (time_down 1,2,2 -> 4x), so with 3 frames of implicit
pre-padding each clip becomes 5 latents that first see 1, 4, 4, 4, 4 source frames (latent k of a clip has seen
frames up to 4k), and the last 3 latents of the video are dropped: 17k+5 frames <-> 5k+2 latents, 16x spatial.
"""

import itertools
import logging

import torch
import torch.nn.functional as F

import comfy.nested_tensor
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN

TTM_KEY = "MiniMaxH3TimeToMove"


def latent_frame_starts(latent_t):
    """First source frame of every video latent plus the end of the last one (37 latents -> ... 119, 120, 124)."""
    return [0] + list(itertools.accumulate(FRAME_PER_TOKEN[k % 5] for k in range(latent_t)))


def _fit_frames(x, n):
    """Trim, or repeat the last frame, so x has n frames."""
    if x.shape[0] >= n:
        return x[:n]
    return torch.cat([x, x[-1:].expand(n - x.shape[0], *x.shape[1:])])


def edit_cell_mask(mask, latent_t, lat_h, lat_w, ratio, grow, hold):
    """[F,H,W] edit mask (white = regenerate) -> [T,1,h,w] latent cell mask, 1 where any covered pixel is edited."""
    starts = latent_frame_starts(latent_t)
    m = _fit_frames(mask.reshape(-1, *mask.shape[-2:]), starts[-1]) > 0.5
    # every step is a max filter, so holding +-hold frames and pooling each latent's frames can come first
    m = torch.stack([m[max(0, a - hold):b + hold].any(dim=0) for a, b in zip(starts, starts[1:])])
    m = (F.interpolate(m.unsqueeze(1).float(), size=(lat_h * ratio, lat_w * ratio), mode="bilinear") > 0.5).float()
    # dilate by grow canvas pixels and keep any covered ratio x ratio cell, as one separable max pool
    k = ratio + 2 * grow
    m = F.max_pool2d(m, (1, k), stride=(1, ratio), padding=(0, grow))
    return F.max_pool2d(m, (k, 1), stride=(ratio, 1), padding=(grow, 0))


def ttm_keep_mask(mask, latent_t, lat_h, lat_w):
    """[F,H,W] subject mask (white = hold) -> [1,1,T,h,w] denoise mask, 0 where the subject is held.

    Each latent samples its anchor, the newest source frame it encodes: clip_start + 4 * (j % 5)."""
    starts = latent_frame_starts(latent_t)
    m = mask.reshape(-1, *mask.shape[-2:]).float().clamp(0.0, 1.0)
    if m.shape[0] != starts[-1]:
        logging.warning("[MiniMaxH3TimeToMove] the mask has %d frames but the reference latents cover %d; "
                        "frames past the mask's end reuse its last frame.", m.shape[0], starts[-1])
    anchors = [min(b - 1, m.shape[0] - 1) for b in starts[1:]]
    m = F.interpolate(m[anchors].unsqueeze(1), size=(lat_h, lat_w), mode="nearest")
    return (1.0 - m).movedim(0, 1).unsqueeze(0)


class MiniMaxH3MaskedVideoEdit:
    """Encode a real clip into the video stream and regenerate only the masked region; the rest is held."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT", {"tooltip": "The video+audio latent from a MiniMax H3 conditioning node, sized to the output canvas."}),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE."}),
                "grow_pixels": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Grow the edit mask by this many pixels on the output canvas, so the new content has room at its edges."}),
                "hold_frames": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Also regenerate the masked area this many frames before and after each masked frame."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "The source video frames. Any size: they are stretched to the canvas. Not connected: the latent passes through unchanged."}),
                "mask": ("MASK", {"tooltip": "White = regenerate, black = keep the source. One frame applies to the whole clip. Not connected: the latent passes through unchanged."}),
                "edge_feather": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Soften the edge of the composite mask by this many pixels."}),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("av_latent", "composite_mask", "canvas_frames", "report")
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/VideoEdit"
    TITLE = "MiniMax H3 Masked Video Edit Latent"
    DESCRIPTION = ("Turn an H3 generation into a masked edit of a real clip: only the masked region is regenerated. "
                   "Feed composite_mask and canvas_frames to MiniMax H3 Masked Edit Composite after decoding.")

    def run(self, av_latent, vae, grow_pixels, hold_frames, images=None, mask=None, edge_feather=8):
        if images is None or mask is None:
            return (av_latent, None, None, "images or mask not connected: latent passed through unchanged")
        samples = av_latent["samples"]
        if not samples.is_nested:
            raise ValueError("Expected a MiniMax H3 video+audio latent. Feed the LATENT output of a MiniMax H3 conditioning node.")
        members = list(samples.unbind())
        video = members[0]
        latent_t, lat_h, lat_w = video.shape[-3:]
        ratio = vae.spacial_compression_encode()
        width, height = lat_w * ratio, lat_h * ratio
        starts = latent_frame_starts(latent_t)
        frames = starts[-1]

        note = ""
        if images.shape[0] != frames:
            note = (f"WARNING: {images.shape[0]} source frames but the latent holds {frames} (H3 uses 17k+5 frames); "
                    f"the source was {'trimmed' if images.shape[0] > frames else 'padded with its last frame'}.\n")
            logging.warning("[MiniMaxH3VideoEdit] %s", note.strip())
        canvas = _fit_frames(images[..., :3], frames).movedim(-1, 1)
        method = "area" if canvas.shape[-2] * canvas.shape[-1] > width * height else "bicubic"
        canvas = comfy.utils.common_upscale(canvas, width, height, method, "disabled").clamp(0.0, 1.0).movedim(1, -1)
        members[0] = vae.encode(canvas).to(video)

        cells = edit_cell_mask(mask, latent_t, lat_h, lat_w, ratio, grow_pixels, hold_frames)
        prev = av_latent.get("noise_mask")
        audio_kept = prev is not None and prev.is_nested
        audio_mask = prev.unbind()[1] if audio_kept else torch.ones_like(members[1])
        out = dict(av_latent)
        out["samples"] = comfy.nested_tensor.NestedTensor(members)
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((cells.movedim(0, 1).unsqueeze(0).to(video.device), audio_mask))

        owner = [j for j, (a, b) in enumerate(zip(starts, starts[1:])) for _ in range(a, b)]
        composite = F.interpolate(cells[owner], size=(height, width), mode="nearest")
        if edge_feather > 0:
            composite = F.avg_pool2d(composite, 2 * edge_feather + 1, stride=1, padding=edge_feather, count_include_pad=False)

        report = (f"masked edit: {cells.mean().item():.0%} of the video latent regenerates "
                  f"({latent_t} latent frames of {lat_w}x{lat_h} cells, {ratio} px each)\n"
                  f"canvas {width}x{height}, {frames} frames, grow {grow_pixels} px, hold +-{hold_frames} frames, "
                  f"audio mask {'kept from the input latent' if audio_kept else 'ones (audio is generated)'}\n{note}")
        return (out, composite.squeeze(1), canvas, report)


class MiniMaxH3MaskedEditComposite:
    """Paste the decoded edit over the source frames through the composite mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The decoded video."}),
            },
            "optional": {
                "canvas_frames": ("IMAGE", {"tooltip": "canvas_frames from MiniMax H3 Masked Video Edit Latent. Not connected: the decoded video passes through."}),
                "composite_mask": ("MASK", {"tooltip": "composite_mask from MiniMax H3 Masked Video Edit Latent. Not connected: the decoded video passes through."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/VideoEdit"
    TITLE = "MiniMax H3 Masked Edit Composite"
    DESCRIPTION = "Keep the untouched source outside the edited area: decoded video inside the mask, source frames outside."

    def run(self, images, canvas_frames=None, composite_mask=None):
        if canvas_frames is None or composite_mask is None:
            return (images,)
        n, h, w = images.shape[:3]
        if canvas_frames.shape[0] != n or composite_mask.shape[0] != n:
            logging.warning("[MiniMaxH3VideoEdit] composite: %d decoded frames, %d canvas frames, %d mask frames; "
                            "trimming or repeating the last frame to match the decoded video.",
                            n, canvas_frames.shape[0], composite_mask.shape[0])
        source = _fit_frames(canvas_frames[..., :3], n).to(images)
        m = _fit_frames(composite_mask.reshape(-1, *composite_mask.shape[-2:]), n).to(images)
        if source.shape[1:3] != (h, w):
            source = comfy.utils.common_upscale(source.movedim(-1, 1), w, h, "bicubic", "disabled").clamp(0.0, 1.0).movedim(1, -1)
        if m.shape[-2:] != (h, w):
            m = F.interpolate(m.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
        return (torch.lerp(source, images[..., :3], m.unsqueeze(-1)),)


class _TimeToMove:
    """OUTER_SAMPLE wrapper: pins the video to the reference latents, then only the subject, then lets go."""

    def __init__(self, reference, keep, start_step, end_step):
        self.reference, self.keep = reference, keep
        self.start_step, self.end_step = start_step, end_step

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None,
                 disable_pbar=False, seed=None, latent_shapes=None):
        args = (callback, disable_pbar, seed)
        if not torch.any(noise):  # a follow-up sampler continuing without new noise
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, *args, latent_shapes=latent_shapes)
        if latent_shapes is None or len(latent_shapes) < 2:
            logging.warning("[MiniMaxH3TimeToMove] needs a MiniMax H3 video+audio latent; sampling without Time-to-Move.")
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, *args, latent_shapes=latent_shapes)
        if tuple(self.reference.shape[1:]) != tuple(latent_shapes[0][1:]):
            raise ValueError(f"Time-to-Move reference latents {tuple(self.reference.shape)} do not match the sampled video "
                             f"latent {tuple(latent_shapes[0])}: encode the cut-and-drag video at the generation size and length.")
        guider = executor.class_obj
        streams = comfy.utils.unpack_latents(latent_image, latent_shapes)
        streams[0] = self.reference.to(streams[0])
        latent_image = comfy.utils.pack_latents(streams)[0]
        base = denoise_mask if denoise_mask is not None else torch.ones_like(latent_image, device=guider.model_patcher.load_device)

        def video_times(keep):
            masks = comfy.utils.unpack_latents(base, latent_shapes)
            masks[0] = masks[0] * keep.to(masks[0])
            return comfy.utils.pack_latents(masks)[0]

        # (sigma floor, packed denoise mask), highest floor first; holding during the step at sigmas[k]
        # leaves the latent held at sigmas[k + 1]
        last = len(sigmas) - 1
        phases = []
        if self.start_step > 0:
            phases.append((float(sigmas[min(self.start_step - 1, last)]), video_times(torch.zeros_like(self.keep))))
        if self.end_step > self.start_step:
            phases.append((float(sigmas[min(self.end_step - 1, last)]), video_times(self.keep)))
        phases.append((float("-inf"), base))
        token_masks = {}

        def phase_at(sigma):
            return next(i for i, (floor, _) in enumerate(phases) if float(sigma.max()) >= floor)

        def denoise_mask_function(sigma, mask, extra_options):
            return phases[phase_at(sigma)][1]

        def apply_model(apply_executor, x, t, *apply_args, **kwargs):
            # keep H3's per-token timestep clock in step with the sampler-side hold
            i = phase_at(t)
            if i not in token_masks:
                token_masks[i] = apply_executor.class_obj._denoise_mask_values(phases[i][1], latent_shapes)
            kwargs.pop("denoise_mask", None)
            kwargs.pop("audio_denoise_mask", None)
            return apply_executor(x, t, *apply_args, **kwargs, **token_masks[i])

        guider.model_options["denoise_mask_function"] = denoise_mask_function
        comfy.patcher_extension.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, TTM_KEY, apply_model,
                                                     guider.model_options, is_model_options=True)
        return executor(noise, latent_image, sampler, sigmas, base, *args, latent_shapes=latent_shapes)


class MiniMaxH3TimeToMove:
    """Time-to-Move motion control: the generation starts from a cut-and-drag video and keeps the dragged subject."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "A MiniMax H3 model (FL2VA image-to-video)."}),
                "reference_latents": ("LATENT", {"tooltip": "The cut-and-drag video encoded with the H3 video VAE at the generation size and length."}),
                "mask": ("MASK", {"tooltip": "White = the dragged subject to hold, one mask per frame of the cut-and-drag video."}),
                "start_step": ("INT", {"default": 1, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "The whole video follows the reference for this many steps."}),
                "end_step": ("INT", {"default": 2, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "After start_step, only the subject follows the reference until this step; then everything is free."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/VideoEdit"
    TITLE = "MiniMax H3 Time-to-Move (cut and drag)"
    DESCRIPTION = "Make an H3 image-to-video generation follow a rough cut-and-drag animation of the subject. Audio is never held."

    def run(self, model, reference_latents, mask, start_step, end_step):
        samples = reference_latents["samples"]
        reference = samples.unbind()[0] if samples.is_nested else samples
        keep = ttm_keep_mask(mask, *reference.shape[-3:])
        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, TTM_KEY, _TimeToMove(reference, keep, start_step, end_step))
        return (m,)


class MiniMaxH3PathFromSubject:
    """Shift a drawn path so it starts at the subject's centre: the user draws only the motion's shape and length."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True, "tooltip": "Path points from the Spline Editor (coord_str)."}),
                "mask": ("MASK", {"tooltip": "The subject mask (first frame is used)."}),
                "start": (["at the subject", "as drawn"], {"default": "at the subject",
                    "tooltip": "'at the subject' moves the whole path so its first point is the centre of the subject; "
                               "'as drawn' keeps your points where you drew them."}),
            },
            "optional": {
                "frames": ("INT", {"default": 0, "min": 0, "max": 100000, "forceInput": True,
                    "tooltip": "Resample the path to exactly this many frames (the video length), keeping its timing. 0 keeps the points as they are."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coordinates",)
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/VideoEdit"
    TITLE = "MiniMax H3 Drag Path From Subject"

    @staticmethod
    def _resample(track, n):
        if n <= 0 or len(track) == n or not track:
            return track
        if len(track) == 1:
            return track * n
        out = []
        for i in range(n):
            f = i * (len(track) - 1) / (n - 1)
            a, b = track[int(f)], track[min(int(f) + 1, len(track) - 1)]
            w = f - int(f)
            out.append({"x": round(a["x"] + (b["x"] - a["x"]) * w), "y": round(a["y"] + (b["y"] - a["y"]) * w)})
        return out

    def run(self, coordinates, mask, start, frames=0):
        import json
        tracks = json.loads(coordinates.replace("'", '"'))
        single = bool(tracks) and isinstance(tracks[0], dict)
        tracks = [tracks] if single else tracks
        m = mask.reshape(-1, *mask.shape[-2:])[0] > 0.5
        if start == "at the subject" and bool(m.any()) and tracks and tracks[0]:
            ys, xs = torch.nonzero(m, as_tuple=True)
            cx, cy = (int(xs.min()) + int(xs.max())) / 2, (int(ys.min()) + int(ys.max())) / 2
            dx, dy = cx - tracks[0][0]["x"], cy - tracks[0][0]["y"]
            tracks = [[{"x": round(p["x"] + dx), "y": round(p["y"] + dy)} for p in t] for t in tracks]
        tracks = [self._resample(t, int(frames or 0)) for t in tracks]
        return (json.dumps(tracks[0] if single else tracks),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MaskedVideoEdit": MiniMaxH3MaskedVideoEdit,
    "MiniMaxH3MaskedEditComposite": MiniMaxH3MaskedEditComposite,
    "MiniMaxH3TimeToMove": MiniMaxH3TimeToMove,
    "MiniMaxH3PathFromSubject": MiniMaxH3PathFromSubject,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
