"""MiniMax H3 face inpainting nodes: YOLO face tracking, crop, AV-latent img2img and stitch-back.

Second-pass face refinement for MiniMax H3 videos: detect and track the subject's face in
every generated frame, crop it to a stabilised constant-size canvas, regenerate the crops
with the same H3 model as ordinary img2img (audio stream locked so speech and lipsync are
preserved), re-align each refined crop onto the source face geometry with dense optical flow
(geometry lock: no per-frame shaking / tilting), then warp the refined faces back with
feathered, colour-matched compositing.

Face detection is YOLO-only (models from the shared "yolov8" model folder, e.g.
yolov9e-face-lindevs.pt). Identity tracking through crowds optionally uses InsightFace
when it is installed; it degrades gracefully to continuity tracking otherwise.

Adapted from Carasibana/ComfyUI-H3-FaceRefine (MIT License,
https://github.com/Carasibana/ComfyUI-H3-FaceRefine) for the SECourses MiniMax H3 presets.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import comfy.nested_tensor
import folder_paths

DEFAULT_FACE_REFINEMENT_PROMPT = (
    "Preserve the exact same identity, expression, head pose, and facial proportions. "
    "Resolve natural coherent eyes, skin texture, beard strands, and hair detail. "
    "No identity change, beautification, or facial reshaping."
)


def _append_face_refinement_prompt(prompt, refinement_prompt):
    base = str(prompt or "").rstrip()
    addition = str(refinement_prompt or "").strip()
    if not addition or addition.casefold() in base.casefold():
        return base
    return f"{base}\n\n{addition}" if base else addition


def _face_heights_from_transform(transform):
    boxes = transform.get("boxes", [])
    stored = transform.get("face_height_src")
    if stored is not None and len(stored) == len(boxes):
        return np.asarray(stored, dtype=np.float64)
    crop_factor = float(transform.get("crop_factor", 3.0)) or 3.0
    return np.asarray([box[3] / crop_factor for box in boxes], dtype=np.float64)


def _size_aware_stitch_weights(face_heights, full_refine_px, passthrough_px):
    heights = np.asarray(face_heights, dtype=np.float64)
    lo = float(full_refine_px)
    hi = max(float(passthrough_px), lo + 1e-6)
    t = np.clip((heights - lo) / (hi - lo), 0.0, 1.0)
    smoothstep = t * t * (3.0 - 2.0 * t)
    return 1.0 - smoothstep


def _warp_similarity_batch(images: torch.Tensor, matrices: np.ndarray) -> torch.Tensor:
    """Warp images with source-to-destination pixel transforms using grid_sample."""
    import torch.nn.functional as F

    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("Expected IMAGE data shaped [B,H,W,C].")
    count, height, width = images.shape[:3]
    if len(matrices) != count:
        raise ValueError(f"Expected {count} alignment matrices, received {len(matrices)}.")

    pixel_from_norm = np.asarray([
        [width * 0.5, 0.0, (width - 1.0) * 0.5],
        [0.0, height * 0.5, (height - 1.0) * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    norm_from_pixel = np.linalg.inv(pixel_from_norm)
    theta = []
    for matrix in matrices:
        source_to_dest = np.eye(3, dtype=np.float64)
        source_to_dest[:2] = np.asarray(matrix, dtype=np.float64)
        dest_to_source = np.linalg.inv(source_to_dest)
        theta.append((norm_from_pixel @ dest_to_source @ pixel_from_norm)[:2])

    result = images.clone()
    import comfy.model_management as mm
    try:
        device = mm.get_torch_device()
    except Exception:
        device = images.device
    chunk = 8
    for start in range(0, count, chunk):
        mm.throw_exception_if_processing_interrupted()
        end = min(start + chunk, count)
        rgb = images[start:end, ..., :3].movedim(-1, 1).to(
            device=device, dtype=torch.float32)
        theta_t = torch.as_tensor(
            np.stack(theta[start:end]), device=device, dtype=torch.float32)
        grid = F.affine_grid(theta_t, rgb.shape, align_corners=False)
        warped = F.grid_sample(
            rgb, grid, mode="bilinear", padding_mode="border", align_corners=False)
        result[start:end, ..., :3] = warped.movedim(1, -1).to(
            device=result.device, dtype=images.dtype)
    return result


def _face_alignment_comparison(original, refined, aligned, panel_size=256):
    """Build a memory-bounded original/raw/aligned strip for visual QA."""
    import torch.nn.functional as F

    panels = []
    for images in (original, refined, aligned):
        rgb = images[..., :3].movedim(-1, 1).float()
        if rgb.shape[-2:] != (panel_size, panel_size):
            rgb = F.interpolate(
                rgb, size=(panel_size, panel_size), mode="bilinear", align_corners=False)
        panels.append(rgb.movedim(1, -1).to(images.dtype))
    return torch.cat(panels, dim=2)

# ----------------------------------------------------------------------------
# detector helpers
# ----------------------------------------------------------------------------

_DETECTOR_CACHE: dict = {}


def _register_yolo_folder():
    """Make sure the shared "yolov8" model folder exists in folder_paths.

    SwarmUI's ExtraNodes register the same name; when they are absent this keeps the
    detector list working from ComfyUI/models/yolov8 plus any extra_model_paths entry.
    """
    try:
        if "yolov8" not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["yolov8"] = (
                [os.path.join(folder_paths.models_dir, "yolov8")],
                folder_paths.supported_pt_extensions,
            )
        else:
            existing = folder_paths.folder_names_and_paths["yolov8"]
            folder_paths.folder_names_and_paths["yolov8"] = (
                existing[0], folder_paths.supported_pt_extensions)
    except Exception:
        pass


_register_yolo_folder()

_DETECTOR_FOLDERS = ("yolov8", "ultralytics_bbox", "ultralytics")


def _detector_list() -> list:
    """YOLO detectors from the shared yolov8 folder plus Impact-style ultralytics folders."""
    _register_yolo_folder()
    names, seen = [], set()
    for key in _DETECTOR_FOLDERS:
        try:
            for name in folder_paths.get_filename_list(key):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        except Exception:
            pass
    preferred = "yolov9e-face-lindevs.pt"
    if preferred in names:
        names.remove(preferred)
        names.insert(0, preferred)
    return names or [preferred]


def _load_detector(name: str):
    if name in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[name]
    path = None
    for key in _DETECTOR_FOLDERS:
        try:
            path = folder_paths.get_full_path(key, name)
        except Exception:
            path = None
        if path:
            break
    if path is None:  # fall back to the standard models tree
        base = getattr(folder_paths, "models_dir", "models")
        for sub in ("yolov8", "ultralytics/bbox", "ultralytics", "ultralytics/segm"):
            cand = os.path.join(base, *sub.split("/"), name)
            if os.path.exists(cand):
                path = cand
                break
    if path is None:
        raise FileNotFoundError(
            f"Face detector '{name}' was not found in the yolov8 / ultralytics model folders. "
            "Place a YOLO face model (e.g. yolov9e-face-lindevs.pt) in models/yolov8."
        )
    from ultralytics import YOLO

    model = YOLO(path)
    _DETECTOR_CACHE[name] = model
    return model


# ----------------------------------------------------------------------------
# identity (InsightFace) helpers - optional, used only for multi-face tracking
# ----------------------------------------------------------------------------

_REC_CACHE: dict = {}


def _face_recogniser(pack: str = "buffalo_l"):
    if pack in _REC_CACHE:
        return _REC_CACHE[pack]
    import insightface

    root = os.path.join(getattr(folder_paths, "models_dir", "models"), "insightface")
    app = insightface.app.FaceAnalysis(
        name=pack, root=root, allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    _REC_CACHE[pack] = app
    return app


def _face_landmark_detector(pack: str = "buffalo_l"):
    """Load InsightFace detection plus its dense 2D landmark model."""
    key = f"landmarks:{pack}"
    if key in _REC_CACHE:
        return _REC_CACHE[key]
    import insightface

    root = os.path.join(getattr(folder_paths, "models_dir", "models"), "insightface")
    app = insightface.app.FaceAnalysis(
        name=pack, root=root, allowed_modules=["detection", "landmark_2d_106"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    _REC_CACHE[key] = app
    return app


def _largest_central_landmarks(app, bgr):
    """Return dense landmarks and detector confidence for the crop's primary face."""
    faces = [
        f for f in app.get(bgr)
        if (getattr(f, "landmark_2d_106", None) is not None
            or getattr(f, "kps", None) is not None)
    ]
    if not faces:
        return None, 0.0
    h, w = bgr.shape[:2]
    center = np.asarray((w * 0.5, h * 0.5), dtype=np.float32)

    def score(face):
        x0, y0, x1, y1 = (float(v) for v in face.bbox)
        area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
        fc = np.asarray(((x0 + x1) * 0.5, (y0 + y1) * 0.5), dtype=np.float32)
        distance_penalty = float(np.linalg.norm(fc - center)) / max(w, h, 1)
        return area * max(0.25, 1.0 - distance_penalty)

    face = max(faces, key=score)
    points = getattr(face, "landmark_2d_106", None)
    if points is None:
        points = face.kps
    return np.asarray(points, dtype=np.float64), float(getattr(face, "det_score", 1.0))


def _embed_faces(app, bgr):
    out = []
    for f in app.get(bgr):
        e = getattr(f, "normed_embedding", None)
        if e is None:
            continue
        out.append((f.bbox.tolist(), np.asarray(e, dtype=np.float32)))
    return out


def _best_match(cands, ref_emb):
    if not cands or ref_emb is None:
        return None, -1.0
    sims = [float(np.dot(e, ref_emb)) for _, e in cands]
    i = int(np.argmax(sims))
    return i, sims[i]


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _continuity_cost(box, last):
    cx, cy, sz = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, box[3] - box[1]
    d = ((cx - last[0]) ** 2 + (cy - last[1]) ** 2) ** 0.5
    return d + abs(sz - last[2]) * 2.0


def _build_clip_anchor(app, images, model, confidence, max_samples=24):
    """Average identity embedding of the subject, sampled from unambiguous frames of the clip."""
    B = images.shape[0]
    step = max(1, B // max_samples)
    embs = []
    for i in range(0, B, step):
        bgr = _to_bgr_u8(images[i])
        det = model.predict(bgr, conf=confidence, verbose=False)[0]
        boxes = det.boxes.xyxy.tolist() if len(det.boxes) else []
        if not boxes:
            continue
        heights = sorted((b[3] - b[1] for b in boxes), reverse=True)
        if len(heights) > 1 and heights[0] < heights[1] * 1.6:
            continue
        cands = _embed_faces(app, bgr)
        if not cands:
            continue
        j = max(range(len(cands)), key=lambda k: cands[k][0][3] - cands[k][0][1])
        embs.append(cands[j][1])
    if not embs:
        return None, 0
    a = np.mean(np.stack(embs), axis=0)
    n = np.linalg.norm(a)
    return (a / n if n > 0 else a), len(embs)


# ----------------------------------------------------------------------------
# math helpers
# ----------------------------------------------------------------------------


def _to_bgr_u8(img: torch.Tensor):
    a = (img[..., :3].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return a[..., ::-1].copy()


def _interp_gaps(vals, valid):
    n = len(vals)
    idx = np.arange(n)
    if not valid.any():
        return np.zeros(n, dtype=np.float64)
    return np.interp(idx, idx[valid], vals[valid])


def _smooth(vals, window: int, method: str = "gaussian"):
    if window <= 1 or len(vals) < 3:
        return vals
    window = min(int(window), len(vals))
    if window % 2 == 0:
        window += 1
    if window < 3:
        return vals
    pad = window // 2
    padded = np.pad(vals, pad, mode="reflect")

    if method == "savgol":
        try:
            from scipy.signal import savgol_filter

            polyorder = 2 if window > 3 else 1
            return np.asarray(savgol_filter(padded, window, polyorder))[pad: pad + len(vals)]
        except Exception:
            method = "gaussian"

    if method == "gaussian":
        x = np.arange(window, dtype=np.float64) - pad
        sigma = max(window / 6.0, 0.5)
        kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        kernel /= kernel.sum()
    else:
        kernel = np.ones(window, dtype=np.float64) / window

    return np.convolve(padded, kernel, mode="valid")[: len(vals)]


def _affine_crop(img: torch.Tensor, box, cw: int, ch: int) -> torch.Tensor:
    """Sub-pixel crop+resize in one bilinear sample. img [1,H,W,C] -> [1,ch,cw,C]."""
    import torch.nn.functional as F

    x, y, bw, bh = box
    _, H, W, C = img.shape
    src = img[..., :3].movedim(-1, 1).float()
    theta = torch.tensor(
        [[[bw / W, 0.0, (2.0 * x + bw) / W - 1.0],
          [0.0, bh / H, (2.0 * y + bh) / H - 1.0]]],
        dtype=torch.float32, device=src.device,
    )
    grid = F.affine_grid(theta, (1, 3, int(ch), int(cw)), align_corners=False)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.movedim(1, -1).to(img.dtype)


def _gaussian_blur_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    import torch.nn.functional as F

    if feather <= 0:
        return mask
    k = 2 * int(feather) + 1
    shortest = min(mask.shape[-2], mask.shape[-1])
    if shortest <= k:
        k = max(3, int(shortest / 2) | 1)
    sigma = max(k / 6.0, 0.5)
    x = torch.arange(k, device=mask.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(mask.dtype)
    pad = k // 2
    m = F.conv2d(F.pad(mask, (pad, pad, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    m = F.conv2d(F.pad(m, (0, 0, pad, pad), mode="replicate"), g.view(1, 1, k, 1))
    return m


def _face_region_mask(ch, cw, face_rect, dilation, feather, shape, device, dtype):
    """FaceDetailer-style paste mask: solid over the face box, dilated then blurred."""
    m = torch.zeros((1, 1, int(ch), int(cw)), device=device, dtype=torch.float32)
    fx, fy, fwd, fhd = face_rect
    fx -= dilation
    fy -= dilation
    fwd += 2 * dilation
    fhd += 2 * dilation

    if shape == "ellipse":
        yy = torch.arange(ch, device=device, dtype=torch.float32).view(-1, 1)
        xx = torch.arange(cw, device=device, dtype=torch.float32).view(1, -1)
        ccx, ccy = fx + fwd / 2.0, fy + fhd / 2.0
        rx, ry = max(fwd / 2.0, 1.0), max(fhd / 2.0, 1.0)
        m[0, 0] = (((xx - ccx) / rx) ** 2 + ((yy - ccy) / ry) ** 2 <= 1.0).float()
    else:
        x0 = max(0, int(round(fx)))
        y0 = max(0, int(round(fy)))
        x1 = min(int(cw), int(round(fx + fwd)))
        y1 = min(int(ch), int(round(fy + fhd)))
        if x1 > x0 and y1 > y0:
            m[0, 0, y0:y1, x0:x1] = 1.0

    return _gaussian_blur_mask(m, feather).clamp(0, 1).to(dtype)


def _geometry_lock(source_crops: torch.Tensor, refined_crops: torch.Tensor, strength: float):
    """Re-align refined crops onto the source crops' geometry with dense optical flow.

    H3 redraws eyes/nose/jaw a few pixels differently in every frame of the face pass, which
    reads as the face shaking or tilting on the head once it is pasted back. For each frame
    this estimates the flow from the source crop to the refined crop (DIS, at half canvas
    resolution, on blurred greyscale so sharpness differences do not matter), smooths and
    clamps it, then resamples the refined crop so its content sits where the source has it.
    Only geometry moves - the regenerated detail is kept. Both tensors are [n,3,H,W] float
    on the same device; returns the re-aligned refined crops plus the mean shift in px.
    """
    import cv2
    import torch.nn.functional as F

    n, _, H, W = refined_crops.shape
    work_w, work_h = max(96, W // 2), max(96, H // 2)
    rel = work_w / 384.0                       # tuned at 768 canvas / 384 work
    max_shift = W / 32.0
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    grey_w = torch.tensor([0.299, 0.587, 0.114], device=refined_crops.device).view(1, 3, 1, 1)

    def to_grey_u8(img):
        g = F.interpolate((img * grey_w).sum(1, keepdim=True), size=(work_h, work_w),
                          mode="area")
        return (g.clamp(0, 1) * 255.0).round().to(torch.uint8)[:, 0].cpu().numpy()

    src_g, ref_g = to_grey_u8(source_crops), to_grey_u8(refined_crops)
    flows = np.zeros((n, H, W, 2), dtype=np.float32)
    for j in range(n):
        a = cv2.GaussianBlur(src_g[j], (0, 0), 2.0 * rel)
        b = cv2.GaussianBlur(ref_g[j], (0, 0), 2.0 * rel)
        flow = cv2.GaussianBlur(dis.calc(a, b, None), (0, 0), 8.0 * rel)
        flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR) * (W / float(work_w))
        mag = np.linalg.norm(flow, axis=2, keepdims=True)
        flows[j] = flow * np.minimum(1.0, max_shift / np.maximum(mag, 1e-6)) * float(strength)

    flow_t = torch.from_numpy(flows).to(refined_crops.device)
    ys, xs = torch.meshgrid(torch.arange(H, device=flow_t.device, dtype=torch.float32),
                            torch.arange(W, device=flow_t.device, dtype=torch.float32),
                            indexing="ij")
    grid = torch.stack([(2.0 * (xs + flow_t[..., 0]) + 1.0) / W - 1.0,
                        (2.0 * (ys + flow_t[..., 1]) + 1.0) / H - 1.0], dim=-1)
    locked = F.grid_sample(refined_crops, grid, mode="bicubic", padding_mode="border",
                           align_corners=False).clamp(0, 1)
    return locked, float(np.linalg.norm(flows, axis=3).mean())


def _feather_mask(h, w, feather, device, dtype):
    m = torch.ones((h, w), device=device, dtype=dtype)
    f = int(max(0, min(feather, min(h, w) // 2 - 1)))
    if f <= 0:
        return m
    ramp = 0.5 - 0.5 * torch.cos(
        torch.linspace(0, np.pi, f + 2, device=device, dtype=dtype)[1:-1]
    )
    m[:f, :] *= ramp.view(-1, 1)
    m[h - f:, :] *= ramp.flip(0).view(-1, 1)
    m[:, :f] *= ramp.view(1, -1)
    m[:, w - f:] *= ramp.flip(0).view(1, -1)
    return m


# ----------------------------------------------------------------------------
# 1. face-specific prompt
# ----------------------------------------------------------------------------


class MiniMaxH3FacePromptEnhance:
    """Append conservative face-detail instructions to the second-pass prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "refinement_prompt": ("STRING", {
                    "default": DEFAULT_FACE_REFINEMENT_PROMPT,
                    "multiline": True,
                    "tooltip": "Instructions used only for the face-refinement pass. Keep identity, pose, and expression locked.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Refinement Prompt"
    DESCRIPTION = "Append identity-preserving detail instructions for the face-only pass."

    def run(self, prompt, refinement_prompt=DEFAULT_FACE_REFINEMENT_PROMPT):
        return (_append_face_refinement_prompt(prompt, refinement_prompt),)


# These face-specific wrappers make the actual 0.55 pass denoise distinct from
# the per-frame size multipliers and give the preset stable, explicit defaults.
class MiniMaxH3FaceSamplerSelect:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        names = list(comfy.samplers.SAMPLER_NAMES)
        preferred = "res_multistep"
        if preferred in names:
            names.remove(preferred)
            names.insert(0, preferred)
        return {"required": {"sampler_name": (names,)}}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Sampler"
    DESCRIPTION = "Sampler selector for the face-refinement pass."

    def run(self, sampler_name):
        import comfy.samplers

        return (comfy.samplers.sampler_object(sampler_name),)


class MiniMaxH3FaceScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        schedulers = list(comfy.samplers.SCHEDULER_NAMES)
        preferred = "beta"
        if preferred in schedulers:
            schedulers.remove(preferred)
            schedulers.insert(0, preferred)
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (schedulers,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Actual face-pass denoise. Preset default: 0.55. Optional size scaling multiplies this value."}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Scheduler"
    DESCRIPTION = "Build the face-pass sigma schedule with an explicit 0.55 denoise default."

    def run(self, model, scheduler, steps, denoise):
        import comfy.samplers

        steps = int(steps)
        denoise = float(denoise)
        if denoise <= 0.0:
            return (torch.FloatTensor([]),)
        total_steps = steps if denoise >= 1.0 else int(steps / denoise)
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        return (sigmas[-(steps + 1):],)


# ----------------------------------------------------------------------------
# 2. track + crop
# ----------------------------------------------------------------------------


class MiniMaxH3FaceTrackCrop:
    """Detect a face per frame, build a smoothed per-frame crop, emit a constant-size batch.

    The crop SIZE varies per frame so the face fills a constant fraction of every crop;
    every crop is then resized to one canvas size, because H3 generates a single fixed
    WxH for a whole sequence. Result: the face is always large in H3's input regardless
    of how small it was in the source frame.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "detector": (_detector_list(), {"tooltip": "YOLO face model from models/yolov8 (yolov9e-face-lindevs.pt recommended)."}),
                "confidence": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05,
                    "tooltip": "Minimum detection confidence. Lower finds more distant/blurry faces at the cost of false positives."}),
                "crop_factor": ("FLOAT", {"default": 2.2, "min": 1.2, "max": 8.0, "step": 0.1,
                    "tooltip": "Crop side as a multiple of detected face HEIGHT. 2.2 puts the face at ~45% of the crop. "
                               "Bigger = more context so the seam lands in hair/background, but less magnification. 2.0-3.0 is the useful range."}),
                "canvas_mode": (["auto_capped_768", "auto_no_downscale", "manual"],
                    {"default": "auto_capped_768",
                     "tooltip": "auto_capped_768: size the canvas from the LARGEST crop so no frame is downscaled, clamped to 384-768 (H3's useful range). Recommended.\n"
                                "auto_no_downscale: same but uncapped above - can get expensive on close-ups.\n"
                                "manual: use canvas_width/height as given."}),
                "canvas_width": ("INT", {"default": 768, "min": 128, "max": 1344, "step": 32,
                    "tooltip": "Resolution H3 regenerates the face at (manual mode only). 768 is H3's native short edge."}),
                "canvas_height": ("INT", {"default": 768, "min": 128, "max": 1344, "step": 32}),
                "face_tracking": ("BOOLEAN", {"default": True,
                    "label_on": "IDENTITY TRACKING", "label_off": "LARGEST FACE ONLY",
                    "tooltip": "Hold one subject through a crowd. Continuity (nearest box to the previous position) decides most frames; "
                               "a face-identity embedding (InsightFace) is consulted only when two candidates are similarly plausible. "
                               "With a single face in frame this costs nothing. Falls back to continuity-only when InsightFace is unavailable."}),
                "smooth_window": ("INT", {"default": 21, "min": 1, "max": 201, "step": 2,
                    "tooltip": "Frames of smoothing on the crop CENTRE (21 at 24fps is ~0.9s). Raise if the box shivers; lower if it lags fast heads."}),
                "size_smooth_window": ("INT", {"default": 51, "min": 1, "max": 201, "step": 2,
                    "tooltip": "Frames of smoothing on the crop SIZE. Wants MORE than the centre: size jitter reads as shimmer."}),
                "smooth_method": (["gaussian", "savgol", "moving_average"], {"default": "gaussian"}),
                "size_mode": (["per_frame", "max_of_clip"], {"default": "per_frame",
                    "tooltip": "per_frame: constant face-fraction in every crop (correct for push-ins). max_of_clip: one size for the whole clip."}),
            },
            "optional": {
                "identity_reference": ("IMAGE", {
                    "tooltip": "A clear face image of the person to track. When supplied, the subject is chosen by FACE IDENTITY "
                               "rather than by size, so a crowd scene locks onto the right person."}),
                "identity_threshold": ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum cosine similarity to accept a face as the reference person; below it tracking falls back to continuity."}),
                "select": (["largest", "most_central"], {"default": "largest",
                    "tooltip": "First-frame subject pick when no identity reference is connected."}),
                "fallback_detector": (["none"] + _detector_list(), {"default": "none",
                    "tooltip": "Used only on frames where the FACE detector finds nothing (subject turned away). "
                               "A person/body model estimates the head from the body box; 'none' interpolates instead."}),
                "fallback_head_frac": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.5, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "H3FACEXFORM", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("crops", "transform", "preview", "report", "canvas_w", "canvas_h")
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Track + Crop"
    DESCRIPTION = ("Per-frame YOLO face track -> smoothed, normalised crop -> constant-size batch for H3, "
                   "plus the transform needed to paste the refined result back.")

    def run(self, images, detector, confidence, crop_factor, canvas_mode, canvas_width,
            canvas_height, face_tracking, smooth_window, size_smooth_window, smooth_method,
            size_mode, identity_reference=None, identity_threshold=0.28, select="largest",
            fallback_detector="none", fallback_head_frac=0.5):
        model = _load_detector(detector)
        B, H, W, _ = images.shape

        cx = np.zeros(B); cy = np.zeros(B); sz = np.zeros(B); fw = np.zeros(B)
        valid = np.zeros(B, dtype=bool)       # a real FACE was seen
        via_body = np.zeros(B, dtype=bool)    # head located from a body box instead

        import comfy.model_management as _mm

        # ---- identity anchor -------------------------------------------------------
        ref_emb, app = None, None
        n_ident, n_cont, n_conflict = 0, 0, 0
        multi = False
        try:
            probe = model.predict(_to_bgr_u8(images[0]), conf=confidence, verbose=False)[0]
            multi = len(probe.boxes) > 1
        except Exception:
            pass

        if face_tracking and (multi or identity_reference is not None):
            try:
                app = _face_recogniser()
                if identity_reference is not None:
                    cands = _embed_faces(app, _to_bgr_u8(identity_reference[0]))
                    if cands:
                        j = max(range(len(cands)),
                                key=lambda k: cands[k][0][3] - cands[k][0][1])
                        ref_emb = cands[j][1]
                        print("[MiniMaxH3Face] identity anchor from the supplied reference")
                if ref_emb is None:
                    ref_emb, used = _build_clip_anchor(app, images, model, confidence)
                    if ref_emb is not None:
                        print(f"[MiniMaxH3Face] identity anchor built from the clip itself "
                              f"({used} unambiguous frames)")
            except Exception as exc:
                print(f"[MiniMaxH3Face] identity matching unavailable ({exc}) - using continuity tracking")

        last = None   # (cx, cy, size) of the subject on the previous resolved frame

        for i in range(B):
            _mm.throw_exception_if_processing_interrupted()
            frame_bgr = _to_bgr_u8(images[i])
            res = model.predict(frame_bgr, conf=confidence, verbose=False)[0]
            boxes = res.boxes.xyxy.tolist() if len(res.boxes) else []
            if not boxes:
                continue

            b = None
            if len(boxes) == 1:
                b = boxes[0]
                n_cont += 1
            elif last is None:
                if ref_emb is not None:
                    cands = _embed_faces(app, frame_bgr)
                    k, _ = _best_match(cands, ref_emb)
                    if k is not None:
                        b = cands[k][0]
                        n_ident += 1
                if b is None:
                    if select == "most_central":
                        fc = (W / 2.0, H / 2.0)
                        b = min(boxes, key=lambda q: ((q[0] + q[2]) / 2 - fc[0]) ** 2
                                + ((q[1] + q[3]) / 2 - fc[1]) ** 2)
                    else:
                        b = max(boxes, key=lambda q: (q[3] - q[1]))
            else:
                # Continuity first: the nearest box to where the subject was, penalised for
                # size change. The identity embedding is consulted only on ambiguity.
                ranked = sorted(boxes, key=lambda q: _continuity_cost(q, last))
                best, second = ranked[0], ranked[1]
                c0, c1 = _continuity_cost(best, last), _continuity_cost(second, last)
                conflict = (c1 < c0 * 2.0) or (_iou(best, second) > 0.2)

                if conflict and ref_emb is not None:
                    n_conflict += 1
                    near = [q for q in boxes if _continuity_cost(q, last) < c0 * 3.0] or boxes
                    cands = [c for c in _embed_faces(app, frame_bgr)
                             if any(_iou(c[0], q) > 0.3 for q in near)]
                    k, score = _best_match(cands, ref_emb)
                    if k is not None and score >= identity_threshold:
                        b = cands[k][0]
                        n_ident += 1
                if b is None:
                    b = best
                    n_cont += 1

            last = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, b[3] - b[1])
            cx[i] = (b[0] + b[2]) / 2.0
            cy[i] = (b[1] + b[3]) / 2.0
            sz[i] = b[3] - b[1]          # face HEIGHT: more stable than width as the head turns
            fw[i] = b[2] - b[0]          # face WIDTH: needed for the paste mask
            valid[i] = True

        found = int(valid.sum())
        if found == 0:
            raise ValueError(
                "Face inpaint: no face was detected in any frame. Lower `confidence`, or "
                "disable face inpainting for this clip."
            )

        sz_seed = _interp_gaps(sz, valid)
        if fallback_detector != "none" and (~valid).any():
            try:
                bmodel = _load_detector(fallback_detector)
                for i in np.nonzero(~valid)[0]:
                    res = bmodel.predict(_to_bgr_u8(images[i]), conf=confidence, verbose=False)[0]
                    if not len(res.boxes):
                        continue
                    bb = res.boxes.xyxy.tolist()
                    cls = (res.boxes.cls.tolist() if getattr(res.boxes, "cls", None) is not None
                           else [0] * len(bb))
                    people = [q for q, cc in zip(bb, cls) if int(cc) == 0] or bb
                    p = max(people, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
                    cx[i] = (p[0] + p[2]) / 2.0
                    cy[i] = p[1] + fallback_head_frac * max(sz_seed[i], 8.0)
                    sz[i] = sz_seed[i]
                    via_body[i] = True
            except Exception as exc:  # never let the fallback kill the run
                print(f"[MiniMaxH3Face] body fallback '{fallback_detector}' failed: {exc}")

        known = valid | via_body
        raw_cx = _interp_gaps(cx, known)
        raw_cy = _interp_gaps(cy, known)
        raw_sz = _interp_gaps(sz, valid)   # size ALWAYS from real face measurements
        raw_fw = _interp_gaps(fw, valid)
        sm_fw = _smooth(raw_fw, size_smooth_window, smooth_method)
        cx = _smooth(raw_cx, smooth_window, smooth_method)
        cy = _smooth(raw_cy, smooth_window, smooth_method)
        sz = _smooth(raw_sz, size_smooth_window, smooth_method)
        if size_mode == "max_of_clip":
            sz[:] = sz.max()

        def _jit(a):
            return float(np.abs(np.diff(a)).mean()) if len(a) > 1 else 0.0

        jit_before = (_jit(raw_cx) + _jit(raw_cy)) / 2.0
        jit_after = (_jit(cx) + _jit(cy)) / 2.0
        sz_before, sz_after = _jit(raw_sz), _jit(sz)

        if canvas_mode != "manual":
            need = float(min(sz.max() * crop_factor, H))
            snapped = int(np.ceil(need / 32.0) * 32)
            if canvas_mode == "auto_capped_768":
                snapped = min(snapped, 768)
            # Floor of 384: below it the regenerated face has too few pixels for H3 to
            # actually add detail. Tiny crops get upscaled further into the canvas, which
            # is exactly what gives distant faces enough resolution to be repainted.
            snapped = max(384, min(snapped, 1344))
            if snapped != canvas_height:
                print(f"[MiniMaxH3Face] canvas_mode={canvas_mode}: "
                      f"{canvas_width}x{canvas_height} -> {snapped}x{snapped} "
                      f"(largest crop {need:.0f}px)")
            canvas_width = canvas_height = snapped

        aspect = canvas_width / float(canvas_height)
        boxes = []
        crops = torch.zeros((B, canvas_height, canvas_width, 3), dtype=images.dtype)
        preview = images[..., :3].clone()

        for i in range(B):
            bh = sz[i] * crop_factor
            bw = bh * aspect
            if bw > W:
                bw, bh = float(W), float(W) / aspect
            if bh > H:
                bh, bw = float(H), float(H) * aspect
            x = min(max(cx[i] - bw / 2.0, 0.0), max(0.0, W - bw))
            y = min(max(cy[i] - bh / 2.0, 0.0), max(0.0, H - bh))
            box = (float(x), float(y), float(bw), float(bh))
            boxes.append(box)

            crops[i: i + 1] = _affine_crop(
                images[i: i + 1], box, canvas_width, canvas_height
            ).to(crops.dtype)

            # preview: green = real face, yellow = body fallback, red = interpolated
            xi, yi = int(round(x)), int(round(y))
            wi, hi = max(4, int(round(bw))), max(4, int(round(bh)))
            xi = min(xi, W - wi); yi = min(yi, H - hi)
            if valid[i]:
                r, g = 0.0, 1.0
            elif via_body[i]:
                r, g = 1.0, 1.0
            else:
                r, g = 1.0, 0.0
            for (yy0, yy1, xx0, xx1) in (
                (yi, yi + 2, xi, xi + wi), (yi + hi - 2, yi + hi, xi, xi + wi),
                (yi, yi + hi, xi, xi + 2), (yi, yi + hi, xi + wi - 2, xi + wi),
            ):
                preview[i, yy0:yy1, xx0:xx1, 0] = r
                preview[i, yy0:yy1, xx0:xx1, 1] = g
                preview[i, yy0:yy1, xx0:xx1, 2] = 0.0

        weights = _smooth(valid.astype(np.float64), max(9, smooth_window // 2), "gaussian")
        weights = np.clip(weights, 0.0, 1.0)

        runs, cur = [], 0
        for v in known:
            if v:
                if cur:
                    runs.append(cur)
                cur = 0
            else:
                cur += 1
        if cur:
            runs.append(cur)
        longest_gap = max(runs) if runs else 0

        mags = [canvas_height / float(b[3]) for b in boxes]
        transform = {
            "boxes": boxes,
            "canvas": (int(canvas_width), int(canvas_height)),
            "src_size": (int(W), int(H)),
            "frames": int(B),
            "weights": [float(w) for w in weights],
            "detected": [bool(v) for v in valid],
            "face_rect": [
                (
                    float(canvas_width) * 0.5 - 0.5 * float(sm_fw[i]) / max(b[2], 1e-6) * canvas_width,
                    float(canvas_height) * 0.5 - 0.5 * float(sz[i]) / max(b[3], 1e-6) * canvas_height,
                    float(sm_fw[i]) / max(b[2], 1e-6) * canvas_width,
                    float(sz[i]) / max(b[3], 1e-6) * canvas_height,
                )
                for i, b in enumerate(boxes)
            ],
            "face_height_src": [float(v) for v in sz],
            "crop_factor": float(crop_factor),
        }

        gapwarn = ""
        if longest_gap >= 12:
            gapwarn = (
                f"\n!! longest dropout is {longest_gap} frames ({longest_gap / 24.0:.1f}s). The crop "
                f"box is interpolated across it and the composite fades out there."
            )

        n_down = sum(1 for m in mags if m < 1.0)
        warn = ""
        if n_down:
            need = max(b[3] for b in boxes)
            warn = (
                f"\n!! {n_down}/{B} frames ({n_down / B * 100:.0f}%) have magnification < 1.0x - "
                f"their crops are DOWNSCALED into the canvas. Raise the canvas to >= {need:.0f}px, "
                f"lower crop_factor, or skip face inpainting on this close-up clip."
            )

        box_jit = float(np.mean([abs(boxes[i][0] - boxes[i - 1][0]) + abs(boxes[i][1] - boxes[i - 1][1])
                                 for i in range(1, len(boxes))])) if len(boxes) > 1 else 0.0
        report = (
            f"tracking: {n_cont} by continuity, {n_conflict} ambiguous "
            f"({n_ident} resolved by face identity)\n"
            f"frames={B}  face={found} ({found / B * 100:.0f}%)  "
            f"body-fallback={int(via_body.sum())}  interpolated={B - int(known.sum())}\n"
            f"face height  min={sz.min():.0f}px  mean={sz.mean():.0f}px  max={sz.max():.0f}px\n"
            f"face fills   ~{100.0 / crop_factor:.0f}% of every crop (crop_factor={crop_factor})\n"
            f"magnification into {canvas_width}x{canvas_height}: "
            f"min={min(mags):.2f}x  mean={sum(mags) / len(mags):.2f}x  max={max(mags):.2f}x\n"
            f"jitter ({smooth_method}) centre {jit_before:.2f} -> {jit_after:.2f} px/frame"
            f"   size {sz_before:.2f} -> {sz_after:.2f} px/frame\n"
            f"box movement {box_jit:.2f} px/frame (sub-pixel float boxes)\n"
            f"dropout runs: {len(runs)}  longest={longest_gap} frames"
            f"{gapwarn}{warn}"
        )
        print("[MiniMaxH3Face] " + report.replace("\n", "\n[MiniMaxH3Face] "))
        return (crops, transform, preview, report, int(canvas_width), int(canvas_height))


# ----------------------------------------------------------------------------
# 3. inject real video into the AV latent (img2img seed)
# ----------------------------------------------------------------------------


class MiniMaxH3FaceInjectVideoLatent:
    """Replace the VIDEO stream of an H3 AV latent with real encoded frames (img2img seed).

    H3's own conditioning nodes always build a zeros latent, so there is no stock
    video-to-video path. This encodes real frames into the video stream while leaving the
    audio stream intact, which turns SamplerCustomAdvanced + BasicScheduler `denoise`
    into ordinary img2img.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT", {"tooltip": "Empty AV latent from the H3 conditioning node sized to the crop canvas."}),
                "images": ("IMAGE", {"tooltip": "The face crops to refine."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Inject Video Latent"
    DESCRIPTION = "Encode real frames into the video stream of an H3 joint AV latent for img2img refinement."

    def run(self, av_latent, images, vae):
        samples = av_latent.get("samples")
        if samples is None:
            raise KeyError('LATENT is missing "samples".')
        is_nested = isinstance(samples, comfy.nested_tensor.NestedTensor) or getattr(
            samples, "is_nested", False
        )
        if not is_nested:
            raise ValueError(
                "Expected a MiniMax H3 joint AV latent (NestedTensor). Feed the LATENT output "
                "of a MiniMax H3 conditioning node."
            )

        members = list(samples.unbind())
        video_tmpl = members[0]

        encoded = vae.encode(images[..., :3])
        if encoded.ndim == 4:  # [B,C,H,W] -> [1,C,T,H,W]
            encoded = encoded.unsqueeze(0).movedim(1, 2)

        tgt_t, tgt_h, tgt_w = video_tmpl.shape[-3], video_tmpl.shape[-2], video_tmpl.shape[-1]
        got_t, got_h, got_w = encoded.shape[-3], encoded.shape[-2], encoded.shape[-1]
        if (got_h, got_w) != (tgt_h, tgt_w):
            raise ValueError(
                f"Spatial latent mismatch: encoded {got_h}x{got_w} but the AV latent expects "
                f"{tgt_h}x{tgt_w}. The crop canvas and the H3 conditioning width/height must match."
            )
        note = ""
        if got_t != tgt_t:
            if got_t > tgt_t:
                encoded = encoded[..., :tgt_t, :, :]
            else:
                pad = video_tmpl[..., : tgt_t - got_t, :, :].to(encoded.device, encoded.dtype)
                encoded = torch.cat([encoded, pad], dim=-3)
            note = (f"  WARNING temporal mismatch: encoded t={got_t} vs latent t={tgt_t} "
                    f"-> {'trimmed' if got_t > tgt_t else 'padded'}. Frame count is probably "
                    f"off H3's 17k+5 grid.\n")

        members[0] = encoded.to(video_tmpl.device, video_tmpl.dtype)
        out = dict(av_latent)
        out["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))

        report = (
            f"injected video latent {tuple(encoded.shape)} into AV latent "
            f"(streams={len(members)})\n{note}"
            f"frames_in={images.shape[0]}  {images.shape[2]}x{images.shape[1]}px"
        )
        return (out, report)


# ----------------------------------------------------------------------------
# 4. lock the audio stream to the first pass (keeps speech + lipsync)
# ----------------------------------------------------------------------------


class MiniMaxH3FaceAudioLock:
    """Copy the audio stream of the first-pass sampled latent into the refine latent and freeze it.

    The main generation already produced the final audio in its AV latent. Reusing that
    latent directly is bit-exact - the refined video keeps the identical soundtrack - and
    the frozen audio stream conditions the video branch, so regenerated mouths stay in
    sync with the speech. The noise mask is set to 1 on the video stream (denoise it) and
    0 on the audio stream (keep it exactly).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT", {"tooltip": "The refine AV latent (after Face Inject Video Latent)."}),
                "source_latent": ("LATENT", {"tooltip": "The SAMPLED latent of the main pass - its audio stream is reused as-is."}),
            },
            "optional": {
                "lock_audio": ("BOOLEAN", {"default": True,
                    "label_on": "LOCK AUDIO (lipsync)", "label_off": "FREE AUDIO",
                    "tooltip": "Freeze the audio stream so the refined pass cannot change the soundtrack. Leave on."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Audio Lock"
    DESCRIPTION = "Reuse the main pass's generated audio latent in the refine pass and freeze it for lipsync."

    @staticmethod
    def _nested_members(latent, name):
        samples = latent.get("samples")
        if samples is None or not (
                isinstance(samples, comfy.nested_tensor.NestedTensor)
                or getattr(samples, "is_nested", False)):
            raise ValueError(f"{name} must be a MiniMax H3 joint AV latent (NestedTensor).")
        return list(samples.unbind())

    def run(self, av_latent, source_latent, lock_audio=True):
        members = self._nested_members(av_latent, "av_latent")
        source = self._nested_members(source_latent, "source_latent")
        if len(members) < 2 or len(source) < 2:
            raise ValueError("MiniMax H3 AV latents must contain a video and an audio stream.")

        video = members[0]
        audio = source[1].to(members[1].device, members[1].dtype)
        note = ""
        if audio.shape != members[1].shape:
            tgt = members[1].shape[-1]
            got = audio.shape[-1]
            if got > tgt:
                audio = audio[..., :tgt]
            elif got < tgt:
                audio = torch.cat([audio, members[1][..., got:].to(audio.device, audio.dtype)], dim=-1)
            note = f"  WARNING audio latent length mismatch: source t={got} vs refine t={tgt} -> adjusted.\n"
        members[1] = audio

        out = dict(av_latent)
        out["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))

        if lock_audio:
            vmask = torch.ones_like(video)
            amask = torch.zeros_like(audio)
            out["noise_mask"] = comfy.nested_tensor.NestedTensor((vmask, amask))

        report = (f"audio stream {tuple(audio.shape)} copied from the main pass"
                  f"{' and locked (video denoises, audio is frozen)' if lock_audio else ''}\n{note}")
        print("[MiniMaxH3Face] " + report.strip())
        return (out, report)


# ----------------------------------------------------------------------------
# 5. per-frame denoise strength by face size
# ----------------------------------------------------------------------------


class MiniMaxH3FacePerFrameDenoise:
    """Scale denoise strength per frame, inversely to how big the face is.

    Tiny-face frames have no detail to preserve and want a strong pass so H3 SYNTHESISES
    a face; large-face frames have real detail and want a gentle one. ComfyUI's
    noise_mask scales denoising per latent position, so varying it along the temporal
    axis gives per-frame strength out of a single sampling pass. Place AFTER the Face
    Audio Lock - the audio-side zeros are preserved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "transform": ("H3FACEXFORM",),
                "strength_small_face": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Scaling start multiplier for small faces. At 1.00, the 0.55 face pass remains 0.55."}),
                "strength_large_face": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Scaling end multiplier for large faces. 0.25 x 0.55 gives about 0.14 effective denoise; 0.50 gives 0.275."}),
                "scale_mode": (["absolute_px", "relative_to_clip"], {"default": "absolute_px"}),
                "face_px_small": ("FLOAT", {"default": 60.0, "min": 4.0, "max": 400.0, "step": 1.0,
                    "tooltip": "Face height (source px) at or below which the full small-face strength applies."}),
                "face_px_large": ("FLOAT", {"default": 150.0, "min": 8.0, "max": 800.0, "step": 1.0,
                    "tooltip": "Face height (source px) at or above which the large-face strength applies."}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.1}),
                "smooth_frames": ("INT", {"default": 9, "min": 1, "max": 61, "step": 2}),
            },
            "optional": {
                "enable_size_scaling": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF keeps a constant 1.00 multiplier. ON interpolates from the small-face multiplier to the large-face multiplier."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Size Scaling"
    DESCRIPTION = "Optionally scale denoise by face size. Disabled by default for a constant 0.55 face pass."

    def run(self, av_latent, transform, strength_small_face, strength_large_face,
            scale_mode, face_px_small, face_px_large, gamma, smooth_frames,
            enable_size_scaling=False):
        import torch.nn.functional as F

        samples = av_latent.get("samples")
        if samples is None or not (
                isinstance(samples, comfy.nested_tensor.NestedTensor)
                or getattr(samples, "is_nested", False)):
            raise ValueError("Expected a MiniMax H3 joint AV latent (NestedTensor).")

        members = list(samples.unbind())
        video = members[0]
        latent_t = video.shape[-3]

        face = _face_heights_from_transform(transform)
        if face.size == 0:
            raise ValueError("transform has no boxes")

        if enable_size_scaling:
            if scale_mode == "relative_to_clip":
                lo, hi = float(face.min()), float(face.max())
            else:
                lo, hi = float(face_px_small), float(face_px_large)
            if hi - lo < 1e-6:
                t = np.zeros_like(face)
            else:
                t = np.clip((face - lo) / (hi - lo), 0.0, 1.0)
            t = np.clip(t, 0.0, 1.0) ** float(gamma)
            strength = strength_small_face + (strength_large_face - strength_small_face) * t
            strength = _smooth(strength, int(smooth_frames), "gaussian")
            strength = np.clip(strength, 0.0, 1.0)
            mode_report = (
                f"enabled, ramp {lo:.0f}-{hi:.0f}px ({scale_mode}), "
                f"multipliers {float(strength_small_face):.2f}->{float(strength_large_face):.2f}"
            )
        else:
            strength = np.ones_like(face, dtype=np.float64)
            mode_report = "disabled, constant multiplier 1.00"

        s = torch.from_numpy(strength).float().view(1, 1, -1)
        s = F.interpolate(s, size=int(latent_t), mode="linear", align_corners=True)
        s = s.view(1, 1, int(latent_t), 1, 1).to(video.device, torch.float32)

        vmask = s.expand(video.shape[0], 1, latent_t, video.shape[-2], video.shape[-1])
        vmask = vmask.expand(-1, video.shape[1], -1, -1, -1).contiguous()

        prev = av_latent.get("noise_mask")
        if prev is not None and (isinstance(prev, comfy.nested_tensor.NestedTensor)
                                 or getattr(prev, "is_nested", False)):
            # keep the audio side exactly as the Audio Lock left it
            pm = list(prev.unbind())
            pm[0] = vmask.to(pm[0].dtype)
            new_mask = comfy.nested_tensor.NestedTensor(tuple(pm))
        else:
            audio_zero = torch.zeros_like(members[1]) if len(members) > 1 else None
            new_mask = comfy.nested_tensor.NestedTensor(
                (vmask.to(video.dtype),) + ((audio_zero,) if audio_zero is not None else ()))

        out = dict(av_latent)
        out["noise_mask"] = new_mask
        report = (
            f"per-frame denoise: face {face.min():.0f}-{face.max():.0f}px, size scaling "
            f"{mode_report}  ->  strength "
            f"{strength.max():.2f} (smallest) .. {strength.min():.2f} (largest)\n"
            f"mean {strength.mean():.2f} over {len(strength)} frames, "
            f"{latent_t} latent steps, gamma={gamma}"
        )
        print("[MiniMaxH3Face] " + report)
        return (out, report)


# ----------------------------------------------------------------------------
# 6. landmark alignment
# ----------------------------------------------------------------------------


class MiniMaxH3FaceLandmarkAlign:
    """Align regenerated facial geometry to the corresponding input crop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_crops": ("IMAGE",),
                "refined_crops": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True,
                    "label_on": "ALIGN LANDMARKS", "label_off": "BYPASS",
                    "tooltip": "Correct small translation, scale, and head-roll changes before stitching."}),
                "alignment_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth_frames": ("INT", {"default": 7, "min": 1, "max": 31, "step": 2,
                    "tooltip": "Smooth only the generated-to-original correction, not the real head motion."}),
                "max_rotation_deg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 15.0, "step": 0.25}),
                "max_scale_change": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.30, "step": 0.01}),
                "max_shift_ratio": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.15, "step": 0.005,
                    "tooltip": "Maximum face-center correction as a fraction of the crop size."}),
                "min_confidence": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("aligned_crops", "comparison", "report")
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Landmark Align"
    DESCRIPTION = "Match refined eyes, nose, and mouth to the original crop before compositing."

    def run(self, original_crops, refined_crops, enabled, alignment_strength, smooth_frames,
            max_rotation_deg, max_scale_change, max_shift_ratio, min_confidence):
        if original_crops.shape[1:3] != refined_crops.shape[1:3]:
            raise ValueError(
                "Landmark alignment requires original and refined crops at the same resolution.")
        count = min(original_crops.shape[0], refined_crops.shape[0])
        if count <= 0:
            raise ValueError("Landmark alignment received no frames.")
        original = original_crops[:count, ..., :3]
        refined = refined_crops[:count, ..., :3]
        if not enabled or float(alignment_strength) <= 0.0:
            comparison = _face_alignment_comparison(original, refined, refined)
            return (refined_crops, comparison, "landmark alignment bypassed")

        try:
            import cv2
            app = _face_landmark_detector("buffalo_l")
        except Exception as exc:
            raise RuntimeError(
                "Could not load InsightFace buffalo_l for landmark alignment. Its first use "
                "downloads into ComfyUI/models/insightface/models/buffalo_l. "
                f"Original error: {exc}") from exc

        import comfy.model_management as mm
        try:
            import comfy.utils
            progress = comfy.utils.ProgressBar(count * 2)
        except Exception:
            progress = None

        angles = np.zeros(count, dtype=np.float64)
        log_scales = np.zeros(count, dtype=np.float64)
        shifts_x = np.zeros(count, dtype=np.float64)
        shifts_y = np.zeros(count, dtype=np.float64)
        source_centers = np.tile(
            np.asarray((refined.shape[2] * 0.5, refined.shape[1] * 0.5)), (count, 1))
        valid = np.zeros(count, dtype=bool)
        rejected = 0
        inlier_counts = []
        max_shift_px = float(max_shift_ratio) * float(min(refined.shape[1:3]))

        for index in range(count):
            mm.throw_exception_if_processing_interrupted()
            source_points, source_score = _largest_central_landmarks(
                app, _to_bgr_u8(refined[index]))
            if progress is not None:
                progress.update(1)
            target_points, target_score = _largest_central_landmarks(
                app, _to_bgr_u8(original[index]))
            if progress is not None:
                progress.update(1)
            if (source_points is None or target_points is None
                    or min(source_score, target_score) < float(min_confidence)):
                continue

            matrix, inliers = cv2.estimateAffinePartial2D(
                source_points, target_points, method=cv2.RANSAC,
                ransacReprojThreshold=max(1.5, refined.shape[1] * 0.003),
                maxIters=2000, confidence=0.995, refineIters=10)
            if matrix is None:
                continue
            inlier_count = int(inliers.sum()) if inliers is not None else len(source_points)
            if inlier_count < 3:
                continue

            a, b = float(matrix[0, 0]), float(matrix[1, 0])
            scale = float(np.hypot(a, b))
            angle = float(np.degrees(np.arctan2(b, a)))
            source_center = source_points.mean(axis=0)
            target_center = target_points.mean(axis=0)
            shift = target_center - source_center

            gross_rotation = max(15.0, float(max_rotation_deg) * 4.0)
            gross_scale = max(0.25, float(max_scale_change) * 4.0)
            gross_shift = max(24.0, max_shift_px * 4.0)
            if (abs(angle) > gross_rotation or abs(scale - 1.0) > gross_scale
                    or float(np.linalg.norm(shift)) > gross_shift):
                rejected += 1
                continue

            clipped_angle = float(np.clip(angle, -max_rotation_deg, max_rotation_deg))
            clipped_scale = float(np.clip(
                scale, 1.0 - max_scale_change, 1.0 + max_scale_change))
            shift_norm = float(np.linalg.norm(shift))
            if max_shift_px >= 0.0 and shift_norm > max_shift_px > 0.0:
                shift *= max_shift_px / shift_norm
            angles[index] = clipped_angle
            log_scales[index] = np.log(max(clipped_scale, 1e-6))
            shifts_x[index], shifts_y[index] = float(shift[0]), float(shift[1])
            source_centers[index] = source_center
            valid[index] = True
            inlier_counts.append(inlier_count)

        found = int(valid.sum())
        if found == 0:
            comparison = _face_alignment_comparison(original, refined, refined)
            report = (
                "landmark alignment: no reliable landmark pairs; returned refined crops unchanged")
            print("[MiniMaxH3Face] " + report)
            return (refined_crops, comparison, report)

        def fill_and_smooth(values):
            filled = _interp_gaps(values, valid)
            return _smooth(filled, int(smooth_frames), "gaussian")

        angles = np.clip(
            fill_and_smooth(angles), -float(max_rotation_deg), float(max_rotation_deg))
        log_limit_low = np.log(max(1.0 - float(max_scale_change), 1e-6))
        log_limit_high = np.log(1.0 + float(max_scale_change))
        log_scales = np.clip(fill_and_smooth(log_scales), log_limit_low, log_limit_high)
        shifts_x = fill_and_smooth(shifts_x)
        shifts_y = fill_and_smooth(shifts_y)
        for values in (source_centers[:, 0], source_centers[:, 1]):
            values[:] = _interp_gaps(values, valid)

        strength = float(alignment_strength)
        matrices = []
        applied_angles = angles * strength
        applied_scales = np.exp(log_scales * strength)
        applied_x = shifts_x * strength
        applied_y = shifts_y * strength
        for index in range(count):
            radians = np.radians(applied_angles[index])
            cosine, sine = np.cos(radians), np.sin(radians)
            scale = applied_scales[index]
            linear = scale * np.asarray(((cosine, -sine), (sine, cosine)))
            center = source_centers[index]
            destination_center = center + np.asarray((applied_x[index], applied_y[index]))
            translation = destination_center - linear @ center
            matrices.append(np.column_stack((linear, translation)))

        aligned = _warp_similarity_batch(refined, np.stack(matrices))
        if refined_crops.shape[0] > count:
            aligned = torch.cat((aligned, refined_crops[count:]), dim=0)
        comparison = _face_alignment_comparison(
            original, refined, aligned[:count, ..., :3])
        report = (
            f"landmark alignment: {found}/{count} measured, {count - found} interpolated, "
            f"{rejected} rejected; median correction rotation={np.median(applied_angles):+.3f}deg "
            f"scale={np.median(applied_scales):.4f} "
            f"shift=({np.median(applied_x):+.2f},{np.median(applied_y):+.2f})px; "
            f"max |rotation|={np.max(np.abs(applied_angles)):.3f}deg "
            f"max shift={np.max(np.hypot(applied_x, applied_y)):.2f}px "
            f"median inliers={np.median(inlier_counts):.0f}"
        )
        print("[MiniMaxH3Face] " + report)
        return (aligned, comparison, report)


# ----------------------------------------------------------------------------
# 7. stitch back
# ----------------------------------------------------------------------------


class MiniMaxH3FaceStitch:
    """Paste refined crops back using the per-frame transform, with feather + colour match.

    Only the face region composites, through a dilated then Gaussian-blurred mask; the
    warp is one batched grid_sample per chunk so the sub-pixel trajectory is preserved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_images": ("IMAGE",),
                "refined_crops": ("IMAGE",),
                "transform": ("H3FACEXFORM",),
                "paste_region": (["face_only", "face_ellipse", "full_crop"], {"default": "face_only",
                    "tooltip": "What gets composited back. face_only / face_ellipse paste just the detected face box; "
                               "full_crop pastes the whole crop and risks a visible rectangle."}),
                "mask_dilation": ("INT", {"default": 24, "min": 0, "max": 256, "step": 2,
                    "tooltip": "Grow the face box before blurring, in canvas px, so the blend has room."}),
                "feather": ("INT", {"default": 16, "min": 0, "max": 256, "step": 2,
                    "tooltip": "Gaussian blur radius on the paste mask, measured in SOURCE pixels so the blend is the "
                               "same physical width whatever the frame's magnification."}),
                "colour_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Match the refined crop's per-channel mean/std to the region it replaces."}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Global opacity of the refined face."}),
                "undetected_frames": (["fade_out", "skip", "composite_anyway"], {"default": "fade_out",
                    "tooltip": "What to do on frames where no face was found: fade the composite out (smooth), skip "
                               "them exactly, or paste regardless."}),
            },
            "optional": {
                "feather_scales_with_crop": ("BOOLEAN", {"default": False,
                    "tooltip": "Treat feather as canvas pixels instead (blend narrows as the crop shrinks). Leave off."}),
                "size_aware_blend": ("BOOLEAN", {"default": True,
                    "tooltip": "Fade the refined patch out as the source face becomes large enough to already contain useful detail."}),
                "full_refine_face_px": ("FLOAT", {"default": 60.0, "min": 4.0, "max": 800.0, "step": 1.0,
                    "tooltip": "Source-face height at or below which the generated face is stitched at full strength."}),
                "passthrough_face_px": ("FLOAT", {"default": 180.0, "min": 8.0, "max": 1200.0, "step": 1.0,
                    "tooltip": "Source-face height at or above which original pixels are kept, avoiding a needless VAE round trip."}),
                "masks": ("MASK", {
                    "tooltip": "Optional per-frame paste masks in canvas space. Overrides paste_region."}),
                "geometry_lock": ("BOOLEAN", {"default": True,
                    "label_on": "LOCK GEOMETRY TO SOURCE", "label_off": "OFF",
                    "tooltip": "Re-align every refined crop onto the source crop with dense optical flow before compositing, "
                               "so the regenerated eyes/nose/mouth sit exactly where the original face has them. Removes the "
                               "per-frame shaking / tilting the face pass introduces (about 60% less relative face motion "
                               "measured at 0.55 denoise) while keeping the regenerated detail. Default ON."}),
                "geometry_lock_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "1.0 follows the source geometry fully; lower values keep part of the regenerated pose."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Stitch Back"
    DESCRIPTION = "Composite H3-refined face crops back into the source frames (with optional geometry lock)."

    def run(self, base_images, refined_crops, transform, paste_region, mask_dilation, feather,
            colour_match, blend, undetected_frames="fade_out", masks=None,
            feather_scales_with_crop=False, size_aware_blend=True,
            full_refine_face_px=60.0, passthrough_face_px=180.0,
            geometry_lock=True, geometry_lock_strength=1.0):
        boxes = transform["boxes"]
        if undetected_frames == "composite_anyway":
            weights = None
        elif undetected_frames == "skip":
            weights = [1.0 if d else 0.0 for d in transform.get("detected", [])] or None
        else:
            weights = transform.get("weights")
        B = min(len(boxes), base_images.shape[0], refined_crops.shape[0])
        if base_images.shape[0] != refined_crops.shape[0]:
            print(f"[MiniMaxH3Face] frame count mismatch: base={base_images.shape[0]} "
                  f"refined={refined_crops.shape[0]} transform={len(boxes)} -> using {B}")

        import torch.nn.functional as F

        cw, ch = transform["canvas"]
        W, H = transform["src_size"]
        face_rects = transform.get("face_rect")
        size_weights = None
        if size_aware_blend:
            face_heights = _face_heights_from_transform(transform)[:B]
            size_weights = _size_aware_stitch_weights(
                face_heights, full_refine_face_px, passthrough_face_px)
            full = int(np.count_nonzero(size_weights >= 1.0 - 1e-6))
            skipped = int(np.count_nonzero(size_weights <= 1e-6))
            faded = int(B - full - skipped)
            print(
                f"[MiniMaxH3Face] size-aware stitch: full={full}, fade={faded}, "
                f"original-only={skipped}, mean refine opacity={size_weights.mean():.2f} "
                f"({float(full_refine_face_px):.0f}-{float(passthrough_face_px):.0f}px)"
            )

        import comfy.model_management as mm

        try:
            dev = mm.get_torch_device()
        except Exception:
            dev = base_images.device
        dt = base_images.dtype
        out = base_images[..., :3].clone()

        per_frame_mb = (H * W * 3 * 4) / 2 ** 20
        chunk = max(1, min(32, int(1024 / max(per_frame_mb, 1e-6))))
        lock = bool(geometry_lock) and float(geometry_lock_strength) > 0.0
        if lock:
            try:
                import cv2  # noqa: F401  (installed with ultralytics, which the detector needs anyway)
            except ImportError:
                print("[MiniMaxH3Face] geometry lock skipped: opencv-python (cv2) is not installed")
                lock = False
        lock_shift = []

        for c0 in range(0, B, chunk):
            mm.throw_exception_if_processing_interrupted()
            c1 = min(c0 + chunk, B)
            n = c1 - c0
            base = out[c0:c1].to(dev).float()

            if feather_scales_with_crop:
                f_can = int(feather)
            else:
                bh_mid = float(boxes[(c0 + c1 - 1) // 2][3])
                f_can = int(round(feather * (ch / max(bh_mid, 1.0))))
                f_can = max(1, min(f_can, ch // 3))

            if masks is not None:
                mk = masks[c0:c1].to(dev).float()
                if mk.shape[-2:] != (ch, cw):
                    mk = F.interpolate(mk.unsqueeze(1), size=(ch, cw),
                                       mode="bilinear", align_corners=False)
                else:
                    mk = mk.unsqueeze(1)
                if mask_dilation > 0:
                    k = 2 * int(mask_dilation) + 1
                    mk = F.max_pool2d(mk, k, stride=1, padding=k // 2)
                mask_can = _gaussian_blur_mask(mk, f_can).clamp(0, 1)
            elif paste_region == "full_crop":
                one = _feather_mask(ch, cw, f_can, dev, torch.float32)
                mask_can = one.view(1, 1, ch, cw).expand(n, 1, ch, cw)
            else:
                mask_can = torch.cat([
                    _face_region_mask(
                        ch, cw,
                        face_rects[i] if face_rects and i < len(face_rects)
                        else (cw * 0.25, ch * 0.25, cw * 0.5, ch * 0.5),
                        int(mask_dilation), f_can,
                        "ellipse" if paste_region == "face_ellipse" else "rect",
                        dev, torch.float32)
                    for i in range(c0, c1)], dim=0)

            th = torch.empty((n, 2, 3), dtype=torch.float32, device=dev)
            for j, i in enumerate(range(c0, c1)):
                x, y, bw, bh = (float(v) for v in boxes[i])
                th[j, 0, 0] = W / bw; th[j, 0, 1] = 0.0
                th[j, 0, 2] = (W - 2.0 * x) / bw - 1.0
                th[j, 1, 0] = 0.0;    th[j, 1, 1] = H / bh
                th[j, 1, 2] = (H - 2.0 * y) / bh - 1.0
            grid = F.affine_grid(th, (n, 3, int(H), int(W)), align_corners=False)

            patch_can = refined_crops[c0:c1, ..., :3].to(dev).movedim(-1, 1).float()
            if lock:
                source_can = torch.cat([
                    _affine_crop(base[j: j + 1], boxes[c0 + j], cw, ch) for j in range(n)
                ], dim=0).movedim(-1, 1)
                patch_can, shift = _geometry_lock(source_can, patch_can, float(geometry_lock_strength))
                lock_shift.append(shift)
            patch = F.grid_sample(patch_can, grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=False)
            m = F.grid_sample(mask_can.to(dev), grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False).clamp(0, 1)

            patch = patch.movedim(1, -1)                 # [n,H,W,3]
            m = m.movedim(1, -1)                         # [n,H,W,1]

            if colour_match > 0.0:
                wsum = m.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                bmu = (base * m).sum(dim=(1, 2), keepdim=True) / wsum
                pmu = (patch * m).sum(dim=(1, 2), keepdim=True) / wsum
                bsd = (((base - bmu) ** 2 * m).sum(dim=(1, 2), keepdim=True)
                       / wsum).sqrt().clamp_min(1e-6)
                psd = (((patch - pmu) ** 2 * m).sum(dim=(1, 2), keepdim=True)
                       / wsum).sqrt().clamp_min(1e-6)
                adj = (patch - pmu) * (bsd / psd) + bmu
                patch = patch + (adj - patch) * float(colour_match)
                patch = patch.clamp(0, 1)

            wv = torch.full((n, 1, 1, 1), float(blend), device=dev, dtype=torch.float32)
            if weights is not None:
                for j, i in enumerate(range(c0, c1)):
                    if i < len(weights):
                        wv[j] *= float(weights[i])
            if size_weights is not None:
                for j, i in enumerate(range(c0, c1)):
                    if i < len(size_weights):
                        wv[j] *= float(size_weights[i])
            mm_ = m * wv

            out[c0:c1] = ((1.0 - mm_) * base + mm_ * patch).to(out.device, dt)

        if lock:
            print(f"[MiniMaxH3Face] geometry lock: {B} crops re-aligned to the source face "
                  f"(strength {float(geometry_lock_strength):.2f}, mean shift "
                  f"{float(np.mean(lock_shift)) if lock_shift else 0.0:.2f} canvas px)")
        return (out,)


# ----------------------------------------------------------------------------
# 8. stitch option comparison
# ----------------------------------------------------------------------------


class MiniMaxH3FaceStitchOptionsComparison:
    """Render a controlled face-only/full-crop and size-aware on/off comparison."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original_crops": ("IMAGE",),
            "refined_crops": ("IMAGE",),
            "transform": ("H3FACEXFORM",),
            "mask_dilation": ("INT", {"default": 24, "min": 0, "max": 256, "step": 2}),
            "feather": ("INT", {"default": 16, "min": 0, "max": 256, "step": 2}),
            "colour_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "feather_scales_with_crop": ("BOOLEAN", {"default": False}),
            "full_refine_face_px": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 1024.0, "step": 1.0}),
            "passthrough_face_px": ("FLOAT", {"default": 180.0, "min": 1.0, "max": 2048.0, "step": 1.0}),
            "panel_size": ("INT", {"default": 256, "min": 128, "max": 512, "step": 64}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("comparison", "report")
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Stitch Options Comparison"
    DESCRIPTION = "Compare face-only/full-crop and size-aware blend on/off using the same raw refinement."

    @staticmethod
    def _colour_match(base, patch, mask, strength):
        if strength <= 0.0:
            return patch
        wsum = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        bmu = (base * mask).sum(dim=(1, 2), keepdim=True) / wsum
        pmu = (patch * mask).sum(dim=(1, 2), keepdim=True) / wsum
        bsd = (((base - bmu) ** 2 * mask).sum(dim=(1, 2), keepdim=True)
               / wsum).sqrt().clamp_min(1e-6)
        psd = (((patch - pmu) ** 2 * mask).sum(dim=(1, 2), keepdim=True)
               / wsum).sqrt().clamp_min(1e-6)
        adjusted = (patch - pmu) * (bsd / psd) + bmu
        return (patch + (adjusted - patch) * float(strength)).clamp(0, 1)

    def run(self, original_crops, refined_crops, transform, mask_dilation, feather,
            colour_match, blend, feather_scales_with_crop, full_refine_face_px,
            passthrough_face_px, panel_size):
        import comfy.model_management as mm
        import torch.nn.functional as F

        boxes = transform["boxes"]
        face_rects = transform.get("face_rect") or []
        tracked_weights = transform.get("weights")
        B = min(original_crops.shape[0], refined_crops.shape[0], len(boxes))
        if B <= 0:
            raise ValueError("No crop frames are available for stitch comparison")

        cw, ch = transform["canvas"]
        if original_crops.shape[1:3] != (ch, cw):
            raise ValueError(
                f"Original crop size {tuple(original_crops.shape[1:3])} does not match "
                f"transform canvas {(ch, cw)}")

        face_heights = _face_heights_from_transform(transform)[:B]
        size_weights = _size_aware_stitch_weights(
            face_heights, full_refine_face_px, passthrough_face_px)

        try:
            dev = mm.get_torch_device()
        except Exception:
            dev = original_crops.device

        panel = int(panel_size)
        output = torch.empty((B, panel * 2, panel * 4, 3), dtype=torch.float32)
        chunk = 4

        for c0 in range(0, B, chunk):
            mm.throw_exception_if_processing_interrupted()
            c1 = min(c0 + chunk, B)
            n = c1 - c0
            base = original_crops[c0:c1, ..., :3].to(dev).float()
            raw = refined_crops[c0:c1, ..., :3].to(dev).float()

            face_masks = []
            full_masks = []
            for i in range(c0, c1):
                if feather_scales_with_crop:
                    f_can = int(feather)
                else:
                    bh = float(boxes[i][3])
                    f_can = int(round(float(feather) * (ch / max(bh, 1.0))))
                    f_can = max(1, min(f_can, ch // 3))

                face_rect = (face_rects[i] if i < len(face_rects)
                             else (cw * 0.25, ch * 0.25, cw * 0.5, ch * 0.5))
                face_masks.append(_face_region_mask(
                    ch, cw, face_rect, int(mask_dilation), f_can,
                    "rect", dev, torch.float32))
                full_masks.append(
                    _feather_mask(ch, cw, f_can, dev, torch.float32)
                    .view(1, 1, ch, cw))

            face_mask = torch.cat(face_masks, dim=0).movedim(1, -1)
            full_mask = torch.cat(full_masks, dim=0).movedim(1, -1)
            face_patch = self._colour_match(base, raw, face_mask, colour_match)
            full_patch = self._colour_match(base, raw, full_mask, colour_match)

            track = torch.ones((n, 1, 1, 1), device=dev, dtype=torch.float32)
            if tracked_weights is not None:
                for j, i in enumerate(range(c0, c1)):
                    if i < len(tracked_weights):
                        track[j] = float(tracked_weights[i])
            aware = track.clone()
            for j, i in enumerate(range(c0, c1)):
                aware[j] *= float(size_weights[i])

            face_aware_alpha = face_mask * aware * float(blend)
            face_full_alpha = face_mask * track * float(blend)
            crop_aware_alpha = full_mask * aware * float(blend)
            crop_full_alpha = full_mask * track * float(blend)

            face_aware = base * (1.0 - face_aware_alpha) + face_patch * face_aware_alpha
            face_full = base * (1.0 - face_full_alpha) + face_patch * face_full_alpha
            crop_aware = base * (1.0 - crop_aware_alpha) + full_patch * crop_aware_alpha
            crop_full = base * (1.0 - crop_full_alpha) + full_patch * crop_full_alpha

            def shrink(images):
                return F.interpolate(
                    images.movedim(-1, 1), size=(panel, panel), mode="area"
                ).movedim(1, -1).cpu()

            base_small = shrink(base)
            raw_small = shrink(raw)
            row_top = torch.cat(
                [base_small, raw_small, shrink(face_aware), shrink(face_full)], dim=2)
            row_bottom = torch.cat(
                [base_small, raw_small, shrink(crop_aware), shrink(crop_full)], dim=2)
            output[c0:c1] = torch.cat([row_top, row_bottom], dim=1)

        report = (
            f"frames={B}; top: original | raw | face_only=on,size_aware=on | "
            f"face_only=on,size_aware=off; bottom: original | raw | "
            f"face_only=off,size_aware=on | face_only=off,size_aware=off; "
            f"dilation={int(mask_dilation)} feather={int(feather)} "
            f"colour_match={float(colour_match):.2f} blend={float(blend):.2f} "
            f"size_range={float(full_refine_face_px):.0f}-{float(passthrough_face_px):.0f}px"
        )
        print("[MiniMaxH3Face] stitch comparison: " + report)
        return (output, report)


# ----------------------------------------------------------------------------
# 9. raw denoise comparison
# ----------------------------------------------------------------------------


class MiniMaxH3FaceRawDenoiseComparison:
    """Render raw face generations made from one crop sequence at fixed denoise levels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original_crops": ("IMAGE",),
            "raw_035": ("IMAGE",),
            "raw_045": ("IMAGE",),
            "raw_055": ("IMAGE",),
            "raw_100": ("IMAGE",),
            "panel_size": ("INT", {"default": 256, "min": 128, "max": 512, "step": 64}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("comparison", "report")
    FUNCTION = "run"
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Raw Denoise Comparison"
    DESCRIPTION = (
        "Compare identical tracked crops generated at constant denoise 0.35, 0.45, "
        "0.55, and 1.00. No stitch or alignment is applied."
    )

    def run(self, original_crops, raw_035, raw_045, raw_055, raw_100, panel_size):
        import comfy.model_management as mm
        import torch.nn.functional as F

        images = [original_crops, raw_035, raw_045, raw_055, raw_100]
        batch = min(image.shape[0] for image in images)
        if batch <= 0:
            raise ValueError("No crop frames are available for raw denoise comparison")

        panel = int(panel_size)
        output = torch.empty((batch, panel, panel * len(images), 3), dtype=torch.float32)
        try:
            device = mm.get_torch_device()
        except Exception:
            device = original_crops.device

        for start in range(0, batch, 4):
            mm.throw_exception_if_processing_interrupted()
            end = min(start + 4, batch)
            panels = []
            for image in images:
                frames = image[start:end, ..., :3].to(device).float()
                panels.append(F.interpolate(
                    frames.movedim(-1, 1), size=(panel, panel), mode="area"
                ).movedim(1, -1).cpu())
            output[start:end] = torch.cat(panels, dim=2)

        report = (
            f"frames={batch}; original | denoise=0.35 | denoise=0.45 | "
            f"denoise=0.55 | denoise=1.00 (new face); panel={panel}px; "
            "raw tracked crops only; constant per-frame strength; same seed"
        )
        print("[MiniMaxH3Face] raw denoise comparison: " + report)
        return (output, report)


# ----------------------------------------------------------------------------
# 10. debug info
# ----------------------------------------------------------------------------


class MiniMaxH3FaceTransformInfo:
    """Print the per-frame transform - sanity-check tracking before spending GPU time."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"transform": ("H3FACEXFORM",),
                             "max_rows": ("INT", {"default": 12, "min": 1, "max": 400})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "TeaCache/MiniMaxH3/FaceInpaint"
    TITLE = "MiniMax H3 Face Transform Info"

    def run(self, transform, max_rows):
        boxes = transform["boxes"]
        cw, ch = transform["canvas"]
        lines = [f"frames={transform['frames']}  canvas={cw}x{ch}  src={transform['src_size']}",
                 f"{'frame':>6} {'x':>6} {'y':>6} {'w':>6} {'h':>6} {'mag':>6}"]
        step = max(1, len(boxes) // max_rows)
        for i in range(0, len(boxes), step):
            x, y, w, h = boxes[i]
            lines.append(f"{i:>6} {x:>6.0f} {y:>6.0f} {w:>6.0f} {h:>6.0f} {ch / h:>5.2f}x")
        txt = "\n".join(lines)
        print("[MiniMaxH3Face]\n" + txt)
        return (txt,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3FacePromptEnhance": MiniMaxH3FacePromptEnhance,
    "MiniMaxH3FaceSamplerSelect": MiniMaxH3FaceSamplerSelect,
    "MiniMaxH3FaceScheduler": MiniMaxH3FaceScheduler,
    "MiniMaxH3FaceTrackCrop": MiniMaxH3FaceTrackCrop,
    "MiniMaxH3FaceInjectVideoLatent": MiniMaxH3FaceInjectVideoLatent,
    "MiniMaxH3FaceAudioLock": MiniMaxH3FaceAudioLock,
    "MiniMaxH3FacePerFrameDenoise": MiniMaxH3FacePerFrameDenoise,
    "MiniMaxH3FaceLandmarkAlign": MiniMaxH3FaceLandmarkAlign,
    "MiniMaxH3FaceStitch": MiniMaxH3FaceStitch,
    "MiniMaxH3FaceStitchOptionsComparison": MiniMaxH3FaceStitchOptionsComparison,
    "MiniMaxH3FaceRawDenoiseComparison": MiniMaxH3FaceRawDenoiseComparison,
    "MiniMaxH3FaceTransformInfo": MiniMaxH3FaceTransformInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
