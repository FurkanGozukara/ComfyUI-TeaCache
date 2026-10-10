"""H3 controls over ComfyUI's native sparse producer, FirstBlockCache and VAE batching.

SOL math and GPU kernels belong to ComfyUI/comfy-kitchen. This adapter preserves
saved workflow controls, protects the first/final steps, composes existing block
patches, and optionally benchmarks the complete native projected-attention path.
SOL and FirstBlockCache are approximate; neither is a lossless speed switch.
"""

import copy
import logging
import math
import os
import time
import types

import torch

import comfy.model_management
import comfy.patcher_extension

try:
    from comfy_extras import nodes_sparse_attention as native_sparse
except ImportError:
    native_sparse = None

try:
    from comfy.model_prefetch import pause_malloc_graph
except ImportError:  # ComfyUI versions before the allocation compiler.
    from contextlib import nullcontext as pause_malloc_graph

class _CondCache:
    """FirstBlockCache state for one cond/uncond stream."""

    __slots__ = ("prev_resid", "tail_resid", "h_b0", "skip", "accum", "consec")

    def __init__(self):
        self.prev_resid = None
        self.tail_resid = None
        self.h_b0 = None
        self.skip = False
        self.accum = 0.0
        self.consec = 0


class H3Optimizer:
    """Shared state + logic behind the wrapper, the block patches and the attention override."""

    def __init__(self, num_blocks, fbc_enabled, fbc_threshold, fbc_start_percent, fbc_end_percent,
                 fbc_max_consecutive, fbc_cache_device, sparse_mode, sparse_tau, sparse_dense_steps_pct,
                 sparse_dense_layers, sparse_min_video_rows, verbose, sparse_backend="auto",
                 sparse_extra_tokens=256, sparse_dense_last_steps=1):
        self.num_blocks = num_blocks
        self.last_block = num_blocks - 1
        self.fbc_enabled = fbc_enabled
        self.fbc_threshold = float(fbc_threshold)
        self.fbc_start_percent = float(fbc_start_percent)
        self.fbc_end_percent = float(fbc_end_percent)
        self.fbc_max_consecutive = int(fbc_max_consecutive)
        self.fbc_cache_device = fbc_cache_device
        self.sparse_mode = sparse_mode  # "auto" | "enabled" | "disabled"
        self.sparse_tau = float(sparse_tau)
        self.sparse_dense_steps_pct = float(sparse_dense_steps_pct)
        self.sparse_dense_layers = int(sparse_dense_layers)
        self.sparse_min_video_rows = int(sparse_min_video_rows)
        self.sparse_backend = sparse_backend
        self.sparse_extra_tokens = int(sparse_extra_tokens)
        self.sparse_dense_last_steps = int(sparse_dense_last_steps)
        self.verbose = verbose

        # per-run state
        self.step_index = -1
        self.total_steps = 0
        self.layer = -1
        self.seq_len = None
        self.video_start = None
        self.cond_key = (0,)
        self.caches = {}
        self.fbc_skipped_steps = 0
        self.fbc_computed_steps = 0
        self._last_sigma = None

        # Backend decisions are specific to the full attention workload.
        self.sparse_failed_reason = None
        self.sparse_calls = 0
        self.dense_calls = 0
        self._gate_done = {}
        self.native_patch = None
        if sparse_mode != "disabled" and native_sparse is not None:
            self.native_patch = native_sparse.SparseAttnPatch(
                tau=self.sparse_tau, topk_ratio=0.0, vsa=False,
                sigma_start=float("inf"), sigma_end=0.0, min_tokens=self.sparse_min_video_rows,
                dense_blocks=set(range(self.sparse_dense_layers)), sink_conditioning="exact_kv_and_rows",
                extra_tokens=self.sparse_extra_tokens, verbose=verbose)

    # ------------------------------------------------------------------ run observation

    def observe(self, timestep, transformer_options, layout):
        sigma = float(timestep.flatten()[0].detach().float().item()) / 1000.0
        sigmas = transformer_options.get("sample_sigmas", None)
        if sigmas is not None and sigmas.numel() > 1:
            idx = int((sigmas.detach().float().cpu() - sigma).abs().argmin().item())
            total = sigmas.numel() - 1
        else:
            # fallback clock: sigma decreasing within a run, jumps up on a new run
            if self._last_sigma is None or sigma > self._last_sigma + 1e-6:
                idx = 0
            else:
                idx = self.step_index + (0 if abs(sigma - (self._last_sigma or -1)) <= 1e-9 else 1)
            total = max(idx + 1, self.total_steps)
        self._last_sigma = sigma

        if idx == 0 and self.step_index != 0:
            self.reset_run()
        self.step_index = idx
        self.total_steps = total
        self.layer = -1
        self.cond_key = tuple(transformer_options.get("cond_or_uncond", [0]))

        if layout is not None:
            self.seq_len = layout.seq_len
            video_start = None
            for a, _b, kind in layout.segments:
                if kind == "video":
                    video_start = a
                    break
            self.video_start = video_start
        else:
            self.seq_len = None
            self.video_start = None

    def reset_run(self):
        if self.verbose and (self.fbc_skipped_steps or self.fbc_computed_steps):
            logging.info(f"[MiniMaxH3Speed] run summary: {self.fbc_skipped_steps} skipped / "
                         f"{self.fbc_computed_steps} computed block-stack steps, "
                         f"{self.sparse_calls} sparse / {self.dense_calls} dense attention calls")
        self.caches = {}
        self.fbc_skipped_steps = 0
        self.fbc_computed_steps = 0
        self.sparse_calls = 0
        self.dense_calls = 0
        if self.native_patch is not None:
            self.native_patch.reset()

    # ------------------------------------------------------------------ FirstBlockCache

    def _to_cache_device(self, t):
        if self.fbc_cache_device == "cpu":
            return t.to("cpu")
        return t

    def _from_cache_device(self, t, device):
        if t.device != device:
            return t.to(device)
        return t

    def fbc_window_open(self):
        if not self.fbc_enabled or self.total_steps <= 0:
            return False
        if self.step_index >= self.total_steps - max(1, self.sparse_dense_last_steps):
            return False
        pct = self.step_index / max(self.total_steps, 1)
        return self.fbc_start_percent <= pct < self.fbc_end_percent

    def block_patch(self, index, block=None, previous=None):
        def patch(args, extra):
            img = args["img"]
            self.layer = index
            def original(values):
                if block is not None and self.sparse_window_open(index) and "attention" not in values:
                    values = {**values, "attention": lambda h, rope_freqs=None, transformer_options={}: self.native_attention(
                        block.attn, h, rope_freqs, transformer_options, index)}
                return previous(values, extra) if previous is not None else extra["original_block"](values)
            if not self.fbc_enabled:
                return original(args)
            cache = self.caches.setdefault(self.cond_key, _CondCache())

            if index == 0:
                h_in = img.clone()
                out = original(args)["img"]
                # Cache tensors outlive the compiler's per-block allocation scope.
                with pause_malloc_graph():
                    resid = out - h_in
                del h_in
                skip = False
                if cache.prev_resid is not None and cache.tail_resid is not None and self.fbc_window_open():
                    prev = self._from_cache_device(cache.prev_resid, resid.device)
                    delta = ((resid - prev).abs().float().mean()
                             / prev.abs().float().mean().clamp_min(1e-8)).item()
                    self.accum_delta = delta
                    cache.accum += delta
                    if cache.accum < self.fbc_threshold and cache.consec < self.fbc_max_consecutive:
                        skip = True
                        cache.consec += 1
                    else:
                        cache.accum = 0.0
                        cache.consec = 0
                cache.prev_resid = self._to_cache_device(resid)
                cache.skip = skip
                if skip:
                    self.fbc_skipped_steps += 1
                else:
                    self.fbc_computed_steps += 1
                    with pause_malloc_graph():
                        cache.h_b0 = out.clone()
                return {"img": out}

            if cache.skip:
                if index == self.last_block:
                    tail = self._from_cache_device(cache.tail_resid, img.device)
                    self._maybe_log_final_step()
                    return {"img": img.add_(tail.to(img.dtype))}
                return {"img": img}

            out = original(args)["img"]
            if index == self.last_block and cache.h_b0 is not None:
                with pause_malloc_graph():
                    cache.tail_resid = self._to_cache_device(out - cache.h_b0)
                cache.h_b0 = None
            if index == self.last_block:
                self._maybe_log_final_step()
            return {"img": out}

        return patch

    def _maybe_log_final_step(self):
        if self.verbose and self.step_index == self.total_steps - 1:
            logging.info(f"[MiniMaxH3Speed] run done: skipped {self.fbc_skipped_steps} of "
                         f"{self.fbc_skipped_steps + self.fbc_computed_steps} block-stack evaluations, "
                         f"{self.sparse_calls} sparse / {self.dense_calls} dense attention calls")

    # ------------------------------------------------------------------ native sparse attention

    def sparse_window_open(self, index):
        if self.sparse_mode == "disabled" or self.native_patch is None:
            return False
        return (index >= self.sparse_dense_layers
                and self.step_index >= math.ceil(self.sparse_dense_steps_pct * max(self.total_steps, 1))
                and self.step_index < self.total_steps - self.sparse_dense_last_steps)

    def _native_key(self, attn, h, transformer_options):
        layout = transformer_options.get("minimax_h3_layout")
        prefix = self.video_start if layout is not None else None
        return (h.device, h.dtype, tuple(h.shape), h.stride(), attn.heads, attn.head_dim,
                prefix, self.sparse_tau, self.sparse_extra_tokens)

    def _gate_native(self, dense, sparse, device):
        """Time the complete projected attention, including the native producer.

        This is a speed/finite-output gate, not a promise of lossless sparsity.
        The error printed here is after output projection on sampled token rows.
        Numerical validation against SDPA belongs to the reproducible audit.
        """
        want = dense()
        got = sparse()
        finite = bool(torch.isfinite(got).all())
        if not finite:
            raise RuntimeError("native SOL output contains NaN/Inf")
        stride = max(1, got.shape[0] // 256)
        a, b = got[::stride].float(), want[::stride].float()
        rel = (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12)).item()
        del got, want, a, b
        times = []
        for fn in (dense, sparse):
            samples = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                out = fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
                del out
            times.append(sorted(samples)[1])
        speedup = times[0] / max(times[1], 1e-9)
        selected = speedup >= 1.05
        logging.info("[MiniMaxH3Speed] native complete attention: %.2f ms sparse / %.2f ms dense (%.2fx), "
                     "sampled projected rel_l2 %.5f; selected %s", times[1], times[0], speedup, rel,
                     "native SOL" if selected else "dense")
        return selected

    def native_attention(self, attn, h, rope_freqs, transformer_options, index):
        dense = lambda: attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
        if not native_sparse.h3_eligible(attn, h, rope_freqs, transformer_options, self.native_patch, index):
            self.dense_calls += 1
            return dense()
        key = self._native_key(attn, h, transformer_options)
        sparse = lambda: native_sparse.h3_sparse_attention(attn, h, rope_freqs, transformer_options, self.native_patch, index)
        try:
            if key not in self._gate_done:
                if self.sparse_mode == "enabled":
                    self._gate_done[key] = True
                else:
                    with pause_malloc_graph():
                        self._gate_done[key] = self._gate_native(dense, sparse, h.device)
            if self._gate_done[key]:
                out = sparse()
                self.sparse_calls += 1
                return out
        except (RuntimeError, NotImplementedError) as exc:
            self._gate_done[key] = False
            self.sparse_failed_reason = f"{type(exc).__name__}: {exc}"
            logging.warning("[MiniMaxH3Speed] native SOL unavailable for this shape, using dense: %s", self.sparse_failed_reason)
        self.dense_calls += 1
        return dense()


class MiniMaxH3SpeedOptimizer:
    """Apply native sparse attention and optional FirstBlockCache to a loaded H3 model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "A MiniMax H3 diffusion model."}),
                "first_block_cache": ("BOOLEAN", {"default": True, "tooltip": "Skip the transformer tail when block 0's residual barely moved since the previous step (FirstBlockCache). The dominant speedup of the reference line."}),
                "fbc_threshold": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005, "tooltip": "Accumulated relative block-0 residual change below which steps are skipped. 0.08 is the conservative default. Cache reuse is approximate; higher values can change the result more."}),
                "fbc_start_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Never skip before this fraction of the schedule (early steps set global structure)."}),
                "fbc_end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Never skip after this fraction of the schedule (final steps set fine detail)."}),
                "fbc_max_consecutive": ("INT", {"default": 3, "min": 1, "max": 50, "tooltip": "Cap on consecutive skipped steps."}),
                "sparse_attention": (["auto", "enabled", "disabled"], {"default": "auto", "tooltip": "ComfyUI's native SOL attention, including its chunked H3 producer. Auto times complete projected attention and requires a 5% speed win; enabled skips that benchmark; disabled preserves dense attention. Sparse output is approximate."}),
                "sparse_dense_steps_pct": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of early steps that always run dense attention (reference: 10 of 50)."}),
                "sparse_dense_layers": ("INT", {"default": 2, "min": 0, "max": 50, "tooltip": "First N transformer blocks always run dense attention (reference: 2)."}),
            },
            "optional": {
                "sparse_tau": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 100.0, "step": 0.1, "tooltip": "Sol-Attn routing threshold temperature. 1.0 is the released H3 policy; higher keeps fewer KV blocks."}),
                "sparse_min_video_rows": ("INT", {"default": 4096, "min": 0, "max": 1000000, "tooltip": "Minimum packed H3 token count for native sparse eligibility. Short sequences often gain little."}),
                "fbc_cache_device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Where cached residuals live. 'cpu' saves VRAM at some transfer cost."}),
                "verbose": ("BOOLEAN", {"default": True}),
                "enable_speedup": ("BOOLEAN", {
                    "default": True,
                    "label_on": "ENABLED",
                    "label_off": "DISABLED (normal)",
                    "tooltip": "Master switch for H3 optimizations. Disable it to pass through the original model and disable the linked MiniMax H3 VAE Speedup. Speed depends on the workload, GPU and selected controls.",
                }),
                "sparse_backend": (["auto", "comfy_kitchen", "vendored", "native"], {"default": "native", "tooltip": "Native ComfyUI/Comfy Kitchen only. Older saved auto, comfy_kitchen and vendored values are accepted as aliases so workflows keep loading; the duplicate bundled kernels have been retired."}),
                "sparse_extra_tokens": ("INT", {"default": 256, "min": 0, "max": 256, "step": 64, "tooltip": "Comfy Kitchen: extra tokens attended outside routed blocks. 256 reduces sparse approximation and brightness/detail pulsing; 0 is faster. Requires current ComfyUI and comfy-kitchen >=0.2.37."}),
                "sparse_dense_last_steps": ("INT", {"default": 1, "min": 0, "max": 50, "tooltip": "Final steps use dense attention and recompute the block stack to protect fine detail. 0 keeps the original sparse schedule; the final block stack still always computes."}),
            },
        }

    RETURN_TYPES = ("MODEL", "BOOLEAN")
    RETURN_NAMES = ("model", "speedup_enabled")
    FUNCTION = "apply"
    CATEGORY = "TeaCache/MiniMaxH3"
    TITLE = "MiniMax H3 Speed Optimizer"
    DESCRIPTION = ("Optional FirstBlockCache and ComfyUI's native SOL sparse attention. "
                   "Auto benchmarks the complete native attention path against the model's dense backend. "
                   "The master switch can also drive the paired VAE speedup node. "
                   "Techniques that don't work or don't win on your GPU fall back to the normal path automatically.")

    @classmethod
    def _invalid_inputs(cls, values):
        """Numbers and choices that ComfyUI's prompt validation would have rejected."""
        spec = cls.INPUT_TYPES()
        invalid = []
        for name, (kind, *options) in {**spec["required"], **spec["optional"]}.items():
            value = values[name]
            if isinstance(kind, list):
                valid = value in kind
            elif kind in ("INT", "FLOAT"):
                limits = options[0]
                valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
                         and limits["min"] <= value <= limits["max"])
            else:
                continue
            if not valid:
                invalid.append(f"{name}={value!r}")
        return invalid

    def apply(self, model, first_block_cache, fbc_threshold, fbc_start_percent, fbc_end_percent,
              fbc_max_consecutive, sparse_attention, sparse_dense_steps_pct, sparse_dense_layers,
              sparse_tau=1.0, sparse_min_video_rows=4096, fbc_cache_device="gpu", verbose=True,
              enable_speedup=True, sparse_backend="auto", sparse_extra_tokens=256,
              sparse_dense_last_steps=1):
        # Prompt validation does not always run before this node: other extensions patch it and
        # saved API prompts bypass the frontend. A workflow whose widget values were saved in the
        # wrong order (see web/js/minimax_h3_speed.js) then arrives here, e.g. with
        # fbc_end_percent='gpu'. Degrade to the normal path instead of killing the run.
        invalid = self._invalid_inputs(locals())
        if invalid:
            logging.warning("[MiniMaxH3Speed] speedup skipped, this node has invalid input values: "
                            f"{', '.join(invalid)}. Its saved widget values are scrambled: reload the "
                            "preset or recreate the node.")
            return (model, False)

        if not enable_speedup:
            return (model, False)

        diffusion_model = model.get_model_object("diffusion_model")
        if type(diffusion_model).__name__ != "MiniMaxH3Model":
            raise ValueError(f"MiniMaxH3SpeedOptimizer requires a MiniMax H3 model, got {type(diffusion_model).__name__}. "
                             "Connect the MiniMax H3 diffusion model (e.g. minimax_h3_fl2va / ref2va).")
        if not first_block_cache and sparse_attention == "disabled":
            return (model, True)
        if sparse_attention != "disabled" and native_sparse is None:
            logging.warning("[MiniMaxH3Speed] native sparse attention is unavailable: update ComfyUI. Using dense attention.")

        m = model.clone()
        opt = H3Optimizer(
            num_blocks=len(diffusion_model.blocks),
            fbc_enabled=first_block_cache,
            fbc_threshold=fbc_threshold,
            fbc_start_percent=fbc_start_percent,
            fbc_end_percent=fbc_end_percent,
            fbc_max_consecutive=fbc_max_consecutive,
            fbc_cache_device=fbc_cache_device,
            sparse_mode=sparse_attention,
            sparse_tau=sparse_tau,
            sparse_dense_steps_pct=sparse_dense_steps_pct,
            sparse_dense_layers=sparse_dense_layers,
            sparse_min_video_rows=sparse_min_video_rows,
            verbose=verbose,
            sparse_backend=sparse_backend,
            sparse_extra_tokens=sparse_extra_tokens,
            sparse_dense_last_steps=sparse_dense_last_steps,
        )

        def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
            payload = kwargs.get("minimax_payload") or {}
            opt.observe(timestep, transformer_options, payload.get("layout"))
            return executor(x, timestep, context, transformer_options, **kwargs)

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                               "minimax_h3_speed", wrapper)
        replacements = m.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for i, block in enumerate(diffusion_model.blocks):
            previous = replacements.get(("double_block", i))
            m.set_model_patch_replace(opt.block_patch(i, block, previous), "dit", "double_block", i)

        m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, "minimax_h3_speed", lambda patcher: opt.reset_run())

        m.model_options["minimax_h3_speed_optimizer"] = opt
        return (m, True)


def _batched_tiled_decode(self, z, batch_size_resolver, frames=slice(None)):
    """Batched drop-in for MiniMaxH3VideoVAE.tiled_decode: same tiles, same blend, same canvas,
    but the (identical-shape) tile decodes run through the ViT decoder as batches.

    `frames` follows ComfyUI's tiled_decode(z, frames) (comfy ec1537d, October 10, 2026): each decoded
    tile keeps only those output frames before blending, as the native decode does; older cores call
    without it and keep every frame."""
    height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
    y_idx, y_len, y_overlap = self.split_tiles(height)
    x_idx, x_len, x_overlap = self.split_tiles(width)

    coords = []
    for i_pos, i_len in zip(y_idx, y_len):
        zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
        for j_pos, j_len in zip(x_idx, x_len):
            zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
            coords.append((zi, zl, zj, zw))

    n_tiles = len(coords)
    batch = max(1, min(batch_size_resolver(z, n_tiles), n_tiles))
    tiles = []
    start = 0
    while start < n_tiles:
        chunk = coords[start:start + batch]
        try:
            stacked = torch.cat([z[..., zi:zi + zl, zj:zj + zw] for zi, zl, zj, zw in chunk], dim=0)
            decoded = self._decode_pixels(stacked)
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
            logging.warning(f"[MiniMaxH3Speed] VAE tile batch {batch} hit OOM, retrying sequentially")
            torch.cuda.empty_cache()
            batch = 1
            continue
        tiles.extend(t[:, :, frames] for t in decoded.split(1, dim=0))
        start += len(chunk)

    canvas = None
    row_tails = []
    out_y = 0
    index = 0
    for i in range(len(y_idx)):
        new_tails = []
        left_tail = None
        out_x = 0
        for j in range(len(x_idx)):
            tile = tiles[index]
            index += 1
            if i < len(y_idx) - 1:
                new_tails.append(tile[..., -y_overlap[i]:, :].clone())
            next_left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
            if i > 0:
                tile = self.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
            left_tail = next_left_tail
            if i < len(y_idx) - 1:
                tile = tile[..., :-y_overlap[i], :]
            if j < len(x_idx) - 1:
                tile = tile[..., :, :-x_overlap[j]]
            if canvas is None:
                canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
            canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
            out_x += tile.shape[-1]
        row_tails = new_tails
        out_y += tile.shape[-2]
    return canvas


class MiniMaxH3VAESpeedup:
    """Batch the MiniMax H3 video VAE's spatial tile decodes (bit-identical, launch-bound win)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE."}),
                "tile_batch_size": ("INT", {"default": 0, "min": 0, "max": 32, "tooltip": "Tiles decoded per launch. 0 = auto from free VRAM."}),
            },
            "optional": {
                "enable_speedup": ("BOOLEAN", {"default": True, "tooltip": "Connect speedup_enabled from MiniMax H3 Speed Optimizer so its master switch also restores the normal VAE decode path."}),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "apply"
    CATEGORY = "TeaCache/MiniMaxH3"
    TITLE = "MiniMax H3 VAE Speedup"
    DESCRIPTION = ("Feeds the video VAE's identical-shape spatial decode tiles through the ViT decoder "
                   "as one batch instead of one launch per tile (NVlabs Sana sol-engine vae_shard line). "
                   "Same arithmetic; any deviation is fp16 rounding from batch-dependent GEMM kernel "
                   "selection, far below visible. Auto-sizes the batch to free VRAM.")

    def apply(self, vae, tile_batch_size=0, enable_speedup=True):
        if not enable_speedup:
            return (vae,)

        fsm = vae.first_stage_model
        if type(fsm).__name__ != "MiniMaxH3VideoVAE":
            raise ValueError(f"MiniMaxH3VAESpeedup requires the MiniMax H3 video VAE, got {type(fsm).__name__}.")

        # Keep the loader's cached VAE pristine so turning the master switch off later really
        # restores the stock decode path instead of retaining this instance-level method patch.
        vae = copy.copy(vae)
        fsm = copy.copy(fsm)
        vae.first_stage_model = fsm

        measured = {}

        def batch_size_resolver(z, n_tiles):
            if tile_batch_size > 0:
                return tile_batch_size
            key = (z.shape[-3], z.device)
            if key not in measured:
                try:
                    free, _total = torch.cuda.mem_get_info(z.device)
                except Exception:  # noqa: BLE001 - non-cuda decode: no batching benefit anyway
                    measured[key] = 1
                    return 1
                # A 256px tile of one temporal clip is a small ViT problem (~1.3k tokens for a
                # 5-latent-frame clip): roughly 0.3 GB per concurrent tile covers activations
                # and the decoded pixels. Under DynamicVRAM 'free' runs low by design and staged
                # weights are evicted on pressure, so keep only a slim reserve; a real OOM is
                # caught by the sequential fallback in the decode loop.
                per_tile = 0.3 * (1024 ** 3) * max(z.shape[-3], 1) / 5.0
                usable = max(free - 0.8 * (1024 ** 3), per_tile)
                measured[key] = int(max(1, min(6, usable // max(per_tile, 1))))
                logging.info(f"[MiniMaxH3Speed] VAE tile batch size {measured[key]} "
                             f"({free / 1024**3:.1f} GB free)")
            return measured[key]

        fsm.tiled_decode = types.MethodType(
            lambda self, z, frames=slice(None): _batched_tiled_decode(self, z, batch_size_resolver, frames), fsm)

        # Ask ComfyUI's memory planner for the extra headroom the batched decode wants, so
        # DynamicVRAM evicts enough staged weight pages *before* the decode instead of the
        # resolver finding no free memory and collapsing to batch 1.
        target_batch = tile_batch_size if tile_batch_size > 0 else 4
        per_tile_headroom = int(0.35 * (1024 ** 3))
        original_estimate = getattr(vae, "memory_used_decode", None)
        if callable(original_estimate) and not getattr(vae, "_h3_batched_estimate", False):
            vae.memory_used_decode = (lambda shape, dtype, _orig=original_estimate:
                                      _orig(shape, dtype) + (target_batch - 1) * per_tile_headroom)
            vae._h3_batched_estimate = True
        logging.info("[MiniMaxH3Speed] batched tiled VAE decode installed")
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SpeedOptimizer": MiniMaxH3SpeedOptimizer,
    "MiniMaxH3VAESpeedup": MiniMaxH3VAESpeedup,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
