"""MiniMax H3 speed optimizations for ComfyUI, ported from NVlabs/Sana `sol-engine`
(models/minimax_h3/optimized, techniques/sparse_backends).

Three techniques, each independently switchable and each with a dense/eager fallback so a
failure degrades to the baseline instead of killing the run:

1. **FirstBlockCache** (`cache_line.py` in the reference): block 0 of the 50-block stack runs
   every step; when its output residual barely moved since the previous step, the remaining 49
   blocks are skipped and the cached tail residual is reused. This is the dominant win of the
   reference acceleration line (2.58x of its 3.97x total).

2. **Sol-Attn sparse attention** (`sol_attn_h3.py` in the reference): the packed
   [text | cond | audio | video] self-attention is routed sparsely with the prefix (everything
   before the target-video tail) as an exact KV sink, and the prefix's own query rows recomputed
   densely. The released policy: tau=1.0, `diag` threshold, first steps and first 2 blocks dense.
   The vendored kernel dispatches CuTe DSL on SM90/100/120 when available and Triton everywhere
   else (>= SM80), which covers RTX 30xx/40xx/50xx on Windows.

3. **Batched tiled VAE decode** (`vae_shard.py` in the reference): the video VAE decodes each
   spatial tile in its own launch; the tiles are identical in shape and the ViT decoder is
   batch-independent, so feeding them through as one batch is the same arithmetic, bit-identical,
   with far less launch overhead.

The AdaLN precompute of the reference line is already baked into ComfyUI's H3 checkpoints
(`adaln_t_table` curve basis), and the fused QKV projection / fused RMSNorm+RoPE / fused SwiGLU
already exist in ComfyUI core via comfy-kitchen, so those are not duplicated here.

GPU support is decided at runtime: the sparse kernel is compiled, checked against dense SDPA on
the model's own tensors (route-everything must match), micro-benchmarked against the incumbent
attention backend, and only kept if it is both correct and faster on *this* GPU.
"""

import copy
import logging
import math
import os
import time
import types

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.patcher_extension

_SOL_ATTN = None
_SOL_ATTN_ERROR = None


def _load_sol_attn():
    """Import the vendored sol_attn package once; remember why it failed if it does."""
    global _SOL_ATTN, _SOL_ATTN_ERROR
    if _SOL_ATTN is not None or _SOL_ATTN_ERROR is not None:
        return _SOL_ATTN
    try:
        import sys
        from . import sol_attn as vendored_pkg
        # The vendored CuTe backends import `sol_attn.*` absolutely; give the nested package a
        # top-level alias (unless a real installation already owns the name).
        sys.modules.setdefault("sol_attn", vendored_pkg)
        _SOL_ATTN = vendored_pkg.sol_attn
    except Exception as e:  # noqa: BLE001 - any import failure means dense fallback
        _SOL_ATTN_ERROR = f"{type(e).__name__}: {e}"
        logging.warning(f"[MiniMaxH3Speed] sol_attn unavailable, sparse attention disabled: {_SOL_ATTN_ERROR}")
    return _SOL_ATTN


def _dense_bthd(q, k, v):
    """SDPA over contiguous [B, T, H, D], same layout out."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2)


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
                 sparse_dense_layers, sparse_min_video_rows, verbose):
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

        # per-process sparse resolution: None = undecided, True/False = decided
        self.sparse_resolved = None if sparse_mode == "auto" else (sparse_mode == "enabled")
        self.sparse_failed_reason = None
        self.sparse_calls = 0
        self.dense_calls = 0
        self._gate_done = set()

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
        pct = self.step_index / max(self.total_steps, 1)
        # end bound is strict so the schedule's final step always computes (detail-setting step)
        return self.fbc_start_percent <= pct < self.fbc_end_percent

    def block_patch(self, index):
        def patch(args, extra):
            img = args["img"]
            original = extra["original_block"]
            self.layer = index
            if not self.fbc_enabled:
                return original(args)
            cache = self.caches.setdefault(self.cond_key, _CondCache())

            if index == 0:
                h_in = img.clone()
                out = original(args)["img"]
                resid = (out - h_in)
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

    # ------------------------------------------------------------------ sparse attention

    def _sparse_eligible(self, q, heads, kwargs):
        if self.sparse_resolved is False or self.sparse_failed_reason is not None:
            return False
        if not kwargs.get("skip_reshape", False) or kwargs.get("mask") is not None:
            return False
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != heads or q.shape[-1] != 128:
            return False
        if self.seq_len is None or self.video_start is None or q.shape[2] != self.seq_len:
            return False
        if not (0 < self.video_start < self.seq_len):
            return False
        if (self.seq_len - self.video_start) < self.sparse_min_video_rows:
            return False
        if self.layer < self.sparse_dense_layers:
            return False
        if self.step_index < math.ceil(self.sparse_dense_steps_pct * max(self.total_steps, 1)):
            return False
        return True

    def _gate_and_bench(self, sol_attn, qb, kb, vb, incumbent):
        """Route-everything correctness gate + micro-benchmark vs the incumbent backend.

        Runs once per (device, sequence-length) pair, on the model's own tensors. A sparse
        configuration that is wrong or slower than what the user already runs is disabled
        for the rest of the process.
        """
        key = (qb.device.index, qb.shape[1])
        if key in self._gate_done:
            return True
        t_compile = time.perf_counter()
        got = sol_attn(qb, kb, vb, tau=-1000.0, thresh_type="diag")
        torch.cuda.synchronize(qb.device)
        compile_s = time.perf_counter() - t_compile
        want = _dense_bthd(qb, kb, vb)
        rel = (torch.linalg.vector_norm(got.float() - want.float())
               / torch.linalg.vector_norm(want.float()).clamp_min(1e-12)).item()
        del got, want
        if rel > 0.02:
            self.sparse_failed_reason = f"correctness gate failed: route-all rel_l2 {rel:.5f} > 0.02"
            logging.warning(f"[MiniMaxH3Speed] {self.sparse_failed_reason}; using dense attention")
            return False

        def run_sparse():
            return sol_attn(qb, kb, vb, tau=self.sparse_tau, thresh_type="diag",
                            sink_start=0, sink_tokens=self.video_start)

        def timeit(fn, n=3):
            fn()
            torch.cuda.synchronize(qb.device)
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize(qb.device)
            return (time.perf_counter() - t0) / n

        t_sparse = timeit(run_sparse)
        t_incumbent = timeit(incumbent)
        speedup = t_incumbent / max(t_sparse, 1e-9)
        decided = speedup >= 1.05 if self.sparse_mode == "auto" else True
        logging.info(f"[MiniMaxH3Speed] sparse attention gate on {torch.cuda.get_device_name(qb.device)}: "
                     f"rel_l2 {rel:.5f}, kernel compile {compile_s:.1f}s, sparse {t_sparse * 1000:.2f} ms vs "
                     f"incumbent {t_incumbent * 1000:.2f} ms ({speedup:.2f}x) -> "
                     f"{'ENABLED' if decided else 'disabled (not faster on this GPU)'}")
        if not decided:
            self.sparse_resolved = False
            return False
        self.sparse_resolved = True
        self._gate_done.add(key)
        return True

    def attention_override(self, func, q, k, v, heads, **kwargs):
        try:
            if self._sparse_eligible(q, heads, kwargs):
                sol_attn = _load_sol_attn()
                if sol_attn is None:
                    self.sparse_failed_reason = _SOL_ATTN_ERROR or "sol_attn import failed"
                else:
                    return self._sparse_attention(sol_attn, func, q, k, v, heads, kwargs)
        except Exception as e:  # noqa: BLE001 - never let the sparse path break sampling
            self.sparse_failed_reason = f"{type(e).__name__}: {e}"
            logging.warning(f"[MiniMaxH3Speed] sparse attention failed, falling back to dense "
                            f"for the rest of this process: {self.sparse_failed_reason}")
        self.dense_calls += 1
        return func(q, k, v, heads, **kwargs)

    def _sparse_attention(self, sol_attn, func, q, k, v, heads, kwargs):
        # [1, H, S, D] -> contiguous bf16 [1, S, H, D] (the kernel contract)
        qb = q.transpose(1, 2).to(torch.bfloat16).contiguous()
        kb = k.transpose(1, 2).to(torch.bfloat16).contiguous()
        vb = v.transpose(1, 2).to(torch.bfloat16).contiguous()

        if self.sparse_resolved is not True or (qb.device.index, qb.shape[1]) not in self._gate_done:
            ok = self._gate_and_bench(sol_attn, qb, kb, vb,
                                      incumbent=lambda: func(q, k, v, heads, **kwargs))
            if not ok:
                self.dense_calls += 1
                return func(q, k, v, heads, **kwargs)

        out = sol_attn(qb, kb, vb, tau=self.sparse_tau, thresh_type="diag",
                       sink_start=0, sink_tokens=self.video_start)
        # The sink keeps the prefix exact as *keys*; its own *query* rows must be dense.
        vs = self.video_start
        out[:, :vs] = _dense_bthd(qb[:, :vs], kb, vb)
        self.sparse_calls += 1
        s = out.shape[1]
        return out.reshape(1, s, heads * 128).to(q.dtype)


class MiniMaxH3SpeedOptimizer:
    """Apply the NVlabs Sana sol-engine MiniMax H3 acceleration line to a loaded H3 model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "A MiniMax H3 diffusion model."}),
                "first_block_cache": ("BOOLEAN", {"default": True, "tooltip": "Skip the transformer tail when block 0's residual barely moved since the previous step (FirstBlockCache). The dominant speedup of the reference line."}),
                "fbc_threshold": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005, "tooltip": "Accumulated relative block-0 residual change below which steps are skipped. 0.08 is the NVlabs-advertised near-lossless policy; raise toward 0.15-0.20 for more speed at some quality cost."}),
                "fbc_start_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Never skip before this fraction of the schedule (early steps set global structure)."}),
                "fbc_end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Never skip after this fraction of the schedule (final steps set fine detail)."}),
                "fbc_max_consecutive": ("INT", {"default": 3, "min": 1, "max": 50, "tooltip": "Cap on consecutive skipped steps."}),
                "sparse_attention": (["auto", "enabled", "disabled"], {"default": "auto", "tooltip": "Sol-Attn sparse attention over the packed sequence. 'auto' verifies correctness and benchmarks against your current attention backend on this GPU, and keeps whichever is faster."}),
                "sparse_dense_steps_pct": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of early steps that always run dense attention (reference: 10 of 50)."}),
                "sparse_dense_layers": ("INT", {"default": 2, "min": 0, "max": 50, "tooltip": "First N transformer blocks always run dense attention (reference: 2)."}),
            },
            "optional": {
                "sparse_tau": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 100.0, "step": 0.1, "tooltip": "Sol-Attn routing threshold temperature. 1.0 is the released H3 policy; higher keeps fewer KV blocks."}),
                "sparse_min_video_rows": ("INT", {"default": 4096, "min": 0, "max": 1000000, "tooltip": "Skip sparse attention when the target-video sequence is shorter than this (short sequences gain nothing)."}),
                "fbc_cache_device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Where cached residuals live. 'cpu' saves VRAM at some transfer cost."}),
                "verbose": ("BOOLEAN", {"default": True}),
                "enable_speedup": ("BOOLEAN", {
                    "default": True,
                    "label_on": "ENABLED",
                    "label_off": "DISABLED (normal)",
                    "tooltip": "Master switch for the complete 4x preset path. Disable it to pass through the original model and disable the linked MiniMax H3 VAE Speedup.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "BOOLEAN")
    RETURN_NAMES = ("model", "speedup_enabled")
    FUNCTION = "apply"
    CATEGORY = "TeaCache/MiniMaxH3"
    TITLE = "MiniMax H3 Speed Optimizer"
    DESCRIPTION = ("4x-line MiniMax H3 acceleration from NVlabs Sana sol-engine: FirstBlockCache step "
                   "skipping plus Sol-Attn sparse attention with per-GPU auto-verification. "
                   "The master switch can also drive the paired VAE speedup node. "
                   "Techniques that don't work or don't win on your GPU fall back to the normal path automatically.")

    def apply(self, model, first_block_cache, fbc_threshold, fbc_start_percent, fbc_end_percent,
              fbc_max_consecutive, sparse_attention, sparse_dense_steps_pct, sparse_dense_layers,
              sparse_tau=1.0, sparse_min_video_rows=4096, fbc_cache_device="gpu", verbose=True,
              enable_speedup=True):
        if not enable_speedup:
            return (model, False)

        diffusion_model = model.get_model_object("diffusion_model")
        if type(diffusion_model).__name__ != "MiniMaxH3Model":
            raise ValueError(f"MiniMaxH3SpeedOptimizer requires a MiniMax H3 model, got {type(diffusion_model).__name__}. "
                             "Connect the MiniMax H3 diffusion model (e.g. minimax_h3_fl2va / ref2va).")
        if not first_block_cache and sparse_attention == "disabled":
            return (model, True)

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
        )

        def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
            payload = kwargs.get("minimax_payload") or {}
            opt.observe(timestep, transformer_options, payload.get("layout"))
            return executor(x, timestep, context, transformer_options, **kwargs)

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                               "minimax_h3_speed", wrapper)
        for i in range(opt.num_blocks):
            m.set_model_patch_replace(opt.block_patch(i), "dit", "double_block", i)

        if sparse_attention != "disabled":
            to = m.model_options.setdefault("transformer_options", {})
            previous = to.get("optimized_attention_override")

            if previous is None:
                to["optimized_attention_override"] = opt.attention_override
            else:
                def chained(func, *args, **kw):
                    def inner(q, k, v, heads, **kw2):
                        return previous(func, q, k, v, heads, **kw2)
                    return opt.attention_override(inner, *args, **kw)
                to["optimized_attention_override"] = chained

        m.model_options["minimax_h3_speed_optimizer"] = opt
        return (m, True)


def _batched_tiled_decode(self, z, batch_size_resolver):
    """Batched drop-in for MiniMaxH3VideoVAE.tiled_decode: same tiles, same blend, same canvas,
    but the (identical-shape) tile decodes run through the ViT decoder as batches."""
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
        tiles.extend(decoded.split(1, dim=0))
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
            lambda self, z: _batched_tiled_decode(self, z, batch_size_resolver), fsm)

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
