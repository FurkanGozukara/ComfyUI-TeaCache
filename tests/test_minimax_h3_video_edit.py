"""CPU tests for the MiniMax H3 masked video edit, edit composite and Time-to-Move nodes.

Needs ComfyUI's root on PYTHONPATH (or this repo inside ComfyUI/custom_nodes).
"""

import importlib.util
import inspect
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parents[1]))  # ComfyUI root when installed under custom_nodes

import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True
with mock.patch("torch.cuda.is_available", return_value=False):  # comfy_kitchen probes the GPU at import
    import comfy.model_base
    import comfy.model_patcher
    import comfy.nested_tensor
    import comfy.patcher_extension
    import comfy.sampler_helpers
    import comfy.utils
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN
    from comfy.ldm.minimax.vae import EncoderFCN3D, MiniMaxH3VideoVAE

    spec = importlib.util.spec_from_file_location("video_edit_under_test", ROOT / "minimax_h3/video_edit.py")
    video_edit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video_edit)

PIKA_SIGMAS = torch.tensor([1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0])


def tearDownModule():
    assert not torch.cuda.is_initialized(), "the tests must stay on the CPU"


def latent_t_for(frames):
    # MiniMaxH3VideoVAE.encode_temporal: one 5-latent chunk per started 17-frame clip, minus the 3 dropped latents
    return math.ceil(frames / 17) * 5 - 3


class FakeVae:
    """encode() keeps the H3 latent shape: 16x spatial, 17k+5 frames -> 5k+2 latents."""

    def __init__(self):
        self.frames = None

    def spacial_compression_encode(self):
        return 16

    def encode(self, frames):
        self.frames = frames
        f, h, w, _ = frames.shape
        return torch.full((1, 24, latent_t_for(f), h // 16, w // 16), 0.5)


class FakeH3:
    """The mask surface of comfy.model_base.MiniMaxH3 without building the model."""
    diffusion_model = types.SimpleNamespace(patch_size=(1, 2, 2))
    _pool_masks_to_token_grid = comfy.model_base.MiniMaxH3._pool_masks_to_token_grid
    _token_grid_masks = comfy.model_base.MiniMaxH3._token_grid_masks
    _denoise_mask_values = comfy.model_base.MiniMaxH3._denoise_mask_values


def av_latent(latent_t=7, lat_h=4, lat_w=6, audio_t=37, noise_mask=None):
    out = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, latent_t, lat_h, lat_w), torch.randn(1, 32, 2, audio_t)))}
    if noise_mask is not None:
        out["noise_mask"] = noise_mask
    return out


class TemporalGeometryTests(unittest.TestCase):
    def test_frame_groups(self):
        self.assertEqual(video_edit.latent_frame_starts(37)[-1], 124)
        self.assertEqual(video_edit.latent_frame_starts(2), [0, 1, 5])
        self.assertEqual(video_edit.latent_frame_starts(7), [0, 1, 5, 9, 13, 17, 18, 22])
        for frames, latents in ((124, 37), (5, 2), (22, 7)):
            self.assertEqual(latent_t_for(frames), latents)
            self.assertEqual((frames - 5) // 17 * 5 + 2, latents)  # the VAE wrapper's downscale_ratio (comfy/sd.py)
            self.assertEqual(video_edit.latent_frame_starts(latents), comfy.model_base.MiniMaxH3._latent_pixel_starts(latents))

    def test_geometry_follows_vae_code(self):
        params = inspect.signature(MiniMaxH3VideoVAE.__init__).parameters
        ratio_t = math.prod(params["time_down"].default)
        clip = params["clip_length"].default
        pre = (-clip) % ratio_t  # frame_pre_padding
        tokens = math.ceil(clip / ratio_t)  # tokens_chunk_size
        self.assertEqual((ratio_t, clip, pre, tokens, params["token_drop"].default), (4, 17, 3, 5, 3))
        self.assertEqual(math.prod(params["space_down"].default), 16)
        sizes = tuple(k * ratio_t + 1 - max(0, k * ratio_t - pre) for k in range(tokens))
        self.assertEqual(sizes, FRAME_PER_TOKEN)

    def test_causal_encoder_first_sees_each_group(self):
        torch.manual_seed(0)
        enc = EncoderFCN3D(ch=32, ch_mult=(1, 1, 1), space_down=(1, 1, 1), time_down=(2, 2, 1),
                           num_res_blocks=1, in_channels=3, z_channels=4)
        for p in enc.parameters():
            torch.nn.init.normal_(p, std=0.2)
        x = torch.randn(1, 3, 17, 12, 12)
        first_seen = {}
        with torch.no_grad():
            base = enc(x)
            for f in range(17):
                y = x.clone()
                y[:, :, f] += torch.randn_like(y[:, :, f])
                changed = (enc(y) - base).abs().amax(dim=(0, 1, 3, 4)) > 1e-6
                first_seen.setdefault(int(changed.nonzero()[0]), []).append(f)
        starts = video_edit.latent_frame_starts(5)
        self.assertEqual(first_seen, {j: list(range(a, b)) for j, (a, b) in enumerate(zip(starts, starts[1:]))})


class MaskedVideoEditTests(unittest.TestCase):
    def run_node(self, latent, images, mask, **kwargs):
        args = dict(grow_pixels=0, hold_frames=0, edge_feather=0)
        args.update(kwargs)
        return video_edit.MiniMaxH3MaskedVideoEdit().run(latent, FakeVae(), images=images, mask=mask, **args)

    def test_pass_through_without_images_or_mask(self):
        latent = av_latent()
        for images, mask in ((None, torch.ones(22, 8, 8)), (torch.rand(22, 8, 8, 3), None)):
            out, composite, canvas, report = self.run_node(latent, images, mask)
            self.assertIs(out, latent)
            self.assertIsNone(composite)
            self.assertIsNone(canvas)

    def test_cell_mask_dilation_and_hold(self):
        mask = torch.zeros(22, 64, 96)
        mask[10, 20, 20] = 1.0  # frame 10 belongs to latent 3 (frames 9-12), pixel (20, 20) to cell (1, 1)
        cells = video_edit.edit_cell_mask(mask, 7, 4, 6, 16, grow=0, hold=0)
        self.assertEqual(tuple(cells.shape), (7, 1, 4, 6))
        self.assertEqual(cells.nonzero().tolist(), [[3, 0, 1, 1]])
        cells = video_edit.edit_cell_mask(mask, 7, 4, 6, 16, grow=16, hold=2)
        self.assertEqual(sorted({j for j, *_ in cells.nonzero().tolist()}), [2, 3])  # frames 8-12
        expected = torch.zeros(4, 6)
        expected[:3, :3] = 1.0  # pixels 4-36 on both axes
        self.assertTrue(torch.equal(cells[3, 0], expected))

    def test_any_covered_pixel_marks_its_cell_at_source_resolution(self):
        mask = torch.zeros(1, 128, 192)  # one frame for the whole clip, twice the canvas size
        mask[0, 70:74, 150:154] = 1.0
        cells = video_edit.edit_cell_mask(mask, 7, 4, 6, 16, grow=0, hold=0)
        self.assertTrue(torch.equal(cells[:, 0, 2, 4], torch.ones(7)))
        self.assertEqual(int(cells.sum()), 7)

    def test_latent_noise_mask_and_outputs(self):
        images = torch.rand(22, 32, 48, 3)
        mask = torch.zeros(22, 32, 48)
        mask[10, 10:12, 10:12] = 1.0
        latent = av_latent()
        out, composite, canvas, report = self.run_node(latent, images, mask)
        video, audio = out["samples"].unbind()
        self.assertEqual(tuple(video.shape), (1, 24, 7, 4, 6))
        self.assertTrue(torch.all(video == 0.5))  # the encoded source replaced the zeros
        self.assertIs(audio, latent["samples"].unbind()[1])
        vmask, amask = out["noise_mask"].unbind()
        self.assertEqual(tuple(vmask.shape), (1, 1, 7, 4, 6))
        self.assertEqual(vmask.nonzero().tolist(), [[0, 0, 3, 1, 1]])
        self.assertTrue(torch.equal(amask, torch.ones(1, 32, 2, 37)))
        self.assertEqual(tuple(composite.shape), (22, 64, 96))
        self.assertEqual(tuple(canvas.shape), (22, 64, 96, 3))
        self.assertEqual(sorted(set(composite.nonzero()[:, 0].tolist())), [9, 10, 11, 12])
        self.assertTrue(torch.equal(composite[10, 16:32, 16:32], torch.ones(16, 16)))
        self.assertEqual(float(composite[10].sum()), 256.0)
        self.assertNotIn("WARNING", report)

        # core's prepare_mask + _denoise_mask_values: the cell mask lifts its whole 2x2 token, audio stays free
        masks = [comfy.sampler_helpers.prepare_mask(m, s.shape, "cpu") for m, s in zip(out["noise_mask"].unbind(), out["samples"].unbind())]
        packed, shapes = comfy.utils.pack_latents(masks)
        values = FakeH3()._denoise_mask_values(packed, shapes)
        self.assertEqual(sorted(values), ["denoise_mask"])
        self.assertEqual(tuple(values["denoise_mask"].shape), (1, 1, 7, 4, 6))
        self.assertEqual(values["denoise_mask"].nonzero()[:, 2:].tolist(), [[3, 0, 0], [3, 0, 1], [3, 1, 0], [3, 1, 1]])

    def test_keeps_existing_audio_mask_and_warns_on_frame_mismatch(self):
        audio_lock = torch.zeros(1, 32, 2, 37)
        latent = av_latent(noise_mask=comfy.nested_tensor.NestedTensor((torch.ones(1, 24, 7, 4, 6), audio_lock)))
        vae = FakeVae()
        with self.assertLogs(level="WARNING"):
            out, composite, canvas, report = video_edit.MiniMaxH3MaskedVideoEdit().run(
                latent, vae, 16, 2, images=torch.rand(20, 64, 96, 3), mask=torch.ones(20, 64, 96), edge_feather=8)
        self.assertIs(out["noise_mask"].unbind()[1], audio_lock)
        self.assertIn("padded", report)
        self.assertEqual(vae.frames.shape[0], 22)
        self.assertTrue(torch.equal(vae.frames[21], vae.frames[19]))
        self.assertEqual(tuple(composite.shape), (22, 64, 96))


    def test_a_clip_of_another_shape_is_centre_cropped_not_stretched(self):
        images = torch.full((22, 32, 64, 3), 0.5)  # 2:1 clip onto a square 64x64 canvas
        images[:, :, :16] = 0.0
        images[:, :, 48:] = 1.0
        mask = torch.zeros(22, 32, 64)
        mask[10, 4, 4] = 1.0     # left quarter: outside the square crop
        mask[10, 16, 32] = 1.0   # centre
        out, composite, canvas, report = self.run_node(av_latent(lat_h=4, lat_w=4), images, mask, pixel_composite=True)
        self.assertEqual(tuple(canvas.shape), (22, 64, 64, 3))
        self.assertTrue(torch.allclose(canvas, torch.full_like(canvas, 0.5), atol=1e-4))  # only the 0.5 middle
        vmask = out["noise_mask"].unbind()[0]
        self.assertEqual(vmask.nonzero().tolist(), [[0, 0, 3, 2, 2]])  # the centre dot, nothing from the left edge
        self.assertEqual(float(composite[10, :, :8].sum()), 0.0)
        self.assertIn("centre-cropped to 32x32", report)

    def test_canvas_grid_differences_keep_the_previous_resize(self):
        images = torch.rand(22, 64, 97, 3)  # 1 % wider than the 96x64 canvas
        mask = torch.zeros(22, 64, 97)
        out, composite, canvas, report = self.run_node(av_latent(), images, mask)
        expected = comfy.utils.common_upscale(images.movedim(-1, 1), 96, 64, "area", "disabled").clamp(0.0, 1.0).movedim(1, -1)
        self.assertTrue(torch.equal(canvas, expected))
        self.assertNotIn("cropped", report)
        self.assertTrue(torch.equal(video_edit.crop_to_aspect(mask, 96, 64), mask))


class MaskedEditCompositeTests(unittest.TestCase):
    def test_pass_through(self):
        images = torch.rand(3, 8, 8, 3)
        node = video_edit.MiniMaxH3MaskedEditComposite()
        self.assertIs(node.run(images)[0], images)
        self.assertIs(node.run(images, canvas_frames=torch.rand(3, 8, 8, 3))[0], images)
        self.assertIs(node.run(images, composite_mask=torch.ones(3, 8, 8))[0], images)

    def test_blend(self):
        decoded = torch.full((3, 8, 8, 3), 0.2)
        canvas = torch.full((3, 8, 8, 3), 0.8)
        mask = torch.zeros(3, 8, 8)
        mask[:, :, 4:] = 1.0
        mask[:, 0, 0] = 0.5
        out = video_edit.MiniMaxH3MaskedEditComposite().run(decoded, canvas, mask)[0]
        self.assertTrue(torch.allclose(out[:, :, 4:], torch.full((3, 8, 4, 3), 0.2)))
        self.assertTrue(torch.allclose(out[:, 1:, :4], torch.full((3, 7, 4, 3), 0.8)))
        self.assertTrue(torch.allclose(out[:, 0, 0], torch.full((3, 3), 0.5)))

    def test_fits_frames_and_size_to_the_decoded_video(self):
        decoded = torch.full((4, 8, 8, 3), 0.2)
        canvas = torch.full((2, 16, 16, 3), 0.8)
        mask = torch.zeros(5, 4, 4)
        with self.assertLogs(level="WARNING"):
            out = video_edit.MiniMaxH3MaskedEditComposite().run(decoded, canvas, mask)[0]
        self.assertEqual(tuple(out.shape), (4, 8, 8, 3))
        self.assertTrue(torch.allclose(out, torch.full((4, 8, 8, 3), 0.8)))


class TimeToMoveTests(unittest.TestCase):
    shapes = [torch.Size((1, 24, 7, 4, 4)), torch.Size((1, 32, 2, 5))]

    def test_anchor_alignment_124_to_37(self):
        mask = (torch.arange(124, dtype=torch.float32) / 123).view(-1, 1, 1).expand(124, 32, 48)
        keep = video_edit.ttm_keep_mask(mask, 37, 2, 3)
        self.assertEqual(tuple(keep.shape), (1, 1, 37, 2, 3))
        sampled = torch.round((1.0 - keep[0, 0, :, 0, 0]) * 123).long().tolist()
        expected = [(j // 5) * 17 + min(17, (j % 5 + 1) * 4 - 3) - 1 for j in range(37)]
        self.assertEqual(sampled, expected)
        self.assertEqual(sampled[:7], [0, 4, 8, 12, 16, 17, 21])
        self.assertEqual(sampled[-1], 123)

    def test_warns_on_frame_mismatch(self):
        with self.assertLogs(level="WARNING"):
            video_edit.ttm_keep_mask(torch.ones(100, 8, 8), 37, 2, 2)

    def setup_run(self, start_step=1, end_step=2, denoise_mask=None, noise=None):
        mask = torch.zeros(22, 64, 64)
        mask[:, :32, :32] = 1.0  # the subject fills the top-left 2x2 token
        keep = video_edit.ttm_keep_mask(mask, 7, 4, 4)
        reference = torch.randn(self.shapes[0])
        wrapper = video_edit._TimeToMove(reference, keep, start_step, end_step)
        guider = types.SimpleNamespace(model_options={}, model_patcher=types.SimpleNamespace(load_device=torch.device("cpu")))
        audio = torch.randn(self.shapes[1])
        latent, _ = comfy.utils.pack_latents([torch.zeros(self.shapes[0]), audio])
        noise = torch.randn_like(latent) if noise is None else noise
        calls = {}

        def outer_sample(noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None, latent_shapes=None):
            calls.update(latent_image=latent_image, denoise_mask=denoise_mask)
            return "sampled"

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(outer_sample, guider, [wrapper])
        self.assertEqual(executor.execute(noise, latent, None, PIKA_SIGMAS, denoise_mask, None, True, 0, latent_shapes=self.shapes), "sampled")
        return guider, calls, reference, audio, keep

    def unpack(self, packed):
        return comfy.utils.unpack_latents(packed, self.shapes)

    def apply(self, guider, sigma, **kwargs):
        wrappers = comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, guider.model_options, is_model_options=True)
        seen = {}

        def apply_model(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kw):
            seen.update(kw)

        comfy.patcher_extension.WrapperExecutor.new_class_executor(apply_model, FakeH3(), wrappers).execute(
            None, torch.tensor([sigma]), None, None, None, {}, **kwargs)
        return seen

    def test_reference_replaces_the_video_and_keeps_audio(self):
        guider, calls, reference, audio, _ = self.setup_run()
        video, kept_audio = self.unpack(calls["latent_image"])
        self.assertTrue(torch.equal(video, reference))
        self.assertTrue(torch.equal(kept_audio, audio))
        self.assertTrue(torch.equal(calls["denoise_mask"], torch.ones(1, 1, 24 * 7 * 16 + 32 * 2 * 5)))

    def test_phase_floors_on_pika_sigmas(self):
        guider, calls, _, _, keep = self.setup_run(start_step=1, end_step=2)
        fn = guider.model_options["denoise_mask_function"]
        # step 0 (sigma 1.0) holds the whole video, step 1 (0.972973) only the subject, steps 2-5 are free
        for step, sigma in enumerate(PIKA_SIGMAS[:-1]):
            video, audio = self.unpack(fn(sigma.view(1), calls["denoise_mask"], extra_options={}))
            self.assertTrue(torch.all(audio == 1.0))
            expected = [torch.zeros_like(video), keep.expand_as(video)][step] if step < 2 else torch.ones_like(video)
            self.assertTrue(torch.equal(video, expected), step)
        self.assertEqual(float(keep[0, 0, 0, :2, :2].sum()), 0.0)
        self.assertEqual(float(keep.sum()), 7 * 12)

    def test_apply_model_follows_the_phases(self):
        guider, _, _, _, keep = self.setup_run(start_step=1, end_step=2)
        seen = self.apply(guider, 1.0, denoise_mask="stale", audio_denoise_mask="stale")
        self.assertEqual(sorted(seen), ["denoise_mask"])
        self.assertTrue(torch.equal(seen["denoise_mask"], torch.zeros(1, 1, 7, 4, 4)))
        seen = self.apply(guider, float(PIKA_SIGMAS[1]), denoise_mask="stale")
        self.assertTrue(torch.equal(seen["denoise_mask"], keep))
        self.assertEqual(self.apply(guider, 0.923077, denoise_mask="stale", audio_denoise_mask="stale"), {})

    def test_existing_audio_lock_survives_every_phase(self):
        lock, _ = comfy.utils.pack_latents([torch.ones(self.shapes[0]), torch.zeros(self.shapes[1])])
        guider, calls, _, _, _ = self.setup_run(denoise_mask=lock)
        fn = guider.model_options["denoise_mask_function"]
        for sigma in (1.0, 0.972973, 0.8):
            self.assertTrue(torch.all(self.unpack(fn(torch.tensor([sigma]), lock, extra_options={}))[1] == 0.0))
        self.assertTrue(torch.equal(self.apply(guider, 0.8)["audio_denoise_mask"], torch.zeros(1, 1, 2, 5)))

    def test_zero_noise_sampler_is_left_alone(self):
        latent_zero = torch.zeros(1, 1, 24 * 7 * 16 + 32 * 2 * 5)
        guider, calls, _, _, _ = self.setup_run(noise=latent_zero)
        self.assertEqual(guider.model_options, {})
        self.assertIsNone(calls["denoise_mask"])
        self.assertFalse(torch.any(calls["latent_image"][:, :, :24 * 7 * 16]))

    def test_node_registers_the_wrapper_on_a_model_clone(self):
        model = comfy.model_patcher.ModelPatcher(torch.nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))
        reference = {"samples": comfy.nested_tensor.NestedTensor((torch.randn(self.shapes[0]), torch.randn(self.shapes[1])))}
        (patched,) = video_edit.MiniMaxH3TimeToMove().run(model, reference, torch.ones(22, 64, 64), 1, 2)
        self.assertIsNot(patched, model)
        outer = comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
        self.assertEqual(model.get_wrappers(outer, video_edit.TTM_KEY), [])
        (wrapper,) = patched.get_wrappers(outer, video_edit.TTM_KEY)
        self.assertEqual(tuple(wrapper.keep.shape), (1, 1, 7, 4, 4))
        self.assertTrue(torch.all(wrapper.keep == 0.0))


class PathFromSubjectTests(unittest.TestCase):
    def test_path_starts_at_the_subject_centre(self):
        import json
        mask = torch.zeros((1, 100, 200))
        mask[0, 40:61, 20:41] = 1.0  # centre (30, 50)
        path = json.dumps([{"x": 100, "y": 10}, {"x": 150, "y": 20}])
        out = json.loads(video_edit.MiniMaxH3PathFromSubject().run(path, mask, "at the subject")[0])
        self.assertEqual(out, [{"x": 30, "y": 50}, {"x": 80, "y": 60}])
        drawn = json.loads(video_edit.MiniMaxH3PathFromSubject().run(path, mask, "as drawn")[0])
        self.assertEqual(drawn, json.loads(path))
        empty = json.loads(video_edit.MiniMaxH3PathFromSubject().run(path, torch.zeros((1, 100, 200)), "at the subject")[0])
        self.assertEqual(empty, json.loads(path))
        five = json.loads(video_edit.MiniMaxH3PathFromSubject().run(path, mask, "at the subject", 5)[0])
        self.assertEqual([p["x"] for p in five], [30, 42, 55, 68, 80])
        self.assertEqual(len(five), 5)


if __name__ == "__main__":
    unittest.main()
