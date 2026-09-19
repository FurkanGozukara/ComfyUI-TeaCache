"""Run with the ComfyUI venv: python tests/test_svi_pro.py (CPU only)."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

COMFY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY))
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]
import cuda_malloc
import torch
import av

spec = importlib.util.spec_from_file_location("svi_pro_tested", Path(__file__).resolve().parents[1] / "svi_pro.py")
svi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svi)
sys.argv = [sys.argv[0]]


class SviTests(unittest.TestCase):
    def test_official_lora_keys_are_converted_without_mutating_file_state(self):
        state = {"blocks.0.cross_attn.k.lora_A.default.weight": torch.ones(1),
                 "blocks.0.cross_attn.k.lora_B.default.weight": torch.ones(1)}
        with patch.object(svi.folder_paths, "get_full_path_or_raise", return_value="local.safetensors"), \
             patch.object(svi.comfy.utils, "load_torch_file", return_value=state), \
             patch.object(svi.comfy.sd, "load_lora_for_models", return_value=("patched", None)) as loader:
            self.assertEqual(svi.load_svi_lora("base", "local", 1), "patched")
            self.assertEqual(set(loader.call_args.args[2]), {
                "diffusion_model.blocks.0.cross_attn.k.lora_A.weight",
                "diffusion_model.blocks.0.cross_attn.k.lora_B.weight"})
            self.assertIn("blocks.0.cross_attn.k.lora_A.default.weight", state)
            self.assertEqual(svi.load_svi_lora("base", "local", 0), "base")
            loader.assert_called_once()

    def test_normalized_padding_anchor_motion_and_mask(self):
        anchor = torch.full((1, 16, 1, 2, 2), 2.0)
        previous = torch.arange(5.0).view(1, 1, 5, 1, 1).expand(1, 16, 5, 2, 2)
        condition = [[torch.zeros(1), {}]]
        pos, neg, latent, motion = svi.svi_conditioning(condition, condition, anchor, previous, 17, 1)
        image = pos[0][1]["concat_latent_image"]
        torch.testing.assert_close(image[:, :, :1], anchor)
        torch.testing.assert_close(image[:, :, 1:2], previous[:, :, -1:])
        normalized = svi.comfy.latent_formats.Wan21().process_in(image[:, :, 2:])
        torch.testing.assert_close(normalized, torch.zeros_like(normalized))
        self.assertEqual(pos[0][1]["concat_mask"][0, 0, :, 0, 0].tolist(), [0, 1, 1, 1, 1])
        self.assertEqual(latent["samples"].shape, (1, 16, 5, 2, 2))
        self.assertEqual(condition[0][1], {})
        self.assertEqual(motion, 1)
        self.assertIs(neg[0][1]["concat_latent_image"], image)

    def test_first_clip_and_motion_disabled(self):
        anchor = torch.ones(1, 16, 1, 2, 2)
        condition = [[torch.zeros(1), {}]]
        for previous, count in [(None, 1), (torch.ones(1, 16, 3, 2, 2), 0)]:
            pos, _, _, motion = svi.svi_conditioning(condition, condition, anchor, previous, 17, count)
            image = svi.comfy.latent_formats.Wan21().process_in(pos[0][1]["concat_latent_image"][:, :, 1:])
            torch.testing.assert_close(image, torch.zeros_like(image))
            self.assertEqual(motion, 0)

    def test_prompt_schedule(self):
        self.assertEqual(svi.prompt_schedule("first\r\nline\r\n\r\nsecond", 3), ["first\nline", "second", "second"])
        self.assertEqual(svi.prompt_schedule("", 1), [""])

    def test_streamed_video_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "test.mp4")
            writer = svi.ClipWriter(path, 32, 16, 16, 18)
            for i in range(33):
                writer.write(torch.full((16, 32, 3), i * 5, dtype=torch.uint8))
            writer.close()
            with av.open(path) as video:
                frames = list(video.decode(video=0))
                self.assertEqual(len(frames), 33)
                self.assertAlmostEqual(frames[-1].time, 2.0, places=5)
                self.assertEqual(video.streams.video[0].average_rate, 16)

    def test_clip_seeds_boundaries_and_prompt_changes(self):
        class Clip:
            def tokenize(self, text):
                return text
            def encode_from_tokens_scheduled(self, text):
                return [[torch.zeros(1), {"text": text}]]
        class Vae:
            def encode(self, image):
                return torch.ones(1, 16, 1 + (image.shape[0] - 1) // 4, 2, 4)
            def decode(self, latent):
                return torch.full((1, 4 * (latent.shape[2] - 1) + 1, 16, 32, 3), 0.5)
        class Model:
            def get_model_object(self, name):
                return None
        calls = []
        def sample(model, add_noise, seed, cfg, pos, neg, sampler, sigmas, latent):
            calls.append((add_noise, seed, pos[0][1]["text"]))
            return ({"samples": latent["samples"] + 1},)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(svi.folder_paths, "get_temp_directory", return_value=directory), \
             patch.object(svi.nodes.LoraLoaderModelOnly, "load_lora_model_only", side_effect=lambda model, *args: (model,)), \
             patch.object(svi, "load_svi_lora", side_effect=lambda model, *args: model), \
             patch.object(svi.ModelSamplingSD3, "patch", side_effect=lambda model, *args: (model,)), \
             patch.object(svi.comfy.samplers, "calculate_sigmas", return_value=torch.tensor([1., .9, .7, .4, 0.])), \
             patch.object(svi.SamplerCustom, "execute", side_effect=sample):
            kwargs = dict(high_model=Model(), low_model=Model(), clip=Clip(), vae=Vae(), prompts="one\n\ntwo",
                          negative_prompt="", width=32, height=16, clip_count=3, frames_per_clip=17, fps=16,
                          seed=99, steps=4, cfg=1, sampler_name="euler", scheduler="simple", shift=5,
                          switch_sigma=.875, motion_latent_count=1, svi_high_lora="high", svi_low_lora="low",
                          svi_strength=1, fast_mode=False, fast_high_lora="", fast_low_lora="", fast_strength=1,
                          tiled_vae=False, continue_video=False, append_source=True, crf=18,
                          start_image=torch.ones(1, 16, 32, 3))
            video, latent, report = svi.SESVIProVideo().generate(**kwargs)
            self.assertEqual(video.get_frame_count(), 17 + 12 + 12)
            self.assertEqual(calls, [(True, 99, "one"), (False, 99, "one"), (True, 100, "two"),
                                     (False, 100, "two"), (True, 101, "two"), (False, 101, "two")])
            kwargs.update(continue_video=True, source_video=video, start_image=None, clip_count=1)
            resumed, _, _ = svi.SESVIProVideo().generate(**kwargs)
            self.assertEqual(resumed.get_frame_count(), 41 + 12)
            kwargs.update(append_source=False, previous_latent=latent)
            continuation, _, _ = svi.SESVIProVideo().generate(**kwargs)
            self.assertEqual(continuation.get_frame_count(), 12)
            kwargs.update(continue_video=False, source_video=None, previous_latent=None,
                          start_image=torch.ones(1, 16, 32, 3), frames_per_clip=5, clip_count=2,
                          motion_latent_count=128)
            short, _, _ = svi.SESVIProVideo().generate(**kwargs)
            self.assertEqual(short.get_frame_count(), 9)


if __name__ == "__main__":
    unittest.main()
