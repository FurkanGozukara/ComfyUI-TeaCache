"""SVI 2.0 Pro clip orchestration using native ComfyUI Wan models/samplers.

Conditioning follows vita-epfl/Stable-Video-Infinity (svi_wan22) and Kijai's
WanImageToVideoSVIPro: fixed anchor, previous clean latents, normalized zeros.
Only one decoded clip is held in memory. No model code or global state is patched.
"""

import collections
import json
import logging
import os
import tempfile
from fractions import Fraction

import av
import torch

import comfy.latent_formats
import comfy.model_management as mm
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
import nodes
from comfy_api.latest import InputImpl
from comfy_extras.nodes_custom_sampler import SamplerCustom
from comfy_extras.nodes_model_advanced import ModelSamplingSD3


SVI_HIGH = "SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors"
SVI_LOW = "SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"
FAST_HIGH = "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_High.safetensors"
FAST_LOW = "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_Low.safetensors"


def load_svi_lora(model, name, strength):
    if strength == 0:
        return model
    state = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", name), safe_load=True)
    # Official adapters use unprefixed PEFT keys; the bundle deliberately keeps the originals.
    state = {("diffusion_model." + key if key.startswith("blocks.") else key)
             .replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B."): value
             for key, value in state.items()}
    return comfy.sd.load_lora_for_models(model, None, state, strength, 0)[0]


def prompt_schedule(text, clip_count):
    """One prompt per paragraph; repeat the final paragraph as necessary."""
    prompts = [part.strip() for part in text.replace("\r\n", "\n").split("\n\n") if part.strip()]
    if not prompts:
        prompts = [""]
    return [prompts[min(i, len(prompts) - 1)] for i in range(clip_count)]


def svi_conditioning(positive, negative, anchor, previous, frames, motion_count):
    count = (frames - 1) // 4 + 1
    image = comfy.latent_formats.Wan21().process_out(torch.zeros(
        (1, 16, count, *anchor.shape[-2:]), device=anchor.device, dtype=anchor.dtype))
    image[:, :, :1] = anchor[:, :, :1]
    motion_count = min(motion_count, count - 1, previous.shape[2] if previous is not None else 0)
    if motion_count:
        image[:, :, 1:1 + motion_count] = previous[:, :, -motion_count:].to(image)
    mask = torch.ones((1, 1, count, *anchor.shape[-2:]), device=anchor.device, dtype=anchor.dtype)
    mask[:, :, :1] = 0
    values = {"concat_latent_image": image, "concat_mask": mask}
    latent = {"samples": torch.zeros_like(image, device=mm.intermediate_device())}
    return (node_helpers.conditioning_set_values(positive, values),
            node_helpers.conditioning_set_values(negative, values), latent, motion_count)


class ClipWriter:
    def __init__(self, path, width, height, fps, crf):
        self.container = av.open(path, "w")
        self.stream = self.container.add_stream("libx264", rate=Fraction(str(fps)))
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf), "preset": "fast"}
        self.time_base = Fraction(1, 1) / Fraction(str(fps))
        self.frames = 0

    def write(self, image):
        frame = av.VideoFrame.from_ndarray(image.numpy(), format="rgb24")
        frame.pts = self.frames
        frame.time_base = self.time_base
        self.container.mux(self.stream.encode(frame))
        self.frames += 1

    def close(self):
        self.container.mux(self.stream.encode())
        self.container.close()


class SESVIProVideo:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {"required": {
            "high_model": ("MODEL",), "low_model": ("MODEL",),
            "clip": ("CLIP",), "vae": ("VAE",),
            "prompts": ("STRING", {"multiline": True, "default": "", "tooltip": "One paragraph per clip. Blank lines change the prompt. The last prompt repeats."}),
            "negative_prompt": ("STRING", {"multiline": True, "default": "static, slow motion, blurry, flicker, subtitles, low quality"}),
            "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
            "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
            "clip_count": ("INT", {"default": 3, "min": 1, "max": 1000000}),
            "frames_per_clip": ("INT", {"default": 81, "min": 5, "max": 4097, "step": 4}),
            "fps": ("FLOAT", {"default": 16, "min": 1, "max": 120}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 30, "min": 2, "max": 1000}),
            "cfg": ("FLOAT", {"default": 4, "min": 0, "max": 30, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
            "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
            "shift": ("FLOAT", {"default": 5, "min": 0.01, "max": 100, "step": 0.1}),
            "switch_sigma": ("FLOAT", {"default": 0.875, "min": 0, "max": 1, "step": 0.001}),
            "motion_latent_count": ("INT", {"default": 1, "min": 0, "max": 128, "tooltip": "1 = Pro default; 0 = independent clips with the same anchor."}),
            "svi_high_lora": (loras, {"default": SVI_HIGH}),
            "svi_low_lora": (loras, {"default": SVI_LOW}),
            "svi_strength": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.05}),
            "fast_mode": ("BOOLEAN", {"default": False, "tooltip": "Apply the selected paired speed LoRAs. Also set steps/CFG to the recipe you want; 4 steps / CFG 1 is the Seko recipe."}),
            "fast_high_lora": (loras, {"default": FAST_HIGH}),
            "fast_low_lora": (loras, {"default": FAST_LOW}),
            "fast_strength": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.05}),
            "tiled_vae": ("BOOLEAN", {"default": True}),
            "continue_video": ("BOOLEAN", {"default": False}),
            "append_source": ("BOOLEAN", {"default": True}),
            "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
        }, "optional": {
            "start_image": ("IMAGE",),
            "source_video": ("VIDEO", {"lazy": True}),
            "previous_latent": ("LATENT",),
        }}

    RETURN_TYPES = ("VIDEO", "LATENT", "STRING")
    RETURN_NAMES = ("video", "last_clip_latent", "report")
    FUNCTION = "generate"
    CATEGORY = "SECourses/SVI Pro"

    def check_lazy_status(self, continue_video=False, source_video=None, **kwargs):
        return ["source_video"] if continue_video and source_video is None else []

    def generate(self, high_model, low_model, clip, vae, prompts, negative_prompt,
                 width, height, clip_count, frames_per_clip, fps, seed, steps, cfg,
                 sampler_name, scheduler, shift, switch_sigma, motion_latent_count,
                 svi_high_lora, svi_low_lora, svi_strength, fast_mode, fast_high_lora,
                 fast_low_lora, fast_strength, tiled_vae, continue_video, append_source,
                 crf, start_image=None, source_video=None, previous_latent=None):
        if continue_video and source_video is None:
            raise ValueError("Enable/load a source video before turning on Continue Video.")
        if start_image is None and not (continue_video and source_video is not None):
            raise ValueError("SVI Pro needs a start image or an enabled source video.")
        width, height = max(16, width // 16 * 16), max(16, height // 16 * 16)
        frames_per_clip = max(5, (frames_per_clip - 1) // 4 * 4 + 1)
        motion_latent_count = min(motion_latent_count, max(0, (frames_per_clip - 5) // 4))
        previous = previous_latent["samples"].cpu() if previous_latent is not None else None
        if previous is not None and previous.shape[-2:] != (height // 8, width // 8):
            raise ValueError("Resume latent resolution must match the requested width and height.")
        fd, path = tempfile.mkstemp(prefix="svi_pro_", suffix=".mp4", dir=folder_paths.get_temp_directory())
        os.close(fd)
        writer = ClipWriter(path, width, height, fps, crf)
        source_frames = 0
        try:
            if continue_video and source_video is not None:
                tail = collections.deque(maxlen=4 * motion_latent_count + 1)
                with av.open(source_video.get_stream_source()) as source:
                    stream = source.streams.video[0]
                    rate = float(stream.average_rate or fps)
                    origin = None
                    emitted = 0
                    for index, frame in enumerate(source.decode(stream)):
                        mm.throw_exception_if_processing_interrupted()
                        timestamp = frame.time if frame.time is not None else index / rate
                        if origin is None:
                            origin = timestamp
                        pixels = torch.from_numpy(frame.reformat(width=width, height=height, format="rgb24").to_ndarray())
                        if start_image is None:
                            start_image = pixels.unsqueeze(0).float() / 255
                        target = int(round((timestamp - origin + 1 / rate) * fps))
                        while emitted < target:
                            tail.append(pixels)
                            if append_source:
                                writer.write(pixels)
                            emitted += 1
                    source_frames = writer.frames
                if not tail:
                    raise ValueError("The source video has no decodable frames.")
                if previous is None and motion_latent_count:
                    context = torch.stack(list(tail)).float() / 255
                    previous = vae.encode(context).cpu()
            image = comfy.utils.common_upscale(start_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            anchor = vae.encode(image[..., :3]).cpu()
            high_model = load_svi_lora(high_model, svi_high_lora, svi_strength)
            low_model = load_svi_lora(low_model, svi_low_lora, svi_strength)
            if fast_mode:
                high_model = nodes.LoraLoaderModelOnly().load_lora_model_only(high_model, fast_high_lora, fast_strength)[0]
                low_model = nodes.LoraLoaderModelOnly().load_lora_model_only(low_model, fast_low_lora, fast_strength)[0]
            high_model = ModelSamplingSD3().patch(high_model, shift)[0]
            low_model = ModelSamplingSD3().patch(low_model, shift)[0]
            sigmas = comfy.samplers.calculate_sigmas(high_model.get_model_object("model_sampling"), scheduler, steps).cpu()
            split = sum(s >= switch_sigma for s in sigmas[:-1].tolist())
            sampler = comfy.samplers.sampler_object(sampler_name)
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
            last_prompt, positive = None, None
            for index, prompt in enumerate(prompt_schedule(prompts, clip_count)):
                mm.throw_exception_if_processing_interrupted()
                clip_seed = (seed + index) % (2 ** 64)
                logging.info("[SVI Pro] clip %s/%s, seed %s", index + 1, clip_count, clip_seed)
                if prompt != last_prompt:
                    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
                    last_prompt = prompt
                pos, neg, latent, motion = svi_conditioning(positive, negative, anchor, previous,
                                                          frames_per_clip, motion_latent_count)
                if split:
                    latent = SamplerCustom.execute(high_model, True, clip_seed, cfg, pos, neg,
                                                   sampler, sigmas[:split + 1], latent)[0]
                if split < steps:
                    latent = SamplerCustom.execute(low_model, split == 0, clip_seed, cfg, pos, neg,
                                                   sampler, sigmas[split:], latent)[0]
                previous = latent["samples"].cpu()
                decoded = (vae.decode_tiled(previous, tile_x=64, tile_y=64, overlap=8)
                           if tiled_vae else vae.decode(previous))
                decoded = decoded.reshape(-1, height, width, decoded.shape[-1])
                # The replayed prefix contains the anchor frame plus four frames per motion latent.
                skip = 1 + 4 * motion if motion else (1 if index > 0 or source_frames else 0)
                for frame in decoded[skip:]:
                    writer.write((frame[..., :3].cpu().clamp(0, 1) * 255).round().to(torch.uint8))
                del decoded, latent, pos, neg
            writer.close()
        except BaseException:
            writer.container.close()
            raise
        report = json.dumps({"clips": clip_count, "frames": writer.frames, "fps": fps,
                             "seconds": writer.frames / fps, "source_frames": source_frames,
                             "high_steps": split, "low_steps": steps - split,
                             "first_seed": seed, "last_seed": (seed + clip_count - 1) % (2 ** 64)})
        return (InputImpl.VideoFromFile(path), {"samples": previous}, report)


NODE_CLASS_MAPPINGS = {"SESVIProVideo": SESVIProVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"SESVIProVideo": "SVI 2.0 Pro - Long Video / Continue Video"}
