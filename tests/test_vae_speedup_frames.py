import unittest

import torch

from test_minimax_h3_optimizer import load_optimizer


class FakeH3Vae:
    """The tiled-decode surface of MiniMaxH3VideoVAE: 2x2 overlapping tiles, decoding = upscale + frame index."""
    vae_ratio = 4

    def split_tiles(self, size):
        half = size // 2
        return [0, half - 4], [half + 4, size - half + 4], [8]

    def blend(self, tail, tile, overlap, dim):
        tile = tile.clone()
        if dim == -2:
            tile[..., :overlap, :] = (tile[..., :overlap, :] + tail[..., -overlap:, :]) / 2
        else:
            tile[..., :, :overlap] = (tile[..., :, :overlap] + tail[..., :, -overlap:]) / 2
        return tile

    def _decode_pixels(self, z):
        frames = torch.arange(z.shape[2], dtype=z.dtype).view(1, 1, -1, 1, 1)
        up = torch.nn.functional.interpolate(z, scale_factor=(1, self.vae_ratio, self.vae_ratio))
        return up[:, :3] + frames


class VaeSpeedupFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_optimizer()

    def decode(self, *args):
        return self.module._batched_tiled_decode(FakeH3Vae(), torch.ones(1, 4, 5, 8, 8), lambda z, n: 2, *args)

    def test_without_frames_keeps_every_frame(self):
        # older ComfyUI cores call tiled_decode(z)
        self.assertEqual(self.decode().shape, (1, 3, 5, 32, 32))

    def test_frames_slice_matches_full_decode(self):
        # ComfyUI ec1537d calls tiled_decode(z, frames) and expects only those output frames
        full = self.decode()
        one = self.decode(slice(2, 3))
        self.assertEqual(one.shape, (1, 3, 1, 32, 32))
        self.assertTrue(torch.equal(one, full[:, :, 2:3]))

    def test_installed_method_accepts_both_call_shapes(self):
        import types
        fsm = FakeH3Vae()
        fsm.tiled_decode = types.MethodType(
            lambda self, z, frames=slice(None): self.module_decode(z, frames), fsm)
        fsm.module_decode = lambda z, frames: self.module._batched_tiled_decode(fsm, z, lambda zz, n: 1, frames)
        z = torch.ones(1, 4, 5, 8, 8)
        self.assertEqual(fsm.tiled_decode(z).shape[2], 5)
        self.assertEqual(fsm.tiled_decode(z, slice(0, 1)).shape[2], 1)


if __name__ == "__main__":
    unittest.main()
