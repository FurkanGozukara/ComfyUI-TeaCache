import contextlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_optimizer():
    # Exercise the optimizer on CPU without importing ComfyUI's GPU startup.
    comfy = types.ModuleType("comfy")
    modules = {"comfy": comfy}
    for name in ("model_management", "patcher_extension", "model_prefetch"):
        module = types.ModuleType(f"comfy.{name}")
        setattr(comfy, name, module)
        modules[f"comfy.{name}"] = module
    comfy.model_prefetch.pause_malloc_graph = contextlib.nullcontext
    native = types.SimpleNamespace(SparseAttnPatch=lambda **kwargs: types.SimpleNamespace(reset=lambda: None),
                                   h3_eligible=lambda *args: True, h3_sparse_attention=mock.Mock())
    extras = types.ModuleType("comfy_extras")
    extras.nodes_sparse_attention = native
    modules["comfy_extras"] = extras
    spec = importlib.util.spec_from_file_location("optimizer_under_test", ROOT / "minimax_h3/optimizer.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_optimizer()

    def make_optimizer(self, **overrides):
        kwargs = dict(num_blocks=50, fbc_enabled=True, fbc_threshold=0.08,
                      fbc_start_percent=0.15, fbc_end_percent=0.95, fbc_max_consecutive=3,
                      fbc_cache_device="gpu", sparse_mode="auto", sparse_tau=1,
                      sparse_dense_steps_pct=0.2, sparse_dense_layers=2,
                      sparse_min_video_rows=0, verbose=False)
        kwargs.update(overrides)
        opt = self.module.H3Optimizer(**kwargs)
        opt.seq_len, opt.video_start, opt.layer = 257, 65, 4
        opt.step_index, opt.total_steps = 2, 8
        return opt

    def test_final_step_never_reuses_tail_at_short_schedules(self):
        for steps in (4, 8, 20, 50):
            opt = self.make_optimizer(sparse_dense_last_steps=0)
            opt.total_steps, opt.step_index = steps, steps - 1
            self.assertFalse(opt.fbc_window_open())
            opt.step_index = steps - 2
            self.assertEqual(opt.fbc_window_open(), opt.step_index / steps < 0.95)

    def test_dense_schedule_applies_to_sparse_attention_and_cache(self):
        opt = self.make_optimizer(sparse_dense_last_steps=2)
        for step, expected in ((0, False), (1, False), (2, True), (5, True), (6, False), (7, False)):
            opt.step_index = step
            self.assertEqual(opt.sparse_window_open(4), expected)
        self.assertFalse(opt.fbc_window_open())
        opt.step_index = 2
        self.assertFalse(opt.sparse_window_open(1))

    def test_rejected_shape_does_not_disable_other_shapes(self):
        opt = self.make_optimizer()
        attn = mock.Mock(return_value="dense", heads=4, head_dim=128)
        h = torch.zeros(257, 512)
        with mock.patch.object(opt, "_gate_native", return_value=False) as gate:
            for _ in range(2):
                self.assertEqual(opt.native_attention(attn, h, None, {}, 4), "dense")
            self.assertEqual(gate.call_count, 1)
            opt.native_attention(attn, h[:193], None, {}, 4)
            self.assertEqual(gate.call_count, 2)

    def test_native_cache_key_includes_shape_dtype_prefix_and_token_budget(self):
        opt = self.make_optimizer()
        attn = types.SimpleNamespace(heads=4, head_dim=128)
        h = torch.zeros(257, 512)
        options = {"minimax_h3_layout": object()}
        base = opt._native_key(attn, h, options)
        self.assertNotEqual(base, opt._native_key(attn, h[:193], options))
        self.assertNotEqual(base, opt._native_key(attn, h.half(), options))
        opt.video_start += 64
        self.assertNotEqual(base, opt._native_key(attn, h, options))
        opt.video_start -= 64
        opt.sparse_extra_tokens = 0
        self.assertNotEqual(base, opt._native_key(attn, h, options))

    def test_failed_native_shape_falls_back_once(self):
        opt = self.make_optimizer(sparse_mode="enabled")
        attn = mock.Mock(return_value="dense", heads=4, head_dim=128)
        h = torch.zeros(257, 512)
        with mock.patch.object(self.module.native_sparse, "h3_sparse_attention", side_effect=RuntimeError("unsupported")) as sparse:
            for _ in range(2):
                self.assertEqual(opt.native_attention(attn, h, None, {}, 4), "dense")
            self.assertEqual(sparse.call_count, 1)
        self.assertIn("unsupported", opt.sparse_failed_reason)

    def test_existing_block_patch_composes_with_cache(self):
        opt = self.make_optimizer(fbc_enabled=True, sparse_mode="disabled")
        previous = mock.Mock(side_effect=lambda args, extra: {"img": args["img"] + 7})
        original = mock.Mock(side_effect=AssertionError("existing patch was bypassed"))
        out = opt.block_patch(0, previous=previous)({"img": torch.zeros(2, 4)}, {"original_block": original})
        self.assertTrue(torch.equal(out["img"], torch.full((2, 4), 7)))
        previous.assert_called_once()

    def test_new_widgets_are_appended_after_legacy_master_switch(self):
        names = list(self.module.MiniMaxH3SpeedOptimizer.INPUT_TYPES()["optional"])
        self.assertEqual(names[:5], ["sparse_tau", "sparse_min_video_rows", "fbc_cache_device", "verbose", "enable_speedup"])
        self.assertEqual(self.module.MiniMaxH3SpeedOptimizer().apply(
            "original", True, 0.08, 0.15, 0.95, 3, "auto", 0.2, 2, enable_speedup=False), ("original", False))

    def test_scrambled_widget_values_degrade_to_the_normal_path(self):
        node_class = self.module.MiniMaxH3SpeedOptimizer
        spec = node_class.INPUT_TYPES()
        defaults = {name: options[0]["default"] for name, (_kind, *options) in
                    {**spec["required"], **spec["optional"]}.items() if name != "model"}
        self.assertEqual(node_class._invalid_inputs({"model": "original", **defaults}), [])

        # The reported prompt: values rotated six slots by the frontend, after the partial
        # coercion of a prompt validation whose failure another extension had discarded.
        with self.assertLogs(level="WARNING") as logs:
            result = node_class().apply(
                "original", True, 1.0, 4096.0, "gpu", 1, False, 1.0, 0, sparse_tau=0.15,
                sparse_min_video_rows=0, fbc_cache_device=3, verbose=True, enable_speedup=True)
        self.assertEqual(result, ("original", False))
        for expected in ("fbc_start_percent=4096.0", "fbc_end_percent='gpu'", "sparse_attention=False", "fbc_cache_device=3"):
            self.assertIn(expected, logs.output[0])


if __name__ == "__main__":
    unittest.main()
