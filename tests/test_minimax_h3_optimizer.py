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
    modules["comfy_kitchen"] = types.ModuleType("comfy_kitchen")
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
        q = types.SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.bfloat16,
                                  ndim=4, shape=(1, 4, 257, 128))
        for step, expected in ((0, False), (1, False), (2, True), (5, True), (6, False), (7, False)):
            opt.step_index = step
            self.assertEqual(opt._sparse_eligible(q, 4, {"skip_reshape": True}), expected)
        self.assertFalse(opt.fbc_window_open())

    def test_native_views_scale_tokens_and_exact_prefix(self):
        opt = self.make_optimizer()
        packed = torch.randn(1, 257, 3, 4, 128, dtype=torch.float16)
        qb, kb, vb = packed.unbind(2)
        q, k, v = (x.transpose(1, 2) for x in (qb, kb, vb))
        native = mock.Mock(return_value=torch.zeros_like(qb))
        with mock.patch.object(self.module, "ck", types.SimpleNamespace(sol_attn=native)):
            out = opt._run_sparse("comfy_kitchen", q, k, v, tau=1, scale=0.25, prefix=65)
        args, kwargs = native.call_args
        for before, after in zip((qb, kb, vb), args):
            self.assertEqual(before.data_ptr(), after.data_ptr())
            self.assertEqual(before.stride(), after.stride())
        self.assertEqual(kwargs, dict(tau=1, scale=0.25, sink_blocks=[0, 2], sink_q=[0, 2], token_aug=256))
        self.assertTrue(torch.equal(out[:, :65], self.module._dense_bthd(qb[:, :65], kb, vb, scale=0.25)))
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out[:, 65:].count_nonzero(), 0)

    def test_rejected_shape_does_not_disable_other_shapes(self):
        opt = self.make_optimizer()
        q = torch.zeros(1, 4, 257, 128)
        incumbent = mock.Mock(return_value="dense")
        with mock.patch.object(opt, "_gate_and_bench", return_value=None) as gate:
            for _ in range(2):
                self.assertEqual(opt._sparse_attention(incumbent, q, q, q, 4, {}), "dense")
            self.assertEqual(gate.call_count, 1)
            q2 = q[:, :, :193]
            opt._sparse_attention(incumbent, q2, q2, q2, 4, {})
            self.assertEqual(gate.call_count, 2)

    def test_cache_key_includes_heads_prefix_dtype_scale_and_token_budget(self):
        opt = self.make_optimizer()
        q = torch.zeros(1, 4, 257, 128)
        base = opt._sparse_key(q, q, q, {})
        self.assertNotEqual(base, opt._sparse_key(q[:, :2], q[:, :2], q[:, :2], {}))
        self.assertNotEqual(base, opt._sparse_key(q.half(), q.half(), q.half(), {}))
        self.assertNotEqual(base, opt._sparse_key(q, q, q, {"scale": 0.25}))
        opt.video_start += 64
        self.assertNotEqual(base, opt._sparse_key(q, q, q, {}))
        opt.video_start -= 64
        opt.sparse_extra_tokens = 0
        self.assertNotEqual(base, opt._sparse_key(q, q, q, {}))

    def test_native_nan_gate_tries_verified_vendor(self):
        opt = self.make_optimizer(sparse_mode="enabled")
        q = torch.zeros(1, 4, 257, 128)
        out = torch.ones(1, 257, 4, 128)

        def run(backend, *args, **kwargs):
            return torch.full_like(out, float("nan")) if backend == "comfy_kitchen" else out

        kitchen = types.SimpleNamespace(sol_attn_is_available=lambda device: True)
        with mock.patch.object(self.module, "ck", kitchen), \
                mock.patch.object(opt, "_run_sparse", side_effect=run), \
                mock.patch.object(self.module, "_dense_bthd", return_value=out), \
                mock.patch.object(torch.cuda, "synchronize"), \
                mock.patch.object(torch.cuda, "get_device_name", return_value="test GPU"):
            self.assertEqual(opt._gate_and_bench(q, q, q, {}, lambda: out), "vendored")

    def test_head_layout_is_preserved(self):
        opt = self.make_optimizer()
        q = torch.zeros(1, 4, 257, 128)
        out = torch.arange(q.numel()).reshape(1, 257, 4, 128)
        opt._gate_done[opt._sparse_key(q, q, q, {})] = "comfy_kitchen"
        with mock.patch.object(opt, "_run_sparse", return_value=out):
            flat = opt._sparse_attention(None, q, q, q, 4, {})
            heads = opt._sparse_attention(None, q, q, q, 4, {"skip_output_reshape": True})
        self.assertTrue(torch.equal(flat, out.reshape(1, 257, 512)))
        self.assertTrue(torch.equal(heads, out.transpose(1, 2)))

    def test_new_widgets_are_appended_after_legacy_master_switch(self):
        names = list(self.module.MiniMaxH3SpeedOptimizer.INPUT_TYPES()["optional"])
        self.assertEqual(names[:5], ["sparse_tau", "sparse_min_video_rows", "fbc_cache_device", "verbose", "enable_speedup"])
        self.assertEqual(self.module.MiniMaxH3SpeedOptimizer().apply(
            "original", True, 0.08, 0.15, 0.95, 3, "auto", 0.2, 2, enable_speedup=False), ("original", False))


if __name__ == "__main__":
    unittest.main()
