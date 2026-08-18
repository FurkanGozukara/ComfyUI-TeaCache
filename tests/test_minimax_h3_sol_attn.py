import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1] / "minimax_h3" / "sol_attn"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BackendDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interface = load_module("test_sol_attn_interface", ROOT / "interface.py")

    def test_sm89_prefers_cute_and_keeps_triton_fallback(self):
        self.assertEqual(self.interface._backend_for_arch((8, 9), cute_available=True), "cute_sm89")
        self.assertEqual(self.interface._backend_for_arch((8, 9), cute_available=False), "triton")

    def test_existing_architecture_dispatch_is_unchanged(self):
        self.assertEqual(self.interface._backend_for_arch((8, 6), cute_available=True), "triton")
        self.assertEqual(self.interface._backend_for_arch((9, 0), cute_available=True), "cute_sm90")
        self.assertEqual(self.interface._backend_for_arch((10, 0), cute_available=True), "cute_sm100")
        self.assertEqual(self.interface._backend_for_arch((12, 0), cute_available=True), "cute_sm120")
        with self.assertRaises(RuntimeError):
            self.interface._backend_for_arch((7, 5), cute_available=True)

    def test_public_backend_probe_does_not_compile(self):
        with mock.patch.object(self.interface.torch.cuda, "get_device_capability", return_value=(8, 9)):
            with mock.patch.object(self.interface, "_cute_runtime_available", return_value=False):
                self.assertEqual(self.interface.get_sol_attn_backend(0), "triton")


class LazyImportTests(unittest.TestCase):
    def test_architecture_entry_points_do_not_eagerly_require_cutlass(self):
        sm89 = load_module("test_sm89_entry", ROOT / "sm89" / "__init__.py")
        sm90 = load_module("test_sm90_entry", ROOT / "sm90" / "__init__.py")
        self.assertTrue(callable(sm89.make_kernel))
        self.assertTrue(callable(sm90.make_kernel))


if __name__ == "__main__":
    unittest.main()
