import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

# Import before the per-test sys.modules patches, which would otherwise drop numpy (it cannot load twice).
import numpy  # noqa: F401
import torch  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


def load_face_refine(models_dir, folders):
    # face_refine needs only folder_paths here; stub ComfyUI so the test runs without it.
    comfy = types.ModuleType("comfy")
    nested = types.ModuleType("comfy.nested_tensor")
    comfy.nested_tensor = nested
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(models_dir)

    def get_folder_paths(key):
        return [str(p) for p in folders[key]]

    folder_paths.get_folder_paths = get_folder_paths
    stubs = {"comfy": comfy, "comfy.nested_tensor": nested, "folder_paths": folder_paths}
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("face_refine_under_test", ROOT / "minimax_h3" / "face_refine.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    module.folder_paths = folder_paths
    return module


def make_pack(insightface_root, *files):
    pack = Path(insightface_root) / "models" / "buffalo_l"
    pack.mkdir(parents=True, exist_ok=True)
    for name in files:
        (pack / name).write_bytes(b"onnx")
    return pack


class InsightFaceRootTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.comfy_models = base / "ComfyUI" / "models"
        self.swarm_models = base / "SwarmUI" / "Models"
        (self.swarm_models / "diffusion_models").mkdir(parents=True)
        self.folders = {"diffusion_models": [self.comfy_models / "diffusion_models", self.swarm_models / "diffusion_models"],
                        "checkpoints": [self.swarm_models / "Stable-Diffusion"]}
        self.download = mock.Mock()
        storage = types.ModuleType("insightface.utils.storage")
        storage.download = self.download
        utils = types.ModuleType("insightface.utils")
        utils.storage = storage
        package = types.ModuleType("insightface")
        package.utils = utils
        self.modules = mock.patch.dict(sys.modules, {"insightface": package, "insightface.utils": utils,
                                                     "insightface.utils.storage": storage})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()
        self.tmp.cleanup()

    def root(self, modules, pack="buffalo_l"):
        return load_face_refine(self.comfy_models, self.folders)._insightface_root(pack, modules)

    def test_complete_pack_next_to_swarm_models_is_used_despite_empty_comfy_folder(self):
        make_pack(self.comfy_models / "insightface")
        make_pack(self.swarm_models / "insightface", "det_10g.onnx", "w600k_r50.onnx")
        self.assertEqual(Path(self.root(("detection", "recognition"))), self.swarm_models / "insightface")
        self.download.assert_not_called()

    def test_incomplete_packs_download_again_into_comfy_folder(self):
        make_pack(self.comfy_models / "insightface")
        make_pack(self.swarm_models / "insightface", "det_10g.onnx")
        self.assertEqual(Path(self.root(("detection", "recognition"))), self.comfy_models / "insightface")
        self.download.assert_called_once_with("models", "buffalo_l", force=True, root=str(self.comfy_models / "insightface"))

    def test_incomplete_comfy_pack_is_refilled_from_the_kept_archive(self):
        folder = make_pack(self.comfy_models / "insightface")
        with zipfile.ZipFile(str(folder) + ".zip", "w") as archive:
            for name in ("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx"):
                archive.writestr(name, b"onnx")
        self.assertEqual(Path(self.root(("detection", "recognition"))), self.comfy_models / "insightface")
        self.assertTrue((folder / "w600k_r50.onnx").is_file())
        self.download.assert_not_called()

    def test_archive_without_the_needed_file_still_downloads(self):
        folder = make_pack(self.comfy_models / "insightface")
        with zipfile.ZipFile(str(folder) + ".zip", "w") as archive:
            archive.writestr("det_10g.onnx", b"onnx")
        self.root(("detection", "recognition"))
        self.download.assert_called_once()

    def test_detection_only_task_accepts_a_detector_only_pack(self):
        make_pack(self.swarm_models / "insightface", "det_10g.onnx")
        self.assertEqual(Path(self.root(("detection",))), self.swarm_models / "insightface")

    def test_comfy_folder_is_preferred_when_complete(self):
        make_pack(self.comfy_models / "insightface", "det_10g.onnx", "2d106det.onnx")
        make_pack(self.swarm_models / "insightface", "det_10g.onnx", "2d106det.onnx")
        self.assertEqual(Path(self.root(("detection", "landmark_2d_106"))), self.comfy_models / "insightface")
        self.download.assert_not_called()

    def test_missing_pack_folder_is_left_to_insightface_download(self):
        self.assertEqual(Path(self.root(("detection", "recognition"))), self.comfy_models / "insightface")
        self.download.assert_not_called()

    def test_registered_insightface_folder_is_searched(self):
        extra = Path(self.tmp.name) / "extra" / "insightface"
        make_pack(extra, "det_10g.onnx", "w600k_r50.onnx")
        self.folders["insightface"] = [extra]
        self.assertEqual(Path(self.root(("detection", "recognition"))), extra)

    def test_other_packs_keep_the_comfy_folder(self):
        make_pack(self.swarm_models / "insightface", "det_10g.onnx", "w600k_r50.onnx")
        self.assertEqual(Path(self.root(("detection",), pack="antelopev2")), self.comfy_models / "insightface")
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
