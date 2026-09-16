import ast
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
HUNYUAN_ROOT = os.environ.get("HUNYUAN_ROOT")
CONFIGURE_RUNTIME_CANDIDATES = (REPO_ROOT / "docker" / "configure_runtime.py", Path("/tmp/configure_runtime.py"))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upstream_path(relative):
    if not HUNYUAN_ROOT:
        return None
    path = Path(HUNYUAN_ROOT) / relative
    return path if path.is_file() else None


class FakeRenderer:
    def __init__(self):
        self.default_resolution = 512

    def set_default_render_resolution(self, value):
        self.default_resolution = value

    def set_boundary_unreliable_scale(self, _value):
        pass

    def get_face_areas(self, from_one_index=True):
        import numpy as np
        return np.array([0.0, 1.0])

    def render_alpha(self, _elev, _azim, return_type="np"):
        import numpy as np
        return np.ones((1, 2, 2, 1), dtype=int)


class ConfigureRuntimeTransformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = None
        for candidate in CONFIGURE_RUNTIME_CANDIDATES:
            if candidate.is_file():
                cls.module = _load_module("configure_runtime", candidate)
                break
        if cls.module is None:
            raise unittest.SkipTest("configure_runtime.py is not available")

    def test_view_selection_limit_transform(self):
        content = "prefix\n        for idx in range(6):\nsuffix\n"
        patched = self.module.transform("hy3dpaint/utils/pipeline_utils.py", content)
        self.assertIn("range(min(6, max_selected_view_num, candidate_view_num))", patched)

    def test_transform_rejects_unpinned_source(self):
        with self.assertRaises(ValueError):
            self.module.transform("hy3dpaint/utils/pipeline_utils.py", "for idx in range(7):")


@unittest.skipUnless(_upstream_path("hy3dpaint/utils/pipeline_utils.py"), "HUNYUAN_ROOT upstream source not available")
class ViewProcessorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"numpy/torch unavailable: {exc}")
        cls.pipeline_utils = _load_module("hunyuan_pipeline_utils", _upstream_path("hy3dpaint/utils/pipeline_utils.py"))

    def _select(self, max_selected):
        processor = self.pipeline_utils.ViewProcessor(SimpleNamespace(), FakeRenderer())
        result = processor.bake_view_selection([0] * 6, list(range(6)), [1.0] * 6, max_selected)
        return processor, result

    def test_selection_respects_max_selected_view_num(self):
        processor, result = self._select(4)
        self.assertEqual(len(result), 3)
        for selected in result:
            self.assertEqual(len(selected), 4)
        self.assertEqual(processor.render.default_resolution, 512)

    def test_selection_keeps_six_views_when_requested(self):
        processor, result = self._select(6)
        for selected in result:
            self.assertEqual(len(selected), 6)
        self.assertEqual(processor.render.default_resolution, 512)


@unittest.skipUnless(_upstream_path("hy3dpaint/utils/multiview_utils.py"), "HUNYUAN_ROOT upstream source not available")
class MultiviewUtilsContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(_upstream_path("hy3dpaint/utils/multiview_utils.py").read_text(encoding="utf-8"))

    def _calls(self):
        return [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]

    @staticmethod
    def _func_name(node):
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        return None

    @staticmethod
    def _is_self_attr(node, attr):
        return (
            isinstance(node, ast.Attribute)
            and node.attr == attr
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    def _keywords(self, call):
        return {kw.arg: kw.value for kw in call.keywords if kw.arg}

    def test_diffusion_uses_configured_steps_not_scheduler_table(self):
        wired = []
        literal_table = []
        for call in self._calls():
            value = self._keywords(call).get("num_inference_steps")
            if value is None:
                continue
            if self._is_self_attr(value, "inference_steps"):
                wired.append(call)
            if isinstance(value, ast.Subscript):
                literal_table.append(call)
        self.assertTrue(wired, "num_inference_steps=self.inference_steps wiring is missing")
        self.assertFalse(literal_table, "num_inference_steps still reads the literal infer_steps_dict table")

    def test_progress_callback_is_forwarded(self):
        found = any(self._is_self_attr(self._keywords(call).get("callback_on_step_end"), "progress_callback") for call in self._calls())
        self.assertTrue(found, "callback_on_step_end=self.progress_callback wiring is missing")

    def test_seed_is_forwarded_not_hardcoded(self):
        seed_everything = any(
            self._func_name(call.func) == "seed_everything" and any(self._is_self_attr(arg, "seed") for arg in call.args)
            for call in self._calls()
        )
        manual_seed = any(
            self._func_name(call.func) == "manual_seed" and any(self._is_self_attr(arg, "seed") for arg in call.args)
            for call in self._calls()
        )
        self.assertTrue(seed_everything, "seed_everything(self.seed) call is missing")
        self.assertTrue(manual_seed, "manual_seed(self.seed) call is missing")

    def test_dino_device_configured_and_cuda_offload(self):
        dino_calls = [
            call
            for call in self._calls()
            if self._func_name(call.func) == "to"
            and isinstance(call.func, ast.Attribute)
            and self._is_self_attr(call.func.value, "dino_v2")
        ]
        configured = any(
            any(isinstance(arg, ast.Attribute) and arg.attr == "dino_device" for arg in call.args)
            for call in dino_calls
        )
        offloaded = any(
            any(isinstance(arg, ast.Constant) and arg.value == "cpu" for arg in call.args)
            for call in dino_calls
        )
        self.assertTrue(configured, "self.dino_v2.to(...dino_device...) call is missing")
        self.assertTrue(offloaded, "self.dino_v2.to('cpu') CUDA offload is missing")


if __name__ == "__main__":
    unittest.main()
