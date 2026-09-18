import asyncio
import base64
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

from fastapi.testclient import TestClient

from .hunyuan_worker import MeshToken, PreviewRequest, WorkerRequest, app, generate, get_engine
from .models import GenerationSettings, JobStage, JobStatus, load_generation_presets
from .storage import sha256_file
from .worker_runtime import RuntimeEngine


SAMPLE_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ORK5CYII="
SAMPLE_IMAGE_URL = f"data:image/png;base64,{SAMPLE_IMAGE_B64}"
SAMPLE_IMAGE_BYTES = base64.b64decode(SAMPLE_IMAGE_B64)


class FakeImage:
    MAX_IMAGE_PIXELS = 16_777_216

    @classmethod
    def open(cls, stream):
        return cls(512, 512, "PNG")

    def __init__(self, width, height, fmt):
        self.width = width
        self.height = height
        self.format = fmt

    def load(self):
        pass

    def convert(self, mode):
        return self


class FakeCUDA:
    def __init__(self, available=False, initialized=False):
        self._available = available
        self._initialized = initialized

    def is_available(self):
        return self._available

    def is_initialized(self):
        return self._initialized

    def get_device_name(self, i):
        return "Fake GPU"

    def empty_cache(self):
        pass

    def synchronize(self):
        pass


class FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"
    __version__ = "2.5.0+cpu"

    def __init__(self, available=False, initialized=False):
        self.cuda = FakeCUDA(available, initialized)
        self.num_threads = None
        self.version = MagicMock(cuda=None)

    def set_num_threads(self, n):
        self.num_threads = n

    @contextmanager
    def inference_mode(self):
        yield

    def Generator(self, device):
        class G:
            def __init__(self, device):
                self.device = device

            def manual_seed(self, seed):
                self.seed = seed
                return self

        return G(device)

    def compile(self, model):
        return model


class FakeRemover:
    def __init__(self):
        self.images = []

    def __call__(self, image):
        self.images.append(image)
        return image


class RemoverFactory:
    calls = []

    @classmethod
    def reset(cls):
        cls.calls = []

    def __call__(self):
        r = FakeRemover()
        RemoverFactory.calls.append(r)
        return r


class FakeShapePipeline:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls = []
        self.export_calls = []
        self.flashvdm = None
        self.compiled = False

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return "latents"

    def _export(self, *args, **kwargs):
        self.export_calls.append((args, kwargs))

        class FakeMesh:
            def export(self, path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    f.write("glb")

        return [FakeMesh()]

    def enable_flashvdm(self, **kwargs):
        self.flashvdm = kwargs

    def compile(self):
        self.compiled = True


class FakeShapeFactory:
    from_pretrained_calls = []
    instances = []

    @classmethod
    def reset(cls):
        cls.from_pretrained_calls = []
        cls.instances = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        pipeline = FakeShapePipeline(*args, **kwargs)
        cls.instances.append(pipeline)
        return pipeline


class FakePaintConfig:
    def __init__(self, max_num_view, resolution):
        self.max_num_view = max_num_view
        self.resolution = resolution


class FakePaintPipeline:
    def __init__(self, conf):
        self.conf = conf
        self.calls = []
        self.view_processor = MagicMock()
        self.view_processor.bake_view_selection = MagicMock(return_value=([], [], []))
        self.view_processor.render_normal_multiview = MagicMock(return_value=[])
        self.view_processor.render_position_multiview = MagicMock(return_value=[])
        self.view_processor.bake_from_multiview = MagicMock(return_value=(None, None))
        self.view_processor.texture_inpaint = MagicMock(return_value=None)
        self.models = {
            "multiview_model": MagicMock(return_value={"albedo": [], "mr": []}),
            "super_model": MagicMock(return_value=None),
        }

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.view_processor.bake_view_selection([], [], [], 0)
        self.view_processor.render_normal_multiview([], [])
        self.view_processor.render_position_multiview([], [])
        for _ in range(2):
            self.models["super_model"](_)
        self.models["multiview_model"]([], [], prompt="")
        self.view_processor.bake_from_multiview([], [], [], [])
        self.view_processor.bake_from_multiview([], [], [], [])
        self.view_processor.texture_inpaint(None, None)
        self.view_processor.texture_inpaint(None, None)
        return str(Path(kwargs["output_mesh_path"]))


class FakePaintFactory:
    calls = []
    pipelines = []

    def __call__(self, conf):
        self.calls.append(conf)
        pipeline = FakePaintPipeline(conf)
        self.pipelines.append(pipeline)
        return pipeline


class FakeConvert:
    calls = []

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def call(cls, obj_path, glb_path):
        cls.calls.append((obj_path, glb_path))
        Path(glb_path).parent.mkdir(parents=True, exist_ok=True)
        with open(glb_path, "w") as f:
            f.write("textured glb")


class FakeOutImage:
    def save(self, buffer, format=None):
        buffer.write(b"fakepng")


class FakeT2IPipeline:
    def __init__(self):
        self.calls = []
        self.offloaded = False

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return MagicMock(images=[FakeOutImage()])


class FakeT2IFactory:
    from_pretrained_calls = []
    instances = []

    @classmethod
    def reset(cls):
        cls.from_pretrained_calls = []
        cls.instances = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        pipeline = FakeT2IPipeline()
        cls.instances.append(pipeline)
        return pipeline


def make_fake_dependencies(available=False, initialized=False):
    return {
        "torch": FakeTorch(available, initialized),
        "Image": FakeImage,
        "shape_factory": FakeShapeFactory,
        "paint_factory": FakePaintFactory(),
        "remover_factory": RemoverFactory(),
        "paint_config": FakePaintConfig,
        "convert": FakeConvert.call,
        "t2i_factory": FakeT2IFactory,
    }


def make_settings(preset="standard"):
    return GenerationSettings(**load_generation_presets()["presets"][preset]["settings"])


def write_job(root, job_id, seed=0, preset="standard", texture=True, image=SAMPLE_IMAGE_URL, settings=None, artifacts=None):
    if settings is None:
        settings = make_settings(preset)
    job_dir = root / "jobs" / str(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    parameters = {**settings.model_dump(), "image": image, "seed": seed, "job_id": str(job_id)}
    job = JobStatus(
        id=job_id,
        backend="hunyuan3d-2.1",
        seed=seed,
        texture=texture,
        image=image,
        parameters=parameters,
        preset=preset,
    )
    (job_dir / "job.json").write_text(job.model_dump_json(), encoding="utf-8")
    if artifacts:
        artifacts_dir = job_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name, data in artifacts.items():
            (artifacts_dir / name).write_bytes(data)
    return job_dir, job


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.model_path = cls.root / "models"
        (cls.model_path / "hunyuan3d-dit-v2-1").mkdir(parents=True)
        (cls.model_path / "hunyuan3d-dit-v2-1" / "model.fp16.ckpt").write_text("model", encoding="utf-8")
        (cls.model_path / "hunyuan3d-paintpbr-v2-1").mkdir(parents=True)
        cls.old_env = {}
        for k, v in {
            "HUNYUAN_MODEL_PATH": str(cls.model_path),
            "HUNYUAN_CPU_THREADS": "8",
            "HUNYUAN_FLASHVDM": "0",
            "HUNYUAN_COMPILE": "0",
            "HUNYUAN_DINO_DEVICE": "cpu",
            "HUNYFORGE_STAGE_ISOLATION": "0",
            "HUNYFORGE_MIN_AVAILABLE_MB": "0",
        }.items():
            cls.old_env[k] = os.environ.get(k)
            os.environ[k] = v

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cls.tmp.cleanup()

    def setUp(self):
        FakeShapeFactory.reset()
        FakePaintFactory.calls = []
        FakePaintFactory.pipelines = []
        FakeConvert.calls = []
        RemoverFactory.reset()
        app.state.engine = None

    def make_engine(self, deps=None):
        return RuntimeEngine(self.root, dependencies=deps)

    def make_request(self, job_id, stage="shape", seed=0, image=SAMPLE_IMAGE_URL, settings=None, mesh_token=None):
        if settings is None:
            settings = make_settings()
        return WorkerRequest(
            job_id=job_id,
            stage=stage,
            seed=seed,
            image=image,
            settings=settings,
            mesh_token=mesh_token,
        )

    def test_schema_bounds(self):
        settings = make_settings()
        with self.assertRaises(Exception):
            WorkerRequest(job_id=uuid.uuid4(), stage="shape", seed=-1, image=SAMPLE_IMAGE_URL, settings=settings)
        with self.assertRaises(Exception):
            WorkerRequest(job_id=uuid.uuid4(), stage="shape", seed=2**33, image=SAMPLE_IMAGE_URL, settings=settings)

    def test_token_mismatch(self):
        job_id = uuid.uuid4()
        token_id = uuid.uuid4()
        white = b"glb"
        h = hashlib.sha256(white).hexdigest()
        write_job(self.root, job_id, artifacts={"white-mesh.glb": white})
        token = MeshToken(job_id=token_id, artifact="white-mesh.glb", sha256=h)
        with self.assertRaises(Exception) as cm:
            self.make_request(job_id, stage="texture", mesh_token=token)
        self.assertIn("job_id", str(cm.exception).lower())

    def test_shape_token_rejected_by_schema(self):
        token = MeshToken(job_id=uuid.uuid4(), artifact="white-mesh.glb", sha256="0" * 64)
        with self.assertRaises(Exception):
            WorkerRequest(
                job_id=uuid.uuid4(),
                stage="shape",
                seed=0,
                image=SAMPLE_IMAGE_URL,
                settings=make_settings(),
                mesh_token=token,
            )

    def test_hash_mismatch(self):
        job_id = uuid.uuid4()
        white = b"glb"
        h = "0" * 64
        write_job(self.root, job_id, artifacts={"white-mesh.glb": white})
        token = MeshToken(job_id=job_id, artifact="white-mesh.glb", sha256=h)
        request = self.make_request(job_id, stage="texture", mesh_token=token)
        engine = self.make_engine(make_fake_dependencies())
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("hash", str(cm.exception).lower())

    def test_settings_mismatch(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        settings = make_settings("draft")
        request = self.make_request(job_id, settings=settings)
        engine = self.make_engine(make_fake_dependencies())
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("settings", str(cm.exception).lower())

    def test_image_mismatch(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id, image=SAMPLE_IMAGE_URL)
        request = self.make_request(job_id, image=f"data:image/png;base64,{base64.b64encode(b'other').decode()}")
        engine = self.make_engine(make_fake_dependencies())
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("image", str(cm.exception).lower())

    def test_seed_mismatch(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id, seed=123)
        request = self.make_request(job_id, seed=0)
        engine = self.make_engine(make_fake_dependencies())
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("seed", str(cm.exception).lower())

    def test_shape_factory_no_paint(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        request = self.make_request(job_id)
        engine = self.make_engine(make_fake_dependencies())
        engine.generate(request)
        self.assertEqual(len(FakeShapeFactory.from_pretrained_calls), 1)
        self.assertEqual(len(FakePaintFactory.calls), 0)
        self.assertIsNone(engine.shape_pipeline)
        self.assertIsNone(engine.paint_pipeline)
        self.assertEqual(engine.states["shape"], "unloaded")

    def test_shape_call_kwargs(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        request = self.make_request(job_id)
        engine = self.make_engine(make_fake_dependencies())
        engine.generate(request)
        _, kwargs = FakeShapeFactory.from_pretrained_calls[0]
        self.assertEqual(kwargs["device"], "cuda")
        self.assertEqual(kwargs["dtype"], "float16")
        self.assertEqual(kwargs["use_safetensors"], False)
        self.assertEqual(kwargs["variant"], "fp16")
        self.assertEqual(kwargs["subfolder"], "hunyuan3d-dit-v2-1")
        pipeline = FakeShapeFactory.instances[0]
        self.assertEqual(pipeline.calls[0]["num_inference_steps"], request.settings.num_inference_steps)
        self.assertEqual(pipeline.calls[0]["guidance_scale"], request.settings.guidance_scale)
        self.assertEqual(pipeline.calls[0]["octree_resolution"], request.settings.octree_resolution)
        self.assertEqual(pipeline.calls[0]["num_chunks"], request.settings.num_chunks)
        self.assertEqual(pipeline.calls[0]["box_v"], 1.01)
        self.assertEqual(pipeline.calls[0]["mc_level"], 0.0)
        self.assertEqual(pipeline.calls[0]["mc_algo"], "mc")
        self.assertEqual(pipeline.calls[0]["output_type"], "latent")
        self.assertEqual(pipeline.calls[0]["callback_steps"], 1)
        args, export_kwargs = pipeline.export_calls[0]
        self.assertEqual(export_kwargs["output_type"], "trimesh")
        self.assertEqual(export_kwargs["box_v"], 1.01)
        self.assertEqual(export_kwargs["mc_level"], 0.0)
        self.assertEqual(export_kwargs["num_chunks"], request.settings.num_chunks)
        self.assertEqual(export_kwargs["octree_resolution"], request.settings.octree_resolution)
        self.assertEqual(export_kwargs["mc_algo"], "mc")

    def test_seed_zero_preserved(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id, seed=0)
        request = self.make_request(job_id, seed=0)
        engine = self.make_engine(make_fake_dependencies())
        engine.generate(request)
        pipeline = FakeShapeFactory.instances[0]
        generator = pipeline.calls[0]["generator"]
        self.assertEqual(generator.seed, 0)

    def test_background_toggle(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        settings = make_settings()
        settings.remove_background = True
        request = self.make_request(job_id, settings=settings)
        engine = self.make_engine(make_fake_dependencies())
        engine.generate(request)
        self.assertEqual(len(RemoverFactory.calls), 1)
        RemoverFactory.reset()
        job_id = uuid.uuid4()
        settings = make_settings()
        settings.remove_background = False
        write_job(self.root, job_id, settings=settings)
        request = self.make_request(job_id, settings=settings)
        engine = self.make_engine(make_fake_dependencies())
        engine.generate(request)
        self.assertEqual(len(RemoverFactory.calls), 0)

    def test_shape_release_after_error(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        request = self.make_request(job_id)

        class FailingFactory:
            def from_pretrained(self, *args, **kwargs):
                raise RuntimeError("shape load failed")

        deps = make_fake_dependencies()
        deps["shape_factory"] = FailingFactory()
        engine = self.make_engine(deps)
        with self.assertRaises(RuntimeError):
            engine.generate(request)
        self.assertIsNone(engine.shape_pipeline)
        self.assertEqual(engine.states["shape"], "failed")
        self.assertEqual(engine.states["texture"], "unloaded")

    def test_texture_release_after_error(self):
        job_id = uuid.uuid4()
        white = b"glb"
        h = hashlib.sha256(white).hexdigest()
        write_job(self.root, job_id, artifacts={"white-mesh.glb": white})
        token = MeshToken(job_id=job_id, artifact="white-mesh.glb", sha256=h)
        request = self.make_request(job_id, stage="texture", mesh_token=token)

        class FailingFactory:
            def __call__(self, conf):
                raise RuntimeError("paint load failed")

        deps = make_fake_dependencies()
        deps["paint_factory"] = FailingFactory()
        engine = self.make_engine(deps)
        with self.assertRaises(RuntimeError):
            engine.generate(request)
        self.assertIsNone(engine.shape_pipeline)
        self.assertIsNone(engine.paint_pipeline)
        self.assertEqual(engine.states["texture"], "failed")

    def test_texture_config(self):
        job_id = uuid.uuid4()
        white = b"glb"
        h = hashlib.sha256(white).hexdigest()
        write_job(self.root, job_id, artifacts={"white-mesh.glb": white})
        token = MeshToken(job_id=job_id, artifact="white-mesh.glb", sha256=h)
        request = self.make_request(job_id, stage="texture", mesh_token=token)
        engine = self.make_engine(make_fake_dependencies())
        engine.generate(request)
        conf = FakePaintFactory.calls[0]
        self.assertEqual(conf.max_num_view, request.settings.max_num_view)
        self.assertEqual(conf.resolution, request.settings.multiview_resolution)
        self.assertEqual(conf.render_size, request.settings.render_size)
        self.assertEqual(conf.texture_size, request.settings.texture_size)
        self.assertEqual(conf.max_selected_view_num, request.settings.max_num_view)
        self.assertEqual(conf.texture_inference_steps, request.settings.texture_inference_steps)
        self.assertEqual(conf.texture_guidance_scale, request.settings.texture_guidance_scale)
        self.assertEqual(conf.face_count, request.settings.face_count)
        self.assertEqual(conf.seed, request.seed)
        self.assertEqual(conf.dino_device, os.environ.get("HUNYUAN_DINO_DEVICE", "cpu"))
        self.assertEqual(conf.multiview_pretrained_path, os.environ["HUNYUAN_MODEL_PATH"])
        self.assertEqual(conf.realesrgan_ckpt_path, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
        self.assertEqual(conf.multiview_cfg_path, "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
        self.assertEqual(conf.custom_pipeline, "hy3dpaint/hunyuanpaintpbr")

    def test_texture_mesh_path(self):
        job_id = uuid.uuid4()
        white = b"glb"
        h = hashlib.sha256(white).hexdigest()
        write_job(self.root, job_id, artifacts={"white-mesh.glb": white})
        token = MeshToken(job_id=job_id, artifact="white-mesh.glb", sha256=h)
        request = self.make_request(job_id, stage="texture", mesh_token=token)
        engine = self.make_engine(make_fake_dependencies())
        engine.generate(request)
        pipeline = FakePaintFactory.pipelines[0]
        mesh_path = pipeline.calls[0]["mesh_path"]
        parts = Path(mesh_path).parts
        self.assertIn("worker", parts)
        self.assertIn("texture", parts)
        self.assertIn("white-mesh.glb", parts)
        self.assertNotIn("artifacts", parts)

    def test_texture_error_no_fallback(self):
        job_id = uuid.uuid4()
        white = b"glb"
        h = hashlib.sha256(white).hexdigest()
        write_job(self.root, job_id, artifacts={"white-mesh.glb": white})
        token = MeshToken(job_id=job_id, artifact="white-mesh.glb", sha256=h)
        request = self.make_request(job_id, stage="texture", mesh_token=token)

        class FailingConvert:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("convert failed")

        deps = make_fake_dependencies()
        deps["convert"] = FailingConvert()
        engine = self.make_engine(deps)
        with self.assertRaises(RuntimeError):
            engine.generate(request)

    def test_symlink_escape(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        outside = self.root.parent / f"outside_{uuid.uuid4()}"
        outside.mkdir(parents=True, exist_ok=True)
        job_dir = self.root / "jobs" / str(job_id)
        try:
            (job_dir / "job.json").unlink()
            job_dir.rmdir()
            os.symlink(outside, job_dir, target_is_directory=True)
        except OSError:
            self.skipTest("platform does not support symlinks")
        request = self.make_request(job_id)
        engine = self.make_engine(make_fake_dependencies())
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("escapes", str(cm.exception).lower())

    def test_stage_parent_symlink_escape(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        job_dir = self.root / "jobs" / str(job_id)
        outside = self.root.parent / f"outside_{uuid.uuid4()}"
        outside.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(outside, job_dir / "worker", target_is_directory=True)
        except OSError:
            self.skipTest("platform does not support symlinks")
        request = self.make_request(job_id)
        engine = self.make_engine(make_fake_dependencies())
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("escapes", str(cm.exception).lower())

    def test_telemetry_symlink_escape(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        job_dir = self.root / "jobs" / str(job_id)
        outside = self.root.parent / f"outside_{uuid.uuid4()}"
        outside.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(outside, job_dir / "telemetry", target_is_directory=True)
        except OSError:
            self.skipTest("platform does not support symlinks")
        request = self.make_request(job_id)
        engine = self.make_engine(make_fake_dependencies())
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("escapes", str(cm.exception).lower())

    def test_stage_dir_symlink_rejected(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        job_dir = self.root / "jobs" / str(job_id)
        outside = self.root.parent / f"outside_{uuid.uuid4()}"
        outside.mkdir(parents=True, exist_ok=True)
        (job_dir / "worker").mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(outside, job_dir / "worker" / "shape", target_is_directory=True)
        except OSError:
            self.skipTest("platform does not support symlinks")
        request = self.make_request(job_id)
        engine = self.make_engine(make_fake_dependencies())
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("symlink", str(cm.exception).lower())

    def test_persisted_job_id_mismatch(self):
        job_id = uuid.uuid4()
        other_id = uuid.uuid4()
        job_dir, job = write_job(self.root, job_id)
        job.id = other_id
        (job_dir / "job.json").write_text(job.model_dump_json(), encoding="utf-8")
        request = self.make_request(job_id)
        engine = self.make_engine(make_fake_dependencies())
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("job_id", str(cm.exception).lower())

    def test_texture_stage_requires_textured_job(self):
        job_id = uuid.uuid4()
        white = b"glb"
        h = hashlib.sha256(white).hexdigest()
        write_job(self.root, job_id, texture=False, artifacts={"white-mesh.glb": white})
        token = MeshToken(job_id=job_id, artifact="white-mesh.glb", sha256=h)
        request = self.make_request(job_id, stage="texture", mesh_token=token)
        engine = self.make_engine(make_fake_dependencies())
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("shape-only", str(cm.exception).lower())

    def test_completed_job_rejects_rerun(self):
        job_id = uuid.uuid4()
        job_dir, job = write_job(self.root, job_id)
        job.stage = JobStage.COMPLETE
        (job_dir / "job.json").write_text(job.model_dump_json(), encoding="utf-8")
        request = self.make_request(job_id)
        engine = self.make_engine(make_fake_dependencies())
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("immutable", str(cm.exception).lower())

    def test_invocation_dirs_are_unique(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        request = self.make_request(job_id)
        engine = self.make_engine(make_fake_dependencies())
        first = engine.generate(request)
        first.write_text("first", encoding="utf-8")
        second = engine.generate(request)
        self.assertNotEqual(first, second)
        self.assertEqual(first.name, "white-mesh.glb")
        self.assertEqual(second.name, "white-mesh.glb")
        self.assertEqual(first.read_text(encoding="utf-8"), "first")
        self.assertNotEqual(first.parent, second.parent)

    def test_health(self):
        engine = self.make_engine(make_fake_dependencies(available=True))
        status = engine.health()
        self.assertTrue(status["ready"])
        self.assertFalse(status["busy"])
        self.assertIsNone(status["active_job_id"])
        self.assertIsNone(status["last_error"])

    def test_health_requires_cuda(self):
        engine = self.make_engine(make_fake_dependencies(available=False))
        status = engine.health()
        self.assertFalse(status["ready"])
        self.assertIn("cuda", (status["last_error"] or "").lower())

    def test_health_reports_missing_paint_model(self):
        engine = RuntimeEngine(self.root, dependencies=make_fake_dependencies(available=True))
        paint_dir = self.model_path / "hunyuan3d-paintpbr-v2-1"
        marker = self.model_path / "hunyuan3d-paintpbr-v2-1.hold"
        paint_dir.rename(marker)
        try:
            status = engine.health()
            self.assertFalse(status["ready"])
            self.assertIn("paint", (status["last_error"] or "").lower())
        finally:
            marker.rename(paint_dir)

    def test_health_blocked(self):
        engine = self.make_engine(make_fake_dependencies(available=True))
        app.state.engine = engine
        client = TestClient(app)

        def hold():
            engine._worker_lock.acquire()
            time.sleep(0.5)
            engine._worker_lock.release()

        t = threading.Thread(target=hold)
        t.start()
        time.sleep(0.05)
        try:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["ready"])
            job_id = uuid.uuid4()
            write_job(self.root, job_id)
            request = self.make_request(job_id)
            response = client.post("/generate", json=request.model_dump(mode="json"))
            self.assertEqual(response.status_code, 409)
        finally:
            t.join()

    def test_cancellation_lock_retained(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)

        class SlowFactory:
            def from_pretrained(self, *args, **kwargs):
                time.sleep(0.3)
                return FakeShapeFactory.from_pretrained(*args, **kwargs)

        deps = make_fake_dependencies()
        deps["shape_factory"] = SlowFactory()
        engine = self.make_engine(deps)
        app.state.engine = engine
        request = self.make_request(job_id)

        async def run():
            return await generate(request)

        async def main():
            task = asyncio.ensure_future(run())
            await asyncio.sleep(0.05)
            task.cancel()
            start = time.monotonic()
            try:
                await task
            except asyncio.CancelledError:
                pass
            elapsed = time.monotonic() - start
            self.assertGreaterEqual(elapsed, 0.25)
            self.assertFalse(engine._worker_lock.locked())

        asyncio.run(main())

    def test_image_format_requires_pil_detection(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        request = self.make_request(job_id)

        class GifImage(FakeImage):
            @classmethod
            def open(cls, stream):
                return cls(512, 512, "GIF")

        deps = make_fake_dependencies()
        deps["Image"] = GifImage
        engine = self.make_engine(deps)
        with self.assertRaises(ValueError) as cm:
            engine.generate(request)
        self.assertIn("format", str(cm.exception).lower())

    def test_provenance_failure_visible_in_health(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        request = self.make_request(job_id)
        engine = self.make_engine(make_fake_dependencies())
        original = engine._write_runtime_provenance
        engine._write_runtime_provenance = MagicMock(side_effect=RuntimeError("disk full"))
        result = engine.generate(request)
        self.assertTrue(result.name == "white-mesh.glb")
        self.assertIn("provenance", (engine.error or "").lower())
        status = engine.health()
        self.assertIn("provenance", (status["last_error"] or "").lower())
        engine._write_runtime_provenance = original

    def test_runtime_provenance(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        request = self.make_request(job_id)
        engine = self.make_engine(make_fake_dependencies())
        engine.generate(request)
        path = self.root / "jobs" / str(job_id) / "telemetry" / "shape-runtime.json"
        self.assertTrue(path.is_file())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["seed"], 0)
        self.assertEqual(data["settings"], request.settings.model_dump())
        self.assertEqual(data["shape_config"], {"box_v": 1.01, "mc_level": 0.0, "mc_algo": "mc"})
        self.assertEqual(data["process_id"], os.getpid())
        self.assertIn("source_revision", data)
        self.assertIn("model_revision", data)


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.t2i_path = cls.root / "t2i-model"
        cls.t2i_path.mkdir()
        cls.old_env = {}
        for k, v in {
            "HUNYFORGE_T2I_ENABLED": "1",
            "HUNYFORGE_T2I_MODEL_PATH": str(cls.t2i_path),
            "HUNYUAN_CPU_THREADS": "8",
            "HUNYFORGE_STAGE_ISOLATION": "0",
            "HUNYFORGE_MIN_AVAILABLE_MB": "0",
        }.items():
            cls.old_env[k] = os.environ.get(k)
            os.environ[k] = v

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cls.tmp.cleanup()

    def setUp(self):
        FakeT2IFactory.reset()
        app.state.engine = None

    def make_engine(self, deps=None):
        return RuntimeEngine(self.root, dependencies=deps or make_fake_dependencies())

    def make_request(self, prompt="a cube", image=None, seed=3):
        return PreviewRequest(prompt=prompt, seed=seed, image=image)

    def test_preview_generates_png(self):
        engine = self.make_engine()
        png, records = engine.generate_preview(self.make_request())
        self.assertEqual(png, b"fakepng")
        self.assertIn("t2i_model_loading", records)
        self.assertIn("t2i_inference", records)
        self.assertIn("model_release", records)
        self.assertIsNone(engine.t2i_pipeline)
        self.assertEqual(engine.states["reference"], "unloaded")
        pipeline = FakeT2IFactory.instances[0]
        self.assertTrue(pipeline.offloaded)
        call = pipeline.calls[0]
        self.assertEqual(call["prompt"], "a cube")
        self.assertIsNone(call["image"])
        self.assertEqual(call["num_inference_steps"], 4)
        self.assertEqual(call["generator"].seed, 3)
        telemetry_dir = self.root / "telemetry" / "previews"
        self.assertTrue(any(telemetry_dir.glob("*.json")))

    def test_preview_edit_mode_passes_image(self):
        engine = self.make_engine()
        engine.generate_preview(self.make_request(image=SAMPLE_IMAGE_URL))
        call = FakeT2IFactory.instances[0].calls[0]
        self.assertIsInstance(call["image"], list)
        self.assertEqual(len(call["image"]), 1)

    def test_preview_disabled(self):
        with unittest.mock.patch.dict(os.environ, {"HUNYFORGE_T2I_ENABLED": "0"}):
            engine = self.make_engine()
            with self.assertRaises(ValueError) as cm:
                engine.generate_preview(self.make_request())
            self.assertIn("disabled", str(cm.exception).lower())

    def test_preview_missing_model_path(self):
        with unittest.mock.patch.dict(os.environ, {"HUNYFORGE_T2I_MODEL_PATH": str(self.root / "missing")}):
            engine = self.make_engine()
            with self.assertRaises(ValueError) as cm:
                engine.generate_preview(self.make_request())
            self.assertIn("model path", str(cm.exception).lower())

    def test_preview_release_on_error(self):
        class FailingFactory:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError("t2i load failed")

        deps = make_fake_dependencies()
        deps["t2i_factory"] = FailingFactory
        engine = self.make_engine(deps)
        with self.assertRaises(RuntimeError):
            engine.generate_preview(self.make_request())
        self.assertIsNone(engine.t2i_pipeline)
        self.assertEqual(engine.states["reference"], "failed")

    def test_preview_schema_bounds(self):
        with self.assertRaises(Exception):
            PreviewRequest(prompt="", seed=0)
        with self.assertRaises(Exception):
            PreviewRequest(prompt="x" * 2001, seed=0)
        with self.assertRaises(Exception):
            PreviewRequest(prompt="a", seed=-1)
        with self.assertRaises(Exception):
            PreviewRequest(prompt="a", seed=0, width=640)
        with self.assertRaises(Exception):
            PreviewRequest(prompt="a", seed=0, num_inference_steps=99)

    def test_preview_busy_returns_409(self):
        engine = self.make_engine()
        app.state.engine = engine
        client = TestClient(app)
        engine._worker_lock.acquire()
        try:
            response = client.post("/preview", json={"prompt": "a cube", "seed": 3})
            self.assertEqual(response.status_code, 409)
        finally:
            engine._worker_lock.release()


if __name__ == "__main__":
    unittest.main()
