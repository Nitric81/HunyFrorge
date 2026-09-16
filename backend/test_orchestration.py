import asyncio
import base64
import hashlib
import json
import tempfile
import threading
import time
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from . import main
from .models import (
    GenerationSettings,
    JobCreate,
    JobStage,
    JobStatus,
    current_model_revision,
    current_runtime_config,
    load_generation_presets,
    now,
    resolve_generation_settings,
)
from .pipeline import DemoAdapter, HunyuanShapeAdapter, Pipeline, make_demo_glb, prepare_unity_geometry
from .storage import JobStore, sha256_file

IMAGE = "data:image/png;base64,aW1hZ2U="


def full_parameters(**overrides):
    settings = resolve_generation_settings("standard", {}).model_dump()
    settings.update(overrides)
    settings.update({"image": IMAGE, "seed": 17, "job_id": "test-job"})
    return settings


class RecordingAdapter(DemoAdapter):
    def __init__(self):
        self.calls = []
        self.failures = set()
        self.blockers = {}
        self.shape_bytes = None

    async def generate_shape(self, seed, parameters, output):
        self.calls.append("shape")
        blocker = self.blockers.get("shape")
        if blocker is not None:
            await asyncio.to_thread(blocker.wait, 30)
        if "shape" in self.failures:
            raise RuntimeError("injected shape failure")
        await super().generate_shape(seed, parameters, output)
        if self.shape_bytes is not None:
            output.write_bytes(self.shape_bytes)

    async def generate_textures(self, mesh, parameters, output):
        self.calls.append("texture")
        blocker = self.blockers.get("texture")
        if blocker is not None:
            await asyncio.to_thread(blocker.wait, 30)
        if "texture" in self.failures:
            raise RuntimeError("injected texture failure")
        await super().generate_textures(mesh, parameters, output)


def run_job(store, job, adapter=None):
    pipeline = Pipeline(store, adapter=adapter)
    asyncio.run(pipeline.run(job))
    return store.get(job.id)


def run_in_thread(pipeline, job):
    thread = threading.Thread(target=lambda: asyncio.run(pipeline.run(job)), daemon=True)
    thread.start()
    return thread


def wait_for(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def spawn_child(store, source, stage):
    child = JobStatus(
        project_id=source.project_id,
        backend=source.backend,
        seed=source.seed,
        texture=source.texture,
        image=source.image,
        control_type=source.control_type,
        parameters=dict(source.parameters),
        preset=source.preset,
        parent_job_id=source.id,
        resume_from=stage,
        runtime_config=current_runtime_config(),
        model_revision=source.model_revision,
    )
    store.prepare_resume(source, child, stage)
    store.save(child)
    return child


class PresetContractTests(unittest.TestCase):
    def test_resolved_settings_match_shared_contract(self):
        contract = load_generation_presets()
        self.assertEqual(set(contract["presets"]), {"draft", "standard", "final"})
        for name, preset in contract["presets"].items():
            request = JobCreate(backend="demo", preset=name)
            for key, value in preset["settings"].items():
                self.assertEqual(request.parameters[key], value, f"{name}.{key}")
            for key in GenerationSettings.model_fields:
                self.assertIn(key, request.parameters)

    def test_overrides_merge_over_preset_defaults(self):
        request = JobCreate(backend="demo", preset="draft", parameters={"num_inference_steps": 3})
        self.assertEqual(request.parameters["num_inference_steps"], 3)
        self.assertEqual(request.parameters["octree_resolution"], 256)

    def test_invalid_combinations_nan_unknown_and_alias_conflicts_rejected(self):
        with self.assertRaises(ValueError):
            JobCreate(backend="demo", parameters={"unknown_setting": 1})
        with self.assertRaises(ValueError):
            JobCreate(backend="demo", parameters={"guidance_scale": float("nan")})
        with self.assertRaises(ValueError):
            JobCreate(backend="demo", parameters={"steps": 5, "num_inference_steps": 10})
        with self.assertRaises(ValueError):
            JobCreate(backend="demo", parameters={"unity_mode": "fast", "collision_mode": "convex_hull", "generate_collision": True})
        with self.assertRaises(ValueError):
            JobCreate(backend="demo", parameters={"num_inference_steps": 500})
        request = JobCreate(backend="demo", parameters={"steps": 9})
        self.assertEqual(request.parameters["num_inference_steps"], 9)
        self.assertNotIn("steps", request.parameters)
        allowed = JobCreate(backend="demo", parameters={"unity_mode": "full", "collision_mode": "convex_hull"})
        self.assertEqual(allowed.parameters["collision_mode"], "convex_hull")

    def test_caller_supplied_internals_are_not_trusted(self):
        request = JobCreate(backend="demo", seed=5, image=IMAGE, parameters={"seed": 999, "image": "evil", "job_id": "spoof"})
        self.assertNotEqual(request.parameters.get("seed"), 999)
        self.assertNotIn("image", request.parameters)
        self.assertNotIn("job_id", request.parameters)


class AdapterContractTests(unittest.TestCase):
    class ChunkedResponse:
        def __init__(self, payload):
            self._buffer = BytesIO(payload)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, size=-1):
            return self._buffer.read(size)

    def _capture(self, payload_bytes):
        captured = []

        def fake_urlopen(request, timeout):
            captured.append((request, timeout))
            return self.ChunkedResponse(payload_bytes)

        return captured, fake_urlopen

    def test_shape_request_streams_response_and_forwards_settings(self):
        body = b"glTF" + b"\x00" * (2 * 1024 * 1024 + 7) + b"tail"
        captured, fake = self._capture(body)
        parameters = full_parameters(num_inference_steps=33)
        adapter = HunyuanShapeAdapter("hunyuan3d-2.1")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "shape.glb"
            with patch("backend.pipeline.urllib.request.urlopen", side_effect=fake):
                asyncio.run(adapter.generate_shape(17, parameters, output))
            self.assertEqual(output.read_bytes(), body)
        request, timeout = captured[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["stage"], "shape")
        self.assertEqual(payload["job_id"], "test-job")
        self.assertEqual(payload["seed"], 17)
        self.assertEqual(payload["image"], "aW1hZ2U=")
        self.assertNotIn("mesh", payload)
        for key in GenerationSettings.model_fields:
            self.assertIn(key, payload["settings"])
        self.assertEqual(payload["settings"]["num_inference_steps"], 33)
        self.assertEqual(timeout, 3600)

    def test_texture_request_uses_mesh_token_without_base64_mesh(self):
        mesh_bytes = b"mesh-binary-content" * 100
        captured, fake = self._capture(b"glTF-textured")
        parameters = full_parameters()
        adapter = HunyuanShapeAdapter("hunyuan3d-2.1")
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "white-mesh.glb"
            mesh.write_bytes(mesh_bytes)
            output = Path(directory) / "textured.glb"
            with patch("backend.pipeline.urllib.request.urlopen", side_effect=fake):
                asyncio.run(adapter.generate_textures(mesh, parameters, output))
            self.assertEqual(output.read_bytes(), b"glTF-textured")
        request, _ = captured[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["stage"], "texture")
        self.assertEqual(payload["seed"], 17)
        token = payload["mesh_token"]
        self.assertEqual(token["job_id"], "test-job")
        self.assertEqual(token["artifact"], "white-mesh.glb")
        self.assertEqual(token["sha256"], hashlib.sha256(mesh_bytes).hexdigest())
        self.assertNotIn("mesh", set(payload) - {"mesh_token"})
        self.assertNotIn(base64.b64encode(mesh_bytes).decode(), request.data.decode())

    def test_missing_image_is_rejected(self):
        adapter = HunyuanShapeAdapter("hunyuan3d-2.1")
        parameters = full_parameters()
        parameters.pop("image")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "reference image"):
                asyncio.run(adapter.generate_shape(1, parameters, Path(directory) / "out.glb"))


class StorageContractTests(unittest.TestCase):
    def test_atomic_save_supports_concurrent_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo")
            store.save(job)
            errors = []
            stop = threading.Event()

            def writer():
                try:
                    while not stop.is_set():
                        job.progress = (job.progress + 1) % 100
                        store.save(job)
                except Exception as exc:
                    errors.append(exc)

            def reader():
                try:
                    while not stop.is_set():
                        loaded = store.get(job.id)
                        if loaded is not None:
                            json.dumps(loaded.model_dump(mode="json"))
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=writer) for _ in range(2)] + [threading.Thread(target=reader) for _ in range(4)]
            for thread in threads:
                thread.start()
            time.sleep(0.4)
            stop.set()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])

    def test_artifacts_are_immutable_and_paths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo")
            store.save(job)
            first = store.add_artifact(job, "white-mesh.glb", b"mesh-a")
            self.assertEqual(first, "artifacts/white-mesh.glb")
            self.assertEqual(store.add_artifact(job, "white-mesh.glb", b"mesh-a"), first)
            with self.assertRaises(ValueError):
                store.add_artifact(job, "white-mesh.glb", b"different")
            with self.assertRaises(ValueError):
                store.add_artifact(job, "../escape.glb", b"x")
            with self.assertRaises(ValueError):
                store.add_artifact(job, "nested/name.glb", b"x")

    def test_commit_artifact_moves_staging_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo")
            store.save(job)
            staging = store.path(job.id) / "staging" / "shape.glb"
            staging.parent.mkdir(parents=True, exist_ok=True)
            staging.write_bytes(b"streamed-glb")
            relative = store.commit_artifact(job, "white-mesh.glb", staging)
            self.assertEqual(relative, "artifacts/white-mesh.glb")
            self.assertFalse(staging.exists())
            target = store.path(job.id) / "artifacts" / "white-mesh.glb"
            self.assertEqual(sha256_file(target), hashlib.sha256(b"streamed-glb").hexdigest())
            self.assertTrue(job.artifact_readiness["white-mesh.glb"])

    def test_recover_marks_interrupted_jobs_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", stage=JobStage.GENERATING_SHAPE)
            job.stage_status["shape"].state = "running"
            job.stage_status["shape"].started_at = now()
            job.started_at = now()
            store.save(job)
            self.assertEqual(store.recover_incomplete_jobs(), 1)
            result = store.get(job.id)
            self.assertEqual(result.stage, JobStage.FAILED)
            self.assertEqual(result.error_code, "SERVICE_RESTARTED")
            self.assertIsNotNone(result.finished_at)
            self.assertEqual(result.stage_status["shape"].state, "failed")
            self.assertIsNotNone(result.stage_status["shape"].finished_at)

    def _checkpointed_job(self, store) -> JobStatus:
        job = JobStatus(backend="demo", runtime_config=current_runtime_config(), model_revision=current_model_revision())
        job.parameters.update(resolve_generation_settings("standard", {}).model_dump())
        store.save(job)
        store.add_artifact(job, "white-mesh.glb", make_demo_glb())
        job.shape_checkpoint = store.make_checkpoint(job, ["white-mesh.glb"])
        store.save(job)
        return job

    def test_checkpoint_rejects_corruption_mismatch_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = self._checkpointed_job(store)
            self.assertTrue(store.valid_checkpoint(job, "shape"))
            target = store.path(job.id) / "artifacts" / "white-mesh.glb"
            original = target.read_bytes()
            target.write_bytes(b"corrupted")
            self.assertFalse(store.valid_checkpoint(job, "shape"))
            target.write_bytes(original)
            self.assertTrue(store.valid_checkpoint(job, "shape"))
            job.parameters["num_inference_steps"] = 5
            self.assertFalse(store.valid_checkpoint(job, "shape"))
            job.parameters["num_inference_steps"] = 30
            with patch.dict("os.environ", {"HUNYUAN_SOURCE_REVISION": "deadbeef"}, clear=False):
                self.assertFalse(store.valid_checkpoint(job, "shape"))
            self.assertTrue(store.valid_checkpoint(job, "shape"))
            job.shape_checkpoint.artifacts = {"../job.json": "abc"}
            self.assertFalse(store.valid_checkpoint(job, "shape"))

    def test_checkpoint_rejects_environment_identity_and_structure_drift(self):
        from uuid import uuid4

        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = self._checkpointed_job(store)
            self.assertTrue(store.valid_checkpoint(job, "shape"))
            with patch.dict("os.environ", {"HUNYUAN_MODEL_REVISION": "other-rev"}, clear=False):
                self.assertFalse(store.valid_checkpoint(job, "shape"))
            self.assertTrue(store.valid_checkpoint(job, "shape"))
            job.texture = False
            self.assertFalse(store.valid_checkpoint(job, "shape"))
            job.texture = True
            self.assertTrue(store.valid_checkpoint(job, "shape"))
            original = job.shape_checkpoint.artifacts
            job.shape_checkpoint.artifacts = {}
            self.assertFalse(store.valid_checkpoint(job, "shape"))
            job.shape_checkpoint.artifacts = original
            (store.path(job.id) / "artifacts" / "white-mesh.glb").unlink()
            self.assertFalse(store.valid_checkpoint(job, "shape"))
            store.add_artifact(job, "white-mesh.glb", make_demo_glb())
            self.assertTrue(store.valid_checkpoint(job, "shape"))
            job.shape_checkpoint.source_job_id = uuid4()
            self.assertFalse(store.valid_checkpoint(job, "shape"))
            job.shape_checkpoint.source_job_id = job.id
            self.assertTrue(store.valid_checkpoint(job, "shape"))


class StageFlowTests(unittest.TestCase):
    def test_prepare_unity_geometry_contract(self):
        source = make_demo_glb(textured=True)
        produced, notes = prepare_unity_geometry(source, resolve_generation_settings("standard", {}))
        self.assertEqual(produced["unity-lod0.glb"], source)
        self.assertIn("unity-lod1.glb", produced)
        self.assertIn("unity-collision.glb", produced)
        produced_draft, _ = prepare_unity_geometry(source, resolve_generation_settings("draft", {}))
        self.assertEqual(set(produced_draft), {"unity-lod0.glb"})
        self.assertTrue(notes)

    def test_direct_job_resolves_default_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo")
            result = run_job(store, job)
            self.assertEqual(result.stage, JobStage.COMPLETE)
            for key in GenerationSettings.model_fields:
                self.assertIn(key, result.parameters)

    def test_shape_only_marks_texture_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=False)
            result = run_job(store, job)
            self.assertEqual(result.stage, JobStage.COMPLETE)
            self.assertEqual(result.stage_status["texture"].state, "skipped")
            self.assertNotIn("artifacts/textured-mesh.glb", result.artifacts)
            self.assertIn("artifacts/unity-package.zip", result.artifacts)

    def test_white_mesh_ready_while_texture_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True)
            store.save(job)
            release = threading.Event()
            adapter = RecordingAdapter()
            adapter.blockers["texture"] = release
            pipeline = Pipeline(store, adapter=adapter)
            thread = run_in_thread(pipeline, job)
            try:
                self.assertTrue(wait_for(lambda: (store.get(job.id) or job).stage_status["texture"].state == "running"))
                mid = store.get(job.id)
                self.assertTrue(mid.artifact_readiness["white-mesh.glb"])
                self.assertIn("artifacts/white-mesh.glb", mid.artifacts)
                self.assertEqual(mid.stage_status["shape"].state, "complete")
            finally:
                release.set()
                thread.join(30)
            self.assertEqual(store.get(job.id).stage, JobStage.COMPLETE)

    def test_failed_texture_resumes_without_rerunning_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True)
            failing = RecordingAdapter()
            failing.failures.add("texture")
            result = run_job(store, job, adapter=failing)
            self.assertEqual(result.stage, JobStage.FAILED)
            self.assertEqual(result.failed_stage, "texture")
            self.assertIsNotNone(result.shape_checkpoint)
            self.assertTrue(store.valid_checkpoint(result, "shape"))
            self.assertTrue(result.artifact_readiness["white-mesh.glb"])
            self.assertEqual(store.earliest_incomplete_stage(result), "texture")
            parent_snapshot = json.loads((store.path(result.id) / "job.json").read_text())

            child = spawn_child(store, result, "texture")
            self.assertIsNotNone(child.shape_checkpoint)
            self.assertEqual(child.shape_checkpoint.source_job_id, child.id)
            self.assertTrue((store.path(child.id) / "artifacts" / "white-mesh.glb").is_file())
            resumed_adapter = RecordingAdapter()
            resumed = run_job(store, child, adapter=resumed_adapter)
            self.assertEqual(resumed.stage, JobStage.COMPLETE)
            self.assertEqual(resumed_adapter.calls, ["texture"])
            self.assertIn("artifacts/unity-package.zip", resumed.artifacts)
            self.assertEqual(resumed.parent_job_id, result.id)
            parent_after = json.loads((store.path(result.id) / "job.json").read_text())
            self.assertEqual(parent_after, parent_snapshot)
            self.assertEqual(store.get(result.id).stage, JobStage.FAILED)

    def test_failed_unity_resumes_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True)
            with patch("backend.pipeline.prepare_unity_geometry", side_effect=RuntimeError("export exploded")):
                result = run_job(store, job)
            self.assertEqual(result.stage, JobStage.FAILED)
            self.assertEqual(result.failed_stage, "unity")
            self.assertIsNotNone(result.texture_checkpoint)
            self.assertEqual(store.earliest_incomplete_stage(result), "unity")
            child = spawn_child(store, result, "unity")
            adapter = RecordingAdapter()
            resumed = run_job(store, child, adapter=adapter)
            self.assertEqual(resumed.stage, JobStage.COMPLETE)
            self.assertEqual(adapter.calls, [])

    def test_corrupted_checkpoint_forces_shape_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True)
            failing = RecordingAdapter()
            failing.failures.add("texture")
            result = run_job(store, job, adapter=failing)
            self.assertEqual(store.earliest_incomplete_stage(result), "texture")
            (store.path(result.id) / "artifacts" / "white-mesh.glb").write_bytes(b"corrupted")
            self.assertFalse(store.valid_checkpoint(result, "shape"))
            self.assertEqual(store.earliest_incomplete_stage(result), "shape")
            decorated = store.get(result.id)
            self.assertNotIn("resume", decorated.available_actions)
            self.assertNotIn("resume_texture", decorated.available_actions)
            self.assertIn("restart", decorated.available_actions)
            self.assertEqual(decorated.resume_stage, "shape")

    def test_actions_reflect_verified_checkpoints_not_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True)
            failing = RecordingAdapter()
            failing.failures.add("texture")
            result = run_job(store, job, adapter=failing)
            decorated = store.get(result.id)
            self.assertEqual(decorated.resume_stage, "texture")
            self.assertEqual(decorated.available_actions, ["restart", "resume", "resume_texture"])
            job.shape_checkpoint.artifacts = {}
            store.save(job)
            decorated = store.get(result.id)
            self.assertNotIn("resume", decorated.available_actions)
            self.assertIn("restart", decorated.available_actions)
            running = JobStatus(backend="demo")
            store.save(running)
            self.assertEqual(store.get(running.id).available_actions, ["cancel"])
            running.stage = JobStage.CANCELLING
            store.save(running)
            self.assertEqual(store.get(running.id).available_actions, [])
            done = run_job(store, JobStatus(backend="demo", texture=False))
            self.assertEqual(store.get(done.id).available_actions, [])

    def test_timings_include_api_leaf_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            result = run_job(store, JobStatus(backend="demo", texture=True))
            for leaf in ("shape_request", "texture_request", "unity", "lod_generation", "collision_generation", "validation", "packaging"):
                self.assertIn(f"api.{leaf}", result.timings)
                self.assertIn("elapsed_seconds", result.timings[f"api.{leaf}"])
            self.assertIn("elapsed_seconds_total", result.timings)

    def test_real_backend_texture_requires_uv_and_material(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="hunyuan3d-2.1", image="data:image/png;base64,aW1hZ2U=", texture=True)
            result = run_job(store, job, adapter=RecordingAdapter())
            self.assertEqual(result.stage, JobStage.FAILED)
            self.assertEqual(result.failed_stage, "texture")
            self.assertIsNone(result.texture_checkpoint)
            self.assertIsNotNone(result.shape_checkpoint)
            self.assertEqual(store.earliest_incomplete_stage(result), "texture")

    def test_cancel_during_unity_waits_for_cpu_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True)
            store.save(job)
            release = threading.Event()

            def blocking(data, settings, telemetry=None):
                release.wait(30)
                return prepare_unity_geometry(data, settings, telemetry)

            pipeline = Pipeline(store)
            with patch("backend.pipeline.prepare_unity_geometry", side_effect=blocking):
                thread = run_in_thread(pipeline, job)
                try:
                    self.assertTrue(wait_for(lambda: (store.get(job.id) or job).stage_status["unity"].state == "running"))
                    pipeline.request_cancel(str(job.id))
                    mid = store.get(job.id)
                    self.assertIn(mid.stage, (JobStage.CANCELLATION_REQUESTED, JobStage.CANCELLING))
                    self.assertTrue(pipeline.is_busy())
                    time.sleep(0.1)
                    self.assertNotEqual(store.get(job.id).stage, JobStage.CANCELLED)
                finally:
                    release.set()
                    thread.join(30)
            result = store.get(job.id)
            self.assertEqual(result.stage, JobStage.CANCELLED)
            self.assertEqual(result.stage_status["unity"].state, "complete")
            self.assertIsNotNone(result.unity_checkpoint)
            self.assertIn("artifacts/white-mesh.glb", result.artifacts)
            self.assertNotIn(result.stage_status["validation"].state, ("running",))

    def test_cancel_before_packaging_preserves_validation_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True)
            store.save(job)
            release = threading.Event()

            def blocking_package(pipeline_self, job_obj, staging):
                release.wait(30)

            pipeline = Pipeline(store)
            with patch.object(Pipeline, "_write_package", blocking_package):
                thread = run_in_thread(pipeline, job)
                try:
                    self.assertTrue(wait_for(lambda: (store.path(job.id) / "artifacts" / "validation-report.json").is_file()))
                    self.assertTrue(wait_for(lambda: (store.get(job.id) or job).validation_checkpoint is not None))
                    pipeline.request_cancel(str(job.id))
                    mid = store.get(job.id)
                    self.assertIn(mid.stage, (JobStage.CANCELLATION_REQUESTED, JobStage.CANCELLING))
                    self.assertTrue(pipeline.is_busy())
                finally:
                    release.set()
                    thread.join(30)
            result = store.get(job.id)
            self.assertEqual(result.stage, JobStage.CANCELLED)
            self.assertIsNotNone(result.validation_checkpoint)
            self.assertFalse((store.path(job.id) / "artifacts" / "unity-package.zip").exists())

    def test_cancel_waits_for_adapter_and_preserves_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True)
            store.save(job)
            release = threading.Event()
            adapter = RecordingAdapter()
            adapter.blockers["shape"] = release
            pipeline = Pipeline(store, adapter=adapter)
            thread = run_in_thread(pipeline, job)
            try:
                self.assertTrue(wait_for(lambda: adapter.calls == ["shape"]))
                pipeline.request_cancel(str(job.id))
                mid = store.get(job.id)
                self.assertIn(mid.stage, (JobStage.CANCELLATION_REQUESTED, JobStage.CANCELLING))
                self.assertTrue(pipeline.is_busy())
                time.sleep(0.1)
                self.assertNotEqual(store.get(job.id).stage, JobStage.CANCELLED)
                self.assertTrue(pipeline.is_busy())
            finally:
                release.set()
                thread.join(30)
            result = store.get(job.id)
            self.assertEqual(result.stage, JobStage.CANCELLED)
            self.assertIn("artifacts/white-mesh.glb", result.artifacts)
            self.assertIsNotNone(result.shape_checkpoint)
            self.assertFalse(pipeline.is_busy())

    def test_invalid_shape_glb_fails_job(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo")
            adapter = RecordingAdapter()
            adapter.shape_bytes = b"not-a-real-glb"
            result = run_job(store, job, adapter=adapter)
            self.assertEqual(result.stage, JobStage.FAILED)
            self.assertEqual(result.failed_stage, "shape")
            self.assertEqual(result.error_code, "PIPELINE_ERROR")
            self.assertEqual(result.artifacts, [])

    def test_error_codes_distinguish_oom_and_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo")
            adapter = RecordingAdapter()
            adapter.failures.add("shape")
            with patch.object(RecordingAdapter, "generate_shape", lambda self, s, p, o: (_ for _ in ()).throw(RuntimeError("CUDA out of memory during inference"))):
                result = run_job(store, job, adapter=adapter)
            self.assertEqual(result.error_code, "CUDA_OUT_OF_MEMORY")
            job2 = JobStatus(backend="demo")
            with patch.object(RecordingAdapter, "generate_shape", lambda self, s, p, o: (_ for _ in ()).throw(RuntimeError("Hunyuan worker is unavailable or still loading"))):
                result2 = run_job(store, job2, adapter=adapter)
            self.assertEqual(result2.error_code, "WORKER_UNAVAILABLE")

    def test_lod0_preserves_source_bytes_and_zip_includes_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True)
            result = run_job(store, job)
            self.assertEqual(result.stage, JobStage.COMPLETE)
            artifacts = store.path(job.id) / "artifacts"
            lod0 = (artifacts / "unity-lod0.glb").read_bytes()
            textured = (artifacts / "textured-mesh.glb").read_bytes()
            self.assertEqual(lod0, textured)
            with zipfile.ZipFile(BytesIO((artifacts / "unity-package.zip").read_bytes())) as package:
                names = set(package.namelist())
            self.assertIn("Assets/HunyForge/white-mesh.glb", names)
            self.assertIn("Assets/HunyForge/textured-mesh.glb", names)
            self.assertIn("Assets/HunyForge/unity-lod0.glb", names)
            manifest = json.loads((artifacts / "hunyforge-manifest.json").read_text())
            self.assertNotIn("image", manifest)
            self.assertNotIn("control_data", manifest)
            self.assertTrue(manifest["unity_ready"])
            self.assertEqual(manifest["settings"]["texture_size"], 1024)

    def test_draft_preset_omits_optional_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True, preset="draft")
            result = run_job(store, job)
            self.assertEqual(result.stage, JobStage.COMPLETE)
            self.assertNotIn("artifacts/unity-lod1.glb", result.artifacts)
            self.assertNotIn("artifacts/unity-collision.glb", result.artifacts)
            manifest = json.loads((store.path(job.id) / "artifacts" / "hunyforge-manifest.json").read_text())
            self.assertEqual(manifest["lods"], ["unity-lod0.glb"])
            self.assertIsNone(manifest["collision"])

    def test_reserve_serializes_single_gpu_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(JobStore(Path(directory)))
            first = JobStatus(backend="demo")
            second = JobStatus(backend="demo")
            self.assertTrue(pipeline.reserve(first))
            self.assertFalse(pipeline.reserve(second))
            pipeline.release(first)
            self.assertTrue(pipeline.reserve(second))

    def test_legacy_job_without_checkpoints_restarts_only(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", stage=JobStage.FAILED)
            store.save(job)
            self.assertEqual(store.earliest_incomplete_stage(job), "shape")
            self.assertNotIn("resume", store.get(job.id).available_actions)
            self.assertIn("restart", store.get(job.id).available_actions)


class ApiOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        main.store = JobStore(Path(self._temp.name))
        main.pipeline = Pipeline(main.store)

    def tearDown(self):
        self._temp.cleanup()

    def _wait_stage(self, client, job_id, stages, timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["stage"] in stages:
                return job
            time.sleep(0.02)
        return client.get(f"/api/jobs/{job_id}").json()

    def test_presets_endpoint_returns_contract_and_schema(self):
        with TestClient(main.app) as client:
            response = client.get("/api/presets")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["presets"]["draft"]["settings"]["octree_resolution"], 256)
            self.assertIn("settings_schema", body)
            self.assertIn("texture_size", body["settings_schema"]["properties"])

    def test_health_reports_workers_and_runtime_config(self):
        with TestClient(main.app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertIsInstance(body["workers"], dict)
            self.assertIsInstance(body["runtime_config"], dict)

    def test_submit_while_busy_returns_409(self):
        release = threading.Event()
        adapter = RecordingAdapter()
        adapter.blockers["shape"] = release
        main.pipeline = Pipeline(main.store, adapter=adapter)
        with TestClient(main.app) as client:
            first = client.post("/api/jobs", json={"backend": "demo", "texture": True})
            self.assertEqual(first.status_code, 202)
            second = client.post("/api/jobs", json={"backend": "demo", "texture": True})
            self.assertEqual(second.status_code, 409)
            release.set()
            job = self._wait_stage(client, first.json()["id"], {"complete"})
            self.assertEqual(job["stage"], "complete")
            self.assertEqual(job["stage_status"]["shape"]["state"], "complete")

    def test_resume_restart_and_retry_endpoints(self):
        failing = RecordingAdapter()
        failing.failures.add("texture")
        main.pipeline = Pipeline(main.store, adapter=failing)
        with TestClient(main.app) as client:
            created = client.post("/api/jobs", json={"backend": "demo", "texture": True})
            self.assertEqual(created.status_code, 202)
            failed = self._wait_stage(client, created.json()["id"], {"failed"})
            self.assertEqual(failed["failed_stage"], "texture")
            self.assertIn("resume", failed["available_actions"])

            main.pipeline = Pipeline(main.store, adapter=RecordingAdapter())
            resumed = client.post(f"/api/jobs/{failed['id']}/resume")
            self.assertEqual(resumed.status_code, 202)
            self.assertEqual(resumed.json()["resume_from"], "texture")
            self.assertEqual(resumed.json()["parent_job_id"], failed["id"])
            done = self._wait_stage(client, resumed.json()["id"], {"complete"})
            self.assertEqual(done["stage"], "complete")

            restarted = client.post(f"/api/jobs/{failed['id']}/restart")
            self.assertEqual(restarted.status_code, 202)
            self.assertEqual(restarted.json()["resume_from"], "shape")
            self.assertEqual(self._wait_stage(client, restarted.json()["id"], {"complete"})["stage"], "complete")

            legacy = JobStatus(backend="demo", stage=JobStage.FAILED)
            main.store.save(legacy)
            self.assertEqual(client.post(f"/api/jobs/{legacy.id}/resume").status_code, 409)
            retried = client.post(f"/api/jobs/{legacy.id}/retry")
            self.assertEqual(retried.status_code, 202)
            self.assertEqual(retried.json()["resume_from"], "shape")
            self.assertEqual(self._wait_stage(client, retried.json()["id"], {"complete"})["stage"], "complete")

            running = client.post("/api/jobs", json={"backend": "demo"})
            for endpoint in ("retry", "resume", "restart"):
                self.assertEqual(client.post(f"/api/jobs/{running.json()['id']}/{endpoint}").status_code, 409)

    def test_resume_rejected_when_shape_checkpoint_corrupt(self):
        failing = RecordingAdapter()
        failing.failures.add("texture")
        main.pipeline = Pipeline(main.store, adapter=failing)
        with TestClient(main.app) as client:
            created = client.post("/api/jobs", json={"backend": "demo", "texture": True})
            self.assertEqual(created.status_code, 202)
            failed = self._wait_stage(client, created.json()["id"], {"failed"})
            self.assertIn("resume", failed["available_actions"])
            (main.store.path(failed["id"]) / "artifacts" / "white-mesh.glb").write_bytes(b"corrupted")
            status = client.get(f"/api/jobs/{failed['id']}").json()
            self.assertNotIn("resume", status["available_actions"])
            self.assertIn("restart", status["available_actions"])
            self.assertEqual(status["resume_stage"], "shape")
            self.assertEqual(client.post(f"/api/jobs/{failed['id']}/resume").status_code, 409)
            main.pipeline = Pipeline(main.store, adapter=RecordingAdapter())
            retried = client.post(f"/api/jobs/{failed['id']}/retry")
            self.assertEqual(retried.status_code, 202)
            self.assertEqual(retried.json()["resume_from"], "shape")

    def test_create_job_releases_reservation_when_save_fails(self):
        with TestClient(main.app, raise_server_exceptions=False) as client:
            with patch.object(main.store, "save", side_effect=OSError("disk full")):
                response = client.post("/api/jobs", json={"backend": "demo", "texture": True})
            self.assertEqual(response.status_code, 500)
            self.assertFalse(main.pipeline.is_busy())

    def test_public_job_output_excludes_image_and_control_payloads(self):
        main.pipeline = Pipeline(main.store, adapter=RecordingAdapter())
        with TestClient(main.app) as client:
            created = client.post("/api/jobs", json={
                "backend": "demo",
                "texture": True,
                "image": IMAGE,
                "control_type": "bbox",
                "control_data": "data:application/json;base64,Y29udHJvbA==",
            })
            self.assertEqual(created.status_code, 202)
            body = created.json()
            self.assertNotIn("image", body)
            self.assertNotIn("image", body["parameters"])
            self.assertNotIn("control_data", body["parameters"])
            job_id = body["id"]
            done = self._wait_stage(client, job_id, {"complete"})
            self.assertNotIn("image", done)
            self.assertNotIn("control_data", done["parameters"])
            listed = client.get("/api/jobs").json()
            self.assertNotIn("image", listed[0])
            events = client.get(f"/api/jobs/{job_id}/events")
            self.assertNotIn("aW1hZ2U=", events.text)
            self.assertNotIn("Y29udHJvbA==", events.text)
            inputs = client.get(f"/api/jobs/{job_id}/input")
            self.assertEqual(inputs.status_code, 200)
            self.assertEqual(inputs.json()["image"], IMAGE)
            self.assertEqual(inputs.json()["control_data"], "data:application/json;base64,Y29udHJvbA==")
            persisted = json.loads((main.store.path(job_id) / "job.json").read_text())
            self.assertEqual(persisted["image"], IMAGE)

    def test_validation_resume_records_manifest_provenance(self):
        calls = []
        original = Pipeline._write_package

        def flaky(pipeline_self, job_obj, staging):
            calls.append(str(job_obj.id))
            if len(calls) == 1:
                raise RuntimeError("packaging exploded")
            return original(pipeline_self, job_obj, staging)

        main.pipeline = Pipeline(main.store)
        with TestClient(main.app) as client:
            with patch.object(Pipeline, "_write_package", flaky):
                created = client.post("/api/jobs", json={"backend": "demo", "texture": True})
                self.assertEqual(created.status_code, 202)
                failed = self._wait_stage(client, created.json()["id"], {"failed"})
                self.assertEqual(failed["failed_stage"], "validation")
                self.assertEqual(failed["resume_stage"], "validation")
                self.assertIn("resume_unity", failed["available_actions"])
                resumed = client.post(f"/api/jobs/{failed['id']}/resume")
                self.assertEqual(resumed.status_code, 202)
                self.assertEqual(resumed.json()["resume_from"], "validation")
                done = self._wait_stage(client, resumed.json()["id"], {"complete"})
                self.assertEqual(done["stage"], "complete")
            report = client.get(f"/api/jobs/{done['id']}/validation").json()
            self.assertEqual(report["attempt_job_id"], done["id"])
            self.assertEqual(report["source_manifest_job_id"], failed["id"])
            parent_manifest = (main.store.path(failed["id"]) / "artifacts" / "hunyforge-manifest.json").read_bytes()
            child_manifest = (main.store.path(done["id"]) / "artifacts" / "hunyforge-manifest.json").read_bytes()
            self.assertEqual(parent_manifest, child_manifest)
            self.assertEqual(json.loads(parent_manifest)["job_id"], failed["id"])

    def test_cancel_waits_for_inflight_stage(self):
        release = threading.Event()
        adapter = RecordingAdapter()
        adapter.blockers["texture"] = release
        main.pipeline = Pipeline(main.store, adapter=adapter)
        with TestClient(main.app) as client:
            created = client.post("/api/jobs", json={"backend": "demo", "texture": True})
            job_id = created.json()["id"]
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["stage_status"]["texture"]["state"] == "running":
                    break
                time.sleep(0.02)
            cancelled = client.post(f"/api/jobs/{job_id}/cancel")
            self.assertEqual(cancelled.status_code, 200)
            self.assertIn(cancelled.json()["stage"], ("cancellation_requested", "cancelling"))
            self.assertTrue(main.pipeline.is_busy())
            release.set()
            result = self._wait_stage(client, job_id, {"cancelled"})
            self.assertEqual(result["stage"], "cancelled")
            self.assertIn("artifacts/white-mesh.glb", result["artifacts"])
            self.assertIsNotNone(result["shape_checkpoint"])


if __name__ == "__main__":
    unittest.main()
