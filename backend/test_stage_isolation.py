import asyncio
import base64
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from . import hunyuan_worker
from .hunyuan_worker import app
from .stage_runner import main as runner_main, run_stage
from .test_worker import (
    SAMPLE_IMAGE_URL,
    make_fake_dependencies,
    make_settings,
    write_job,
)
from .hunyuan_worker import PreviewRequest, WorkerRequest
from .worker_runtime import RuntimeEngine


class FakeProcess:
    """Stands in for asyncio.subprocess.Process; behavior writes the result file."""

    def __init__(self, cmd, behavior):
        self._cmd = list(cmd)
        self._behavior = behavior
        self.pid = 4321
        self.returncode = None
        self.terminated = False
        self.killed = False

    async def communicate(self):
        result_path = Path(self._cmd[self._cmd.index("--result") + 1])
        stderr, rc = self._behavior(result_path)
        self.returncode = rc
        return b"", stderr

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


def fake_exec(behavior, captured):
    async def _exec(*cmd, **kwargs):
        proc = FakeProcess(cmd, behavior)
        captured.append(proc)
        return proc

    return _exec


def ok_generate(result_path: Path):
    artifact = result_path.parent / "white-mesh.glb"
    artifact.write_bytes(b"glb-bytes")
    result_path.write_text(json.dumps({"status": "ok", "artifact": str(artifact)}))
    return b"", 0


def ok_preview(result_path: Path):
    artifact = result_path.parent / "preview.png"
    artifact.write_bytes(b"png-bytes")
    result_path.write_text(json.dumps({"status": "ok", "artifact": str(artifact), "timings": {"t2i_inference": 1.5}}))
    return b"", 0


def oom_kill(result_path: Path):
    return b"", -9


def child_value_error(result_path: Path):
    result_path.write_text(json.dumps({"status": "error", "error_type": "ValueError", "error": "seed does not match persisted job seed"}))
    return b"", 1


def child_crash(result_path: Path):
    return b"RuntimeError: boom\nstack tail", 1


class IsolatedEndpointTests(unittest.TestCase):
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
            "HUNYFORGE_STAGE_ISOLATION": "1",
            "HUNYFORGE_MIN_AVAILABLE_MB": "0",
            "HUNYFORGE_DATA_ROOT": str(cls.root),
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
        app.state.engine = None
        app.state.active_stage = None
        self.client = TestClient(app)
        self.captured = []

    def generate_request(self):
        return {
            "job_id": str(uuid.uuid4()),
            "stage": "shape",
            "seed": 0,
            "image": SAMPLE_IMAGE_URL,
            "settings": make_settings().model_dump(mode="json"),
        }

    def spawn_slot(self):
        proc = self.captured[0]
        return Path(proc._cmd[proc._cmd.index("--request") + 1]).parent

    def test_generate_isolated_ok(self):
        with mock.patch.object(asyncio, "create_subprocess_exec", fake_exec(ok_generate, self.captured)):
            response = self.client.post("/generate", json=self.generate_request())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"glb-bytes")
        self.assertEqual(len(self.captured), 1)
        self.assertIn("backend.stage_runner", self.captured[0]._cmd)
        self.assertFalse(self.spawn_slot().exists())

    def test_generate_oom_kill_maps_503(self):
        with mock.patch.object(asyncio, "create_subprocess_exec", fake_exec(oom_kill, self.captured)):
            response = self.client.post("/generate", json=self.generate_request())
        self.assertEqual(response.status_code, 503)
        self.assertIn("worker_oom_killed", response.json()["detail"])

    def test_generate_child_value_error_maps_422(self):
        with mock.patch.object(asyncio, "create_subprocess_exec", fake_exec(child_value_error, self.captured)):
            response = self.client.post("/generate", json=self.generate_request())
        self.assertEqual(response.status_code, 422)
        self.assertIn("seed", response.json()["detail"])

    def test_generate_child_crash_maps_500_with_stderr_tail(self):
        with mock.patch.object(asyncio, "create_subprocess_exec", fake_exec(child_crash, self.captured)):
            response = self.client.post("/generate", json=self.generate_request())
        self.assertEqual(response.status_code, 500)
        self.assertIn("stack tail", response.json()["detail"])

    def test_preview_isolated_ok(self):
        with mock.patch.object(asyncio, "create_subprocess_exec", fake_exec(ok_preview, self.captured)):
            response = self.client.post("/preview", json={"prompt": "a cube", "seed": 3})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["image"], "data:image/png;base64," + base64.b64encode(b"png-bytes").decode("ascii"))
        self.assertEqual(body["timings"], {"t2i_inference": 1.5})
        self.assertFalse(self.spawn_slot().exists())

    def test_admission_gate_blocks_generate(self):
        with mock.patch.dict(os.environ, {"HUNYFORGE_MIN_AVAILABLE_MB": "12288"}), \
             mock.patch.object(hunyuan_worker, "memory_snapshot", lambda: {"available_mb": 100, "rss_mb": 50, "total_mb": 24000}):
            response = self.client.post("/generate", json=self.generate_request())
        self.assertEqual(response.status_code, 503)
        self.assertIn("insufficient_memory", response.json()["detail"])

    def test_admission_gate_blocks_preview(self):
        with mock.patch.dict(os.environ, {"HUNYFORGE_MIN_AVAILABLE_MB": "12288"}), \
             mock.patch.object(hunyuan_worker, "memory_snapshot", lambda: {"available_mb": 100, "rss_mb": 50, "total_mb": 24000}):
            response = self.client.post("/preview", json={"prompt": "a cube", "seed": 3})
        self.assertEqual(response.status_code, 503)
        self.assertIn("insufficient_memory", response.json()["detail"])

    def test_admission_passes_with_headroom(self):
        with mock.patch.dict(os.environ, {"HUNYFORGE_MIN_AVAILABLE_MB": "12288"}), \
             mock.patch.object(hunyuan_worker, "memory_snapshot", lambda: {"available_mb": 20000}), \
             mock.patch.object(asyncio, "create_subprocess_exec", fake_exec(ok_generate, self.captured)):
            response = self.client.post("/generate", json=self.generate_request())
        self.assertEqual(response.status_code, 200)

    def test_health_reports_memory_and_isolation(self):
        response = self.client.get("/health")
        body = response.json()
        status = body if response.status_code == 200 else body["detail"]
        self.assertIn("memory", status)
        self.assertTrue(status["stage_isolation"])

    def test_cancel_terminates_child(self):
        captured = []

        async def hanging_exec(*cmd, **kwargs):
            proc = FakeProcess(cmd, ok_generate)

            async def hang():
                await asyncio.Event().wait()

            proc.communicate = hang
            captured.append(proc)
            return proc

        async def run():
            with mock.patch.object(asyncio, "create_subprocess_exec", hanging_exec):
                task = asyncio.ensure_future(hunyuan_worker._isolated_stage("generate", {}))
                await asyncio.sleep(0.05)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(run())
        self.assertTrue(captured[0].terminated)


class StageRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.model_path = cls.root / "models"
        (cls.model_path / "hunyuan3d-dit-v2-1").mkdir(parents=True)
        (cls.model_path / "hunyuan3d-dit-v2-1" / "model.fp16.ckpt").write_text("model", encoding="utf-8")
        (cls.model_path / "hunyuan3d-paintpbr-v2-1").mkdir(parents=True)
        cls.t2i_path = cls.root / "t2i-model"
        cls.t2i_path.mkdir()
        cls.old_env = {}
        for k, v in {
            "HUNYUAN_MODEL_PATH": str(cls.model_path),
            "HUNYUAN_CPU_THREADS": "8",
            "HUNYUAN_DINO_DEVICE": "cpu",
            "HUNYFORGE_T2I_ENABLED": "1",
            "HUNYFORGE_T2I_MODEL_PATH": str(cls.t2i_path),
            "HUNYFORGE_DATA_ROOT": str(cls.root),
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
        self.slot = Path(tempfile.mkdtemp(prefix="runner-test-"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.slot, ignore_errors=True)

    def test_run_stage_generate_ok(self):
        job_id = uuid.uuid4()
        write_job(self.root, job_id)
        payload = WorkerRequest(job_id=job_id, stage="shape", seed=0, image=SAMPLE_IMAGE_URL, settings=make_settings()).model_dump(mode="json")
        engine = RuntimeEngine(self.root, dependencies=make_fake_dependencies())
        result = run_stage({"kind": "generate", "payload": payload}, engine, self.slot)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(Path(result["artifact"]).is_file())
        self.assertEqual(Path(result["artifact"]).name, "white-mesh.glb")

    def test_run_stage_preview_ok(self):
        payload = PreviewRequest(prompt="a cube", seed=3).model_dump(mode="json")
        engine = RuntimeEngine(self.root, dependencies=make_fake_dependencies())
        result = run_stage({"kind": "preview", "payload": payload}, engine, self.slot)
        self.assertEqual(result["status"], "ok")
        self.assertEqual((self.slot / "preview.png").read_bytes(), b"fakepng")
        self.assertIn("t2i_inference", result["timings"])

    def test_run_stage_unknown_kind(self):
        engine = RuntimeEngine(self.root, dependencies=make_fake_dependencies())
        with self.assertRaises(ValueError):
            run_stage({"kind": "bogus", "payload": {}}, engine, self.slot)

    def test_main_writes_result_on_error(self):
        request_path = self.slot / "request.json"
        result_path = self.slot / "result.json"
        request_path.write_text(json.dumps({"kind": "bogus", "payload": {}}), encoding="utf-8")
        rc = runner_main(["--request", str(request_path), "--result", str(result_path)])
        self.assertEqual(rc, 1)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "ValueError")

    def test_main_missing_request(self):
        rc = runner_main(["--request", str(self.slot / "missing.json"), "--result", str(self.slot / "result.json")])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
