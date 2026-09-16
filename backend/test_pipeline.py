import asyncio
import json
import os
import shlex
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from .models import GenerationSettings, JobCreate, JobStage, JobStatus, Project, load_generation_presets, resolve_generation_settings
from .pipeline import HunyuanOmniAdapter, HunyuanShapeAdapter, Pipeline, prepare_unity_geometry, scene_to_mesh, texture_quality_checks, validate_glb
from .storage import JobStore, sha256_file


def _textured_glb() -> bytes:
    import numpy as np
    import trimesh
    from PIL import Image

    mesh = trimesh.creation.box()
    image = Image.new("RGB", (8, 8), (200, 40, 40))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=image),
    )
    return mesh.export(file_type="glb")


def _settings(**overrides) -> GenerationSettings:
    merged = {**load_generation_presets()["presets"]["standard"]["settings"], **overrides}
    return GenerationSettings(**merged)


class PipelineTests(unittest.TestCase):
    def test_omni_job_requires_control_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "10 MB"):
            JobCreate(backend="demo", image="x" * 14_000_001)
        with self.assertRaises(ValueError):
            JobCreate(backend="hunyuan3d-2.1")
        with self.assertRaises(ValueError):
            JobCreate(backend="hunyuan3d-omni", image="data:image/png;base64,abc")
        with self.assertRaisesRegex(ValueError, "shape-only"):
            JobCreate(backend="hunyuan3d-omni", image="data:image/png;base64,img", control_type="bbox", control_data="data:application/json;base64,abc")
        request = JobCreate(backend="hunyuan3d-omni", image="data:image/png;base64,img", texture=False, control_type="bbox", control_data="data:application/json;base64,abc")
        self.assertEqual(request.control_type, "bbox")

    def test_project_metadata_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            project = Project(name="Lantern collection", description="Unity test assets")
            store.save_project(project)
            loaded = store.get_project(project.id)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.name, project.name)
            self.assertEqual(store.list_projects()[0].id, project.id)

    def test_demo_pipeline_creates_valid_unity_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True)
            store.save(job)
            asyncio.run(Pipeline(store).run(job))
            result = store.get(job.id)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.stage, JobStage.COMPLETE)
            self.assertEqual(len(result.artifacts), 8)
            glb = (store.path(job.id) / "artifacts" / "white-mesh.glb").read_bytes()
            self.assertIn("valid-glb-header", validate_glb(glb))
            self.assertTrue((store.path(job.id) / "artifacts" / "unity-package.zip").stat().st_size > 0)
            self.assertTrue((store.path(job.id) / "artifacts" / "unity-lod1.glb").exists())
            self.assertTrue((store.path(job.id) / "artifacts" / "unity-collision.glb").exists())
            manifest = json.loads((store.path(job.id) / "artifacts" / "hunyforge-manifest.json").read_text())
            self.assertEqual(manifest["memory_policy"], "sequential-shape-texture")

    def test_demo_pipeline_without_textures_keeps_white_mesh_workflow_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=False)
            store.save(job)
            asyncio.run(Pipeline(store).run(job))
            result = store.get(job.id)
            assert result is not None
            self.assertEqual(result.stage, JobStage.COMPLETE)
            self.assertIn("artifacts/white-mesh.glb", result.artifacts)
            self.assertNotIn("artifacts/textured-mesh.glb", result.artifacts)
            self.assertIn("artifacts/unity-package.zip", result.artifacts)

    def test_cancelled_job_does_not_generate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="demo", texture=True)
            store.save(job)
            pipeline = Pipeline(store)
            pipeline.request_cancel(str(job.id))
            asyncio.run(pipeline.run(job))
            result = store.get(job.id)
            assert result is not None
            self.assertEqual(result.stage, JobStage.CANCELLED)
            self.assertEqual(result.artifacts, [])

    def test_pipeline_reports_gpu_worker_busy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(JobStore(Path(directory)))
            pipeline.running.add("active-job")
            self.assertTrue(pipeline.is_busy())

    def test_hunyuan_adapter_sends_shape_and_texture_requests(self) -> None:
        class Response:
            def __init__(self):
                self._buffer = BytesIO(b"glTF-shape-or-texture")

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, size=-1):
                return self._buffer.read(size)

        requests = []

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return Response()

        adapter = HunyuanShapeAdapter("hunyuan3d-2.1")
        parameters = resolve_generation_settings("standard", {}).model_dump()
        parameters.update({
            "image": "data:image/png;base64,abc",
            "seed": 17,
            "job_id": "job-under-test",
            "control_type": "bbox",
            "control_data": "data:application/json;base64,xyz",
        })
        with tempfile.TemporaryDirectory() as directory:
            shape_out = Path(directory) / "shape.glb"
            texture_out = Path(directory) / "texture.glb"
            mesh = Path(directory) / "white-mesh.glb"
            mesh.write_bytes(b"mesh-bytes")
            with patch.dict(os.environ, {"HUNYUAN_STRIP_DATA_URL": "1"}, clear=False), patch("backend.pipeline.urllib.request.urlopen", side_effect=fake_urlopen):
                asyncio.run(adapter.generate_shape(17, parameters, shape_out))
                asyncio.run(adapter.generate_textures(mesh, parameters, texture_out))
            self.assertEqual(shape_out.read_bytes(), b"glTF-shape-or-texture")
            self.assertEqual(texture_out.read_bytes(), b"glTF-shape-or-texture")

        self.assertEqual(len(requests), 2)
        shape_payload = json.loads(requests[0][0].data)
        texture_payload = json.loads(requests[1][0].data)
        self.assertEqual(shape_payload["stage"], "shape")
        self.assertEqual(shape_payload["image"], "abc")
        self.assertEqual(shape_payload["job_id"], "job-under-test")
        self.assertEqual(texture_payload["stage"], "texture")
        self.assertEqual(texture_payload["mesh_token"]["artifact"], "white-mesh.glb")
        self.assertNotIn("mesh", set(shape_payload) | set(texture_payload) - {"mesh_token"})
        self.assertEqual(requests[0][1], 3600)

    def test_invalid_glb_fails_geometry_validation(self) -> None:
        self.assertIn("artifact-unreadable", validate_glb(b"not-a-glb"))

    def test_scene_flattening_supports_trimesh_without_to_geometry(self) -> None:
        import trimesh

        scene = trimesh.Scene(trimesh.creation.box())
        mesh = scene_to_mesh(scene)
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertGreater(mesh.faces.shape[0], 0)

    def test_non_finite_vertices_are_rejected(self) -> None:
        import numpy as np
        import trimesh
        fake_mesh = trimesh.Trimesh(vertices=np.array([[float('nan'), 0, 0]]), faces=np.array([[0, 0, 0]]), process=False)
        with patch("backend.pipeline.trimesh.load", return_value=fake_mesh):
            self.assertIn("non-finite-vertices", validate_glb(b"glTF" + b"0" * 16))

    def test_omni_runner_boundary_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "official inference.py CLI"):
                asyncio.run(HunyuanOmniAdapter().generate_shape(1, {"image": "abc"}, Path(directory) / "out.glb"))

    def test_omni_runner_executes_configured_local_command(self) -> None:
        command = f"{shlex.quote(sys.executable)} -c \"import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])\" {{image}} {{output}}"
        parameters = {"image": "data:image/png;base64,aW1hZ2U=", "control_type": "bbox", "control_data": "data:text/plain;base64,Y29udHJvbA=="}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.glb"
            with patch.dict(os.environ, {"HUNYUAN_OMNI_COMMAND": command, "HUNYUAN_STRIP_DATA_URL": "1"}, clear=False):
                asyncio.run(HunyuanOmniAdapter().generate_shape(22, parameters, output))
            self.assertEqual(output.read_bytes(), b"image")

    def test_hunyuan_http_error_preserves_service_detail(self) -> None:
        from urllib.error import HTTPError

        parameters = resolve_generation_settings("standard", {}).model_dump()
        parameters.update({"image": "abc", "seed": 1, "job_id": "job-under-test"})
        error = HTTPError("http://127.0.0.1:8082/generate", 404, "not found", {}, BytesIO(b'{"text":"model unavailable"}'))
        with tempfile.TemporaryDirectory() as directory:
            with patch("backend.pipeline.urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                    asyncio.run(HunyuanShapeAdapter("hunyuan3d-2.1").generate_shape(1, parameters, Path(directory) / "out.glb"))

    def test_textured_lod1_never_loses_pbr_silently(self) -> None:
        data = _textured_glb()
        produced, notes = prepare_unity_geometry(data, _settings(generate_collision=False))
        self.assertIn("unity-lod1.glb", produced)
        lod1 = produced["unity-lod1.glb"]
        if lod1 == data:
            self.assertIn("lod1-fallback-source-preserve-pbr-no-uv-decimator", notes)
        else:
            quality = texture_quality_checks(lod1)
            self.assertIn("uv-coordinates-present", quality)
            self.assertIn("material-present", quality)

    def test_lod1_receives_transferred_uvs_when_decimator_drops_them(self) -> None:
        import trimesh

        data = _textured_glb()
        with patch.object(trimesh.Trimesh, "simplify_quadric_decimation", return_value=trimesh.creation.icosphere()):
            produced, notes = prepare_unity_geometry(data, _settings(generate_collision=False))
        self.assertNotEqual(produced["unity-lod1.glb"], data)
        self.assertIn("lod1-uv-transferred-nearest-source", notes)
        quality = texture_quality_checks(produced["unity-lod1.glb"])
        self.assertIn("uv-coordinates-present", quality)
        self.assertIn("material-present", quality)

    def test_lod1_falls_back_to_source_bytes_when_uv_transfer_fails(self) -> None:
        import trimesh
        from unittest.mock import PropertyMock

        data = _textured_glb()
        with patch.object(trimesh.Trimesh, "simplify_quadric_decimation", return_value=trimesh.creation.icosphere()):
            with patch.object(trimesh.Trimesh, "kdtree", new_callable=PropertyMock, side_effect=RuntimeError("kdtree unavailable")):
                produced, notes = prepare_unity_geometry(data, _settings(generate_collision=False))
        self.assertEqual(produced["unity-lod1.glb"], data)
        self.assertIn("lod1-fallback-source-preserve-pbr-no-uv-decimator", notes)

    def test_validation_blocks_textured_lod_without_uv(self) -> None:
        import trimesh

        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="hunyuan3d-2.1", texture=True)
            store.save(job)
            artifacts = store.path(job.id) / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            from .pipeline import make_demo_glb

            textured = _textured_glb()
            geometry_only = trimesh.creation.box().export(file_type="glb")
            settings = _settings(generate_collision=False)
            files = {
                "white-mesh.glb": make_demo_glb(),
                "textured-mesh.glb": textured,
                "unity-lod0.glb": textured,
                "unity-lod1.glb": geometry_only,
            }
            for name, content in files.items():
                (artifacts / name).write_bytes(content)
            manifest = {
                "job_id": str(job.id),
                "lods": ["unity-lod0.glb", "unity-lod1.glb"],
                "collision": None,
                "artifacts": {name: sha256_file(artifacts / name) for name in ("unity-lod0.glb", "unity-lod1.glb")},
            }
            (artifacts / "hunyforge-manifest.json").write_text(json.dumps(manifest))
            report, ok = Pipeline(store)._build_validation_report(job, settings)
            self.assertFalse(ok)
            blocking = {entry["check"] for entry in report["blocking_failures"]}
            self.assertIn("export-unity-lod1.glb-uv-material", blocking)

    def test_validation_requires_recorded_hash_for_each_expected_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = JobStatus(backend="hunyuan3d-2.1", texture=True)
            store.save(job)
            artifacts = store.path(job.id) / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            from .pipeline import make_demo_glb

            textured = _textured_glb()
            settings = _settings(generate_collision=False)
            files = {
                "white-mesh.glb": make_demo_glb(),
                "textured-mesh.glb": textured,
                "unity-lod0.glb": textured,
                "unity-lod1.glb": textured,
            }
            for name, content in files.items():
                (artifacts / name).write_bytes(content)
            manifest = {
                "job_id": str(job.id),
                "lods": ["unity-lod0.glb", "unity-lod1.glb"],
                "collision": None,
                "artifacts": {"unity-lod0.glb": sha256_file(artifacts / "unity-lod0.glb")},
            }
            (artifacts / "hunyforge-manifest.json").write_text(json.dumps(manifest))
            report, ok = Pipeline(store)._build_validation_report(job, settings)
            self.assertFalse(ok)
            blocking = {entry["check"] for entry in report["blocking_failures"]}
            self.assertIn("manifest-hash-missing-unity-lod1.glb", blocking)


if __name__ == "__main__":
    unittest.main()
