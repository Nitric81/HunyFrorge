import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from . import main
from .models import JobStage, JobStatus, T2I_SCAFFOLD_EDIT, T2I_SCAFFOLD_GENERATE
from .pipeline import Pipeline
from .storage import JobStore

SAMPLE_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ORK5CYII="


def _demo_job(client, **overrides):
    body = {"backend": "demo", "texture": True, "seed": 7}
    body.update(overrides)
    response = client.post("/api/jobs", json=body)
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    for _ in range(500):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["stage"] == "complete":
            return job
        time.sleep(0.02)
    raise AssertionError(f"demo job did not complete: {job['stage']}")


class ApiContractTests(unittest.TestCase):
    def test_health_does_not_accept_nonexistent_model_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main.store = JobStore(Path(directory))
            with TestClient(main.app) as client:
                with patch.dict("os.environ", {"HUNYUAN_ROOT": str(Path(directory) / "missing")}, clear=False):
                    response = client.get("/health")
                self.assertEqual(response.status_code, 200)
                self.assertFalse(response.json()["model_paths_configured"])

    def test_project_job_and_artifact_http_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main.store = JobStore(Path(directory))
            main.pipeline = Pipeline(main.store)
            with TestClient(main.app) as client:
                project_response = client.post("/api/projects", json={"name": "API test project"})
                self.assertEqual(project_response.status_code, 201)
                project_id = project_response.json()["id"]
                self.assertEqual(client.get("/api/projects").status_code, 200)
                job_response = client.post("/api/jobs", json={"project_id": project_id, "backend": "demo", "texture": True, "seed": 7})
                self.assertEqual(job_response.status_code, 202)
                job_id = job_response.json()["id"]
                listed = client.get(f"/api/jobs?project_id={project_id}")
                self.assertEqual(listed.status_code, 200)
                self.assertEqual(listed.json()[0]["id"], job_id)
                job = {}
                for _ in range(500):
                    job = client.get(f"/api/jobs/{job_id}").json()
                    if job["stage"] == "complete":
                        break
                    time.sleep(0.02)
                self.assertEqual(job["stage"], "complete")
                artifacts = client.get(f"/api/jobs/{job_id}/artifacts").json()["artifacts"]
                self.assertIn("artifacts/unity-package.zip", artifacts)
                download = client.get(f"/api/jobs/{job_id}/artifacts/unity-package.zip")
                self.assertEqual(download.status_code, 200)
                self.assertGreater(len(download.content), 0)
                validation = client.get(f"/api/jobs/{job_id}/validation")
                self.assertEqual(validation.status_code, 200)
                self.assertEqual(validation.json()["status"], "passed")
                events = client.get(f"/api/jobs/{job_id}/events")
                self.assertEqual(events.status_code, 200)
                self.assertIn("event: job", events.text)
                retry_response = client.post(f"/api/jobs/{job_id}/retry")
                self.assertEqual(retry_response.status_code, 409)

    def test_failed_job_can_be_retried_as_new_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main.store = JobStore(Path(directory))
            main.pipeline = Pipeline(main.store)
            failed = JobStatus(backend="demo", stage=JobStage.FAILED, project_id=None, seed=99, texture=False, parameters={"seed": 99})
            main.store.save(failed)
            with TestClient(main.app) as client:
                response = client.post(f"/api/jobs/{failed.id}/retry")
                self.assertEqual(response.status_code, 202)
                retry_id = response.json()["id"]
                self.assertNotEqual(retry_id, str(failed.id))
                result = {}
                for _ in range(500):
                    result = client.get(f"/api/jobs/{retry_id}").json()
                    if result["stage"] == "complete":
                        break
                    time.sleep(0.02)
                self.assertEqual(result["stage"], "complete")

    def test_unreal_export_job_serves_unreal_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main.store = JobStore(Path(directory))
            main.pipeline = Pipeline(main.store)
            with TestClient(main.app) as client:
                job = _demo_job(client, unreal_export=True)
                self.assertTrue(job["unreal_export"])
                artifacts = client.get(f"/api/jobs/{job['id']}/artifacts").json()["artifacts"]
                self.assertIn("artifacts/unreal-package.zip", artifacts)
                self.assertIn("artifacts/hunyforge-unreal-manifest.json", artifacts)
                download = client.get(f"/api/jobs/{job['id']}/artifacts/unreal-package.zip")
                self.assertEqual(download.status_code, 200)
                self.assertGreater(len(download.content), 0)
                validation = client.get(f"/api/jobs/{job['id']}/validation")
                self.assertTrue(validation.json()["unreal_ready"])

    def test_unreal_export_propagates_to_retry_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main.store = JobStore(Path(directory))
            main.pipeline = Pipeline(main.store)
            failed = JobStatus(backend="demo", stage=JobStage.FAILED, project_id=None, seed=99, texture=True, unreal_export=True, parameters={"seed": 99})
            main.store.save(failed)
            with TestClient(main.app) as client:
                response = client.post(f"/api/jobs/{failed.id}/retry")
                self.assertEqual(response.status_code, 202)
                child = response.json()
                self.assertTrue(child["unreal_export"])
                result = {}
                for _ in range(500):
                    result = client.get(f"/api/jobs/{child['id']}").json()
                    if result["stage"] == "complete":
                        break
                    time.sleep(0.02)
                self.assertEqual(result["stage"], "complete")
                artifacts = client.get(f"/api/jobs/{child['id']}/artifacts").json()["artifacts"]
                self.assertIn("artifacts/unreal-package.zip", artifacts)

    def test_multi_view_job_requires_cardinal_views_and_redacts_source_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main.store = JobStore(Path(directory))
            main.pipeline = Pipeline(main.store)
            data_url = "data:image/png;base64," + SAMPLE_PNG_B64
            def reference(view):
                return {"view": view, "image": data_url, "filename": f"truck-{view}.png", "content_type": "image/png", "byte_size": 70}
            with TestClient(main.app) as client:
                incomplete = client.post("/api/jobs", json={"backend": "demo", "input_mode": "multi-image", "reference_images": [reference("front")]})
                self.assertEqual(incomplete.status_code, 422)
                response = client.post("/api/jobs", json={"backend": "demo", "input_mode": "multi-image", "reference_images": [reference(view) for view in ("front", "rear", "left", "right")]})
                self.assertEqual(response.status_code, 202, response.text)
                body = response.json()
                self.assertEqual(body["input_mode"], "multi-image")
                self.assertEqual({item["view"] for item in body["reference_images"]}, {"front", "rear", "left", "right"})
                self.assertNotIn("image", body["reference_images"][0])
                persisted = main.store.get(body["id"])
                self.assertEqual(len(persisted.reference_images), 4)
                self.assertTrue(persisted.image.startswith("data:image/png"))
                omni = client.post("/api/jobs", json={"backend": "hunyuan3d-omni", "texture": False, "input_mode": "multi-image", "reference_images": [reference(view) for view in ("front", "rear", "left", "right")], "control_type": "bbox", "control_data": "data:application/json;base64,e30="})
                self.assertEqual(omni.status_code, 422)


class TextFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        main.store = JobStore(Path(self.tmp.name))
        main.pipeline = Pipeline(main.store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reference_preview_demo_scaffolds_prompt(self):
        with TestClient(main.app) as client:
            response = client.post("/api/reference-preview", json={"prompt": "a castle", "seed": 5})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["image"].startswith("data:image/png;base64,"))
        self.assertIn(T2I_SCAFFOLD_GENERATE, body["prompt_effective"])
        self.assertEqual(body["seed"], 5)

    def test_reference_preview_scaffold_off(self):
        with TestClient(main.app) as client:
            response = client.post("/api/reference-preview", json={"prompt": "a castle", "scaffold": False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prompt_effective"], "a castle")

    def test_reference_preview_edit_uses_edit_scaffold(self):
        with TestClient(main.app) as client:
            response = client.post("/api/reference-preview", json={"prompt": "make it rusty", "image": SAMPLE_PNG_B64})
        self.assertEqual(response.status_code, 200)
        self.assertIn(T2I_SCAFFOLD_EDIT, response.json()["prompt_effective"])

    def test_reference_preview_parent_validation(self):
        with TestClient(main.app) as client:
            missing = client.post("/api/reference-preview", json={"prompt": "x", "image": SAMPLE_PNG_B64, "parent_job_id": str(uuid4())})
            self.assertEqual(missing.status_code, 404)
            pending = JobStatus(backend="demo", seed=1, parameters={"seed": 1})
            main.store.save(pending)
            no_checkpoint = client.post("/api/reference-preview", json={"prompt": "x", "image": SAMPLE_PNG_B64, "parent_job_id": str(pending.id)})
            self.assertEqual(no_checkpoint.status_code, 409)

    def test_reference_preview_validates_prompt(self):
        with TestClient(main.app) as client:
            self.assertEqual(client.post("/api/reference-preview", json={"prompt": ""}).status_code, 422)
            self.assertEqual(client.post("/api/reference-preview", json={"prompt": "x" * 2001}).status_code, 422)

    def test_job_create_text_mode_records_provenance(self):
        with TestClient(main.app) as client:
            response = client.post("/api/jobs", json={"backend": "demo", "input_mode": "text", "prompt": "a castle", "t2i_seed": 9})
        self.assertEqual(response.status_code, 202)
        job = response.json()
        self.assertEqual(job["input_mode"], "text")
        self.assertEqual(job["prompt"], "a castle")
        self.assertEqual(job["t2i_seed"], 9)
        self.assertEqual(job["t2i_model"], "black-forest-labs/FLUX.2-klein-4B")

    def test_job_create_text_mode_requires_prompt(self):
        with TestClient(main.app) as client:
            self.assertEqual(client.post("/api/jobs", json={"backend": "demo", "input_mode": "text"}).status_code, 422)

    def test_job_create_image_mode_rejects_prompt(self):
        with TestClient(main.app) as client:
            self.assertEqual(client.post("/api/jobs", json={"backend": "demo", "prompt": "a castle"}).status_code, 422)

    def test_retexture_rejected_via_jobs_endpoint(self):
        with TestClient(main.app) as client:
            self.assertEqual(client.post("/api/jobs", json={"backend": "demo", "input_mode": "retexture", "prompt": "rusty"}).status_code, 422)

    def test_retexture_requires_parent_and_checkpoint(self):
        with TestClient(main.app) as client:
            self.assertEqual(client.post(f"/api/jobs/{uuid4()}/retexture", json={"image": SAMPLE_PNG_B64}).status_code, 404)
            pending = JobStatus(backend="demo", seed=1, parameters={"seed": 1})
            main.store.save(pending)
            self.assertEqual(client.post(f"/api/jobs/{pending.id}/retexture", json={"image": SAMPLE_PNG_B64}).status_code, 409)

    def test_retexture_creates_lineage_child(self):
        with TestClient(main.app) as client:
            parent = _demo_job(client)
            response = client.post(f"/api/jobs/{parent['id']}/retexture", json={"image": SAMPLE_PNG_B64, "prompt": "make it rusty", "t2i_seed": 11})
            self.assertEqual(response.status_code, 202, response.text)
            child = response.json()
            self.assertEqual(child["parent_job_id"], parent["id"])
            self.assertEqual(child["resume_from"], "texture")
            self.assertEqual(child["input_mode"], "retexture")
            self.assertEqual(child["prompt"], "make it rusty")
            self.assertEqual(child["t2i_seed"], 11)
            self.assertTrue(child["texture"])
            for _ in range(500):
                child = client.get(f"/api/jobs/{child['id']}").json()
                if child["stage"] == "complete":
                    break
                time.sleep(0.02)
            self.assertEqual(child["stage"], "complete")
            artifacts = client.get(f"/api/jobs/{child['id']}/artifacts").json()["artifacts"]
            self.assertIn("artifacts/textured-mesh.glb", artifacts)
            self.assertIn("artifacts/white-mesh.glb", artifacts)
            parent_after = client.get(f"/api/jobs/{parent['id']}").json()
            self.assertEqual(parent_after["stage"], "complete")

    def test_retexture_rejects_bad_settings(self):
        with TestClient(main.app) as client:
            parent = _demo_job(client)
            bad_key = client.post(f"/api/jobs/{parent['id']}/retexture", json={"image": SAMPLE_PNG_B64, "parameters": {"bogus_key": 1}})
            self.assertEqual(bad_key.status_code, 422)
            bad_value = client.post(f"/api/jobs/{parent['id']}/retexture", json={"image": SAMPLE_PNG_B64, "parameters": {"texture_size": 4096}})
            self.assertEqual(bad_value.status_code, 422)


if __name__ == "__main__":
    unittest.main()
