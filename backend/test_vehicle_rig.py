import io
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import trimesh
from fastapi.testclient import TestClient

from . import main, vehicle_rig
from .models import VehicleRigSpec, VehicleWheelSpec
from .pipeline import Pipeline
from .storage import JobStore
from .telemetry import StageTelemetry


def _synthetic_vehicle(axle_positions=(0.55, -0.55)) -> bytes:
    """Box chassis + paired wheel cylinders (axles along X) + a floor sheet."""
    parts = [trimesh.creation.box(extents=[1.0, 0.4, 2.0])]
    parts[0].apply_translation([0, -0.05, 0])
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    for z in axle_positions:
        for x in (-0.45, 0.45):
            wheel = trimesh.creation.cylinder(radius=0.18, height=0.12, sections=32, transform=rot)
            wheel.apply_translation([x, -0.3, z])
            parts.append(wheel)
    floor = trimesh.creation.box(extents=[4.0, 0.002, 4.0])
    floor.apply_translation([0, -0.481, 0])
    parts.append(floor)
    return bytes(trimesh.util.concatenate(parts).export(file_type="glb"))


def _spec(wheels=None) -> VehicleRigSpec:
    wheels = wheels or [
        VehicleWheelSpec(name="Wheel_FL", center=(-0.45, -0.3, 0.55), radius=0.19, half_width=0.08, steer=True),
        VehicleWheelSpec(name="Wheel_FR", center=(0.45, -0.3, 0.55), radius=0.19, half_width=0.08, steer=True),
        VehicleWheelSpec(name="Wheel_RL", center=(-0.45, -0.3, -0.55), radius=0.19, half_width=0.08),
        VehicleWheelSpec(name="Wheel_RR", center=(0.45, -0.3, -0.55), radius=0.19, half_width=0.08),
    ]
    return VehicleRigSpec(wheels=wheels)


class GeometryTests(unittest.TestCase):
    def test_strip_ground_plane_removes_floor(self) -> None:
        mesh = vehicle_rig.load_single_mesh(_synthetic_vehicle())
        stripped, removed = vehicle_rig.strip_ground_plane(mesh)
        self.assertGreater(removed, 0)
        self.assertLess(len(stripped.faces), len(mesh.faces))

    def test_suggest_wheel_regions_finds_four_named_wheels(self) -> None:
        mesh = vehicle_rig.load_single_mesh(_synthetic_vehicle())
        stripped, _ = vehicle_rig.strip_ground_plane(mesh)
        wheels = vehicle_rig.suggest_wheel_regions(stripped)
        self.assertEqual({w.name for w in wheels}, {"Wheel_FL", "Wheel_FR", "Wheel_RL", "Wheel_RR"})
        by_name = {w.name: w for w in wheels}
        self.assertTrue(by_name["Wheel_FL"].steer)
        self.assertTrue(by_name["Wheel_FR"].steer)
        self.assertFalse(by_name["Wheel_RL"].steer)
        self.assertFalse(by_name["Wheel_RR"].steer)
        self.assertLess(by_name["Wheel_FL"].center[0], 0)
        self.assertGreater(by_name["Wheel_FR"].center[0], 0)
        self.assertGreater(by_name["Wheel_FL"].center[2], 0)
        self.assertLess(by_name["Wheel_RL"].center[2], 0)

    def test_partition_export_reload_preserves_nodes_and_pivots(self) -> None:
        spec = _spec()
        glb, report = vehicle_rig.build_rigged_glb(_synthetic_vehicle(), spec)
        self.assertGreater(report["ground_faces_removed"], 0)
        summary = vehicle_rig.rigged_node_summary(glb)
        nodes = {entry["node"]: entry["pivot"] for entry in summary["nodes"]}
        self.assertIn("Chassis", nodes)
        for wheel in spec.wheels:
            self.assertIn(wheel.name, nodes)
            self.assertTrue(np.allclose(nodes[wheel.name], wheel.center, atol=1e-3))
        checks, blocking = vehicle_rig.rig_quality_checks(glb, spec)
        self.assertIn("rig-chassis-present", checks)
        self.assertEqual(blocking, [])

    def test_partition_supports_six_wheels(self) -> None:
        wheels = [
            VehicleWheelSpec(name=f"Wheel_{side}_{axle}", center=(x, -0.3, z), radius=0.19, half_width=0.08, steer=axle == "Front")
            for axle, z in (("Front", 0.7), ("Middle", 0.0), ("Rear", -0.7))
            for side, x in (("L", -0.45), ("R", 0.45))
        ]
        spec = _spec(wheels)
        glb, report = vehicle_rig.build_rigged_glb(_synthetic_vehicle((0.7, 0.0, -0.7)), spec)
        self.assertEqual(len(report["wheels"]), 6)
        self.assertTrue(all(entry["faces"] > 0 for entry in report["wheels"]))
        checks, blocking = vehicle_rig.rig_quality_checks(glb, spec)
        self.assertEqual(blocking, [])
        self.assertEqual(sum(item.startswith("rig-wheel-") for item in checks), 6)

    def test_suggest_wheel_regions_finds_six_detached_wheels(self) -> None:
        mesh = vehicle_rig.load_single_mesh(_synthetic_vehicle((0.7, 0.0, -0.7)))
        stripped, _ = vehicle_rig.strip_ground_plane(mesh)
        wheels = vehicle_rig.suggest_wheel_regions(stripped)
        self.assertEqual(len(wheels), 6)
        self.assertEqual(len({wheel.name for wheel in wheels}), 6)
        self.assertEqual(sum(wheel.steer for wheel in wheels), 2)

    def test_wheel_geometry_recentred_on_axle(self) -> None:
        spec = _spec()
        glb, _ = vehicle_rig.build_rigged_glb(_synthetic_vehicle(), spec)
        scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
        for wheel in spec.wheels:
            transform, geom_name = scene.graph.get(wheel.name)
            part = scene.geometry[geom_name]
            centroid = part.vertices.mean(axis=0)
            self.assertTrue(np.allclose(centroid, np.zeros(3), atol=0.12), f"{wheel.name} centroid {centroid}")

    def test_empty_wheel_region_fails_honestly(self) -> None:
        wheels = [
            VehicleWheelSpec(name="Wheel_FL", center=(-0.45, -0.3, 0.55), radius=0.19, half_width=0.08),
            VehicleWheelSpec(name="Wheel_FR", center=(50.0, 50.0, 50.0), radius=0.19, half_width=0.08),
        ]
        with self.assertRaisesRegex(ValueError, "captured no geometry"):
            vehicle_rig.build_rigged_glb(_synthetic_vehicle(), _spec(wheels))

    def test_overlapping_wheels_flagged(self) -> None:
        wheels = [
            VehicleWheelSpec(name="Wheel_FL", center=(-0.45, -0.3, 0.55), radius=0.19, half_width=0.08),
            VehicleWheelSpec(name="Wheel_FR", center=(-0.40, -0.3, 0.55), radius=0.19, half_width=0.08),
        ]
        spec = _spec(wheels)
        glb, report = vehicle_rig.build_rigged_glb(_synthetic_vehicle(), spec)
        self.assertTrue(any("overlapping" in warning for warning in report.get("warnings", [])))
        _, blocking = vehicle_rig.rig_quality_checks(glb, spec)
        self.assertTrue(any("overlap" in item for item in blocking))


def _wait_complete(client, job_id: str, stage: str = "complete"):
    job = {}
    for _ in range(500):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["stage"] == stage or job["stage"] in ("failed", "cancelled"):
            return job
        time.sleep(0.02)
    return job


class ApiFlowTests(unittest.TestCase):
    def _vehicle_parent(self, client):
        response = client.post("/api/jobs", json={"backend": "demo", "texture": True, "seed": 7, "asset_type": "vehicle"})
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["id"]
        job = _wait_complete(client, job_id)
        self.assertEqual(job["stage"], "complete", job.get("error_message"))
        self.assertEqual(job["asset_type"], "vehicle")
        # Replace the single-triangle demo mesh with a synthetic vehicle and re-sign.
        record = main.store.get(job_id)
        assert record is not None
        artifacts = main.store.path(record.id) / "artifacts"
        payload = _synthetic_vehicle()
        for name in ("white-mesh.glb", "textured-mesh.glb"):
            (artifacts / name).write_bytes(payload)
        record.shape_checkpoint = main.store.make_checkpoint(record, ["white-mesh.glb"])
        record.texture_checkpoint = main.store.make_checkpoint(record, ["textured-mesh.glb"])
        main.store.save(record)
        return job_id

    def test_generic_job_marks_rig_stage_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main.store = JobStore(Path(directory))
            main.pipeline = Pipeline(main.store)
            with TestClient(main.app) as client:
                response = client.post("/api/jobs", json={"backend": "demo", "texture": True, "seed": 7})
                self.assertEqual(response.status_code, 202)
                job = _wait_complete(client, response.json()["id"])
                self.assertEqual(job["stage"], "complete")
                self.assertEqual(job["stage_status"]["rig"]["state"], "skipped")

    def test_wheel_suggest_and_rig_child_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main.store = JobStore(Path(directory))
            main.pipeline = Pipeline(main.store)
            with TestClient(main.app) as client:
                job_id = self._vehicle_parent(client)
                suggest = client.post(f"/api/jobs/{job_id}/wheel-suggest")
                self.assertEqual(suggest.status_code, 200, suggest.text)
                payload = suggest.json()
                self.assertEqual(len(payload["wheels"]), 4)
                self.assertGreater(payload["ground_faces_removed"], 0)
                rig = client.post(f"/api/jobs/{job_id}/rig", json={"rig_spec": {"wheels": payload["wheels"]}})
                self.assertEqual(rig.status_code, 202, rig.text)
                child = _wait_complete(client, rig.json()["id"])
                self.assertEqual(child["stage"], "complete", child.get("error_message"))
                self.assertEqual(child["input_mode"], "vehicle-rig")
                self.assertEqual(child["asset_type"], "vehicle")
                self.assertEqual(child["parent_job_id"], job_id)
                self.assertEqual(child["stage_status"]["rig"]["state"], "complete")
                self.assertIn("artifacts/vehicle-rigged.glb", child["artifacts"])
                self.assertIn("artifacts/vehicle-rig-report.json", child["artifacts"])
                manifest = client.get(f"/api/jobs/{child['id']}/artifacts/hunyforge-manifest.json")
                self.assertEqual(manifest.status_code, 200)
                self.assertIn("vehicle", manifest.json())
                self.assertEqual(len(manifest.json()["vehicle"]["wheels"]), 4)
                package = client.get(f"/api/jobs/{child['id']}/artifacts/unity-package.zip")
                self.assertEqual(package.status_code, 200)
                names = zipfile.ZipFile(io.BytesIO(package.content)).namelist()
                self.assertIn("Assets/HunyForge/vehicle-rigged.glb", names)
                self.assertIn("Assets/HunyForge/Editor/HunyForgeVehicleSetup.cs", names)
                validation = client.get(f"/api/jobs/{child['id']}/validation").json()
                self.assertEqual(validation["status"], "passed")
                self.assertTrue(any("rig-wheel" in check for check in validation["checks"]["export"]))
                parent = client.get(f"/api/jobs/{job_id}").json()
                self.assertNotIn("artifacts/vehicle-rigged.glb", parent["artifacts"])

    def test_rig_rejects_out_of_bounds_center(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main.store = JobStore(Path(directory))
            main.pipeline = Pipeline(main.store)
            with TestClient(main.app) as client:
                job_id = self._vehicle_parent(client)
                wheels = [VehicleWheelSpec(name="Wheel_FL", center=(99.0, 99.0, 99.0), radius=0.1, half_width=0.05).model_dump()]
                response = client.post(f"/api/jobs/{job_id}/rig", json={"rig_spec": {"wheels": wheels}})
                self.assertEqual(response.status_code, 422)

    def test_rig_rejects_job_without_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main.store = JobStore(Path(directory))
            main.pipeline = Pipeline(main.store)
            with TestClient(main.app) as client:
                missing = client.post(f"/api/jobs/{uuid4()}/rig", json={"rig_spec": {"wheels": [{"name": "W1", "center": [0, 0, 0], "radius": 0.1, "half_width": 0.05}]}})
                self.assertEqual(missing.status_code, 404)

    def test_resume_at_unity_copies_rig_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main.store = JobStore(Path(directory))
            main.pipeline = Pipeline(main.store)
            with TestClient(main.app) as client:
                job_id = self._vehicle_parent(client)
                source = main.store.get(job_id)
                spec = _spec()
                # Run the rig stage inline to give the source a rig checkpoint.
                import asyncio
                pipeline = main.pipeline
                job = source
                job.rig_spec = spec
                main.store.save(job)
                telemetry_path = main.store.path(job.id) / "telemetry" / "api.json"
                with StageTelemetry(telemetry_path) as telemetry:
                    asyncio.run(pipeline._rig_stage(job, telemetry))
                main.store.save(job)
                child = main._child_job(source, "unity")
                main.store.prepare_resume(source, child, "unity")
                target = main.store.path(child.id) / "artifacts" / "vehicle-rigged.glb"
                self.assertTrue(target.is_file())
                self.assertTrue(main.store.valid_checkpoint(child, "rig"))


if __name__ == "__main__":
    unittest.main()
