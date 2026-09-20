import json
import os
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from . import hunyuan_worker, main
from .models import AppSettings, JobStage, JobStatus, now
from .pipeline import Pipeline
from .retention import sweep_jobs
from .storage import JobStore, SettingsStore


def _job(stage: JobStage = JobStage.COMPLETE) -> JobStatus:
    return JobStatus(backend="demo", stage=stage, project_id=None, seed=7, texture=False, parameters={"seed": 7})


def _age_job(store: JobStore, job: JobStatus, days: int) -> None:
    """Backdate a persisted job; store.save() would stamp updated_at=now."""
    job.updated_at = now() - timedelta(days=days)
    job.finished_at = job.updated_at
    (store.path(job.id) / "job.json").write_text(job.model_dump_json(), encoding="utf-8")


class SettingsStoreTests(unittest.TestCase):
    def test_missing_file_returns_env_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsStore(Path(directory)).load()
        self.assertEqual(settings.job_max_age_days, 0)
        self.assertEqual(settings.data_max_gb, 0)
        self.assertEqual(settings.min_available_mb, 12288)
        self.assertTrue(settings.stage_isolation)

    def test_env_seeds_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"HUNYFORGE_MIN_AVAILABLE_MB": "4096", "HUNYFORGE_STAGE_ISOLATION": "0"}, clear=False):
                settings = SettingsStore(Path(directory)).load()
        self.assertEqual(settings.min_available_mb, 4096)
        self.assertFalse(settings.stage_isolation)

    def test_saved_file_wins_over_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory))
            store.save(AppSettings(min_available_mb=2048, stage_isolation=False, job_max_age_days=14))
            with patch.dict(os.environ, {"HUNYFORGE_MIN_AVAILABLE_MB": "9999", "HUNYFORGE_STAGE_ISOLATION": "1"}, clear=False):
                settings = store.load()
        self.assertEqual(settings.min_available_mb, 2048)
        self.assertFalse(settings.stage_isolation)
        self.assertEqual(settings.job_max_age_days, 14)

    def test_corrupt_file_quarantined_and_defaults_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("{not json", encoding="utf-8")
            settings = SettingsStore(Path(directory)).load()
            self.assertEqual(settings.min_available_mb, 12288)
            self.assertFalse(path.exists())
            self.assertTrue((Path(directory) / "settings.json.corrupt").is_file())


class SweepTests(unittest.TestCase):
    def test_completed_job_older_than_limit_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            old, recent = _job(), _job()
            store.save(old)
            store.save(recent)
            _age_job(store, old, days=40)
            report = sweep_jobs(store, AppSettings(job_max_age_days=30))
            self.assertEqual(report["deleted_jobs"], [str(old.id)])
            self.assertIsNone(store.get(old.id))
            self.assertIsNotNone(store.get(recent.id))

    def test_failed_limit_only_applies_to_failed_and_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            failed, complete = _job(JobStage.FAILED), _job(JobStage.COMPLETE)
            store.save(failed)
            store.save(complete)
            _age_job(store, failed, days=40)
            _age_job(store, complete, days=40)
            report = sweep_jobs(store, AppSettings(failed_job_max_age_days=30))
            self.assertEqual(report["deleted_jobs"], [str(failed.id)])
            self.assertIsNotNone(store.get(complete.id))

    def test_running_job_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            running = _job(JobStage.GENERATING_SHAPE)
            store.save(running)
            _age_job(store, running, days=365)
            report = sweep_jobs(store, AppSettings(job_max_age_days=1, failed_job_max_age_days=1, data_max_gb=0.0000001))
            self.assertEqual(report["deleted_jobs"], [])
            self.assertIsNotNone(store.get(running.id))

    def test_cap_evicts_oldest_terminal_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            oldest, middle, newest = _job(), _job(), _job()
            for job in (oldest, middle, newest):
                store.save(job)
            _age_job(store, oldest, days=3)
            _age_job(store, middle, days=2)
            report = sweep_jobs(store, AppSettings(data_max_gb=0.000001))
            self.assertIn(str(oldest.id), report["deleted_jobs"])
            self.assertIn(str(middle.id), report["deleted_jobs"])
            self.assertTrue(store.disk_usage()["total_bytes"] <= int(0.000001 * 1024**3) or store.get(newest.id) is not None)

    def test_zero_settings_is_a_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = _job()
            store.save(job)
            _age_job(store, job, days=3650)
            report = sweep_jobs(store, AppSettings())
            self.assertEqual(report["deleted_jobs"], [])
            self.assertIsNotNone(store.get(job.id))

    def test_stale_tmp_files_reaped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = _job()
            store.save(job)
            stale = store.path(job.id) / "artifact.abc123.tmp"
            fresh = store.path(job.id) / "artifact.def456.tmp"
            stale.write_bytes(b"partial")
            fresh.write_bytes(b"partial")
            old = time.time() - 7200
            os.utime(stale, (old, old))
            report = sweep_jobs(store, AppSettings())
            self.assertEqual(report["tmp_files_removed"], 1)
            self.assertFalse(stale.exists())
            self.assertTrue(fresh.exists())


class SettingsApiTests(unittest.TestCase):
    def _bind(self, directory: str) -> None:
        main.store = JobStore(Path(directory))
        main.settings_store = SettingsStore(Path(directory))
        main.pipeline = Pipeline(main.store)

    def test_get_settings_returns_defaults_and_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._bind(directory)
            with TestClient(main.app) as client:
                response = client.get("/api/settings")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["settings"]["job_max_age_days"], 0)
            self.assertIn("job_count", body["usage"])
            self.assertIn("total_bytes", body["usage"])

    def test_put_settings_persists_and_rejects_unknown_preset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._bind(directory)
            payload = {"job_max_age_days": 10, "failed_job_max_age_days": 3, "data_max_gb": 5, "default_preset": "final", "default_unreal_export": True, "min_available_mb": 8192, "stage_isolation": False}
            with TestClient(main.app) as client:
                response = client.put("/api/settings", json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["settings"]["default_preset"], "final")
                bad = client.put("/api/settings", json={**payload, "default_preset": "bogus"})
                self.assertEqual(bad.status_code, 422)
            persisted = SettingsStore(Path(directory)).load()
            self.assertEqual(persisted.job_max_age_days, 10)
            self.assertEqual(persisted.min_available_mb, 8192)
            self.assertFalse(persisted.stage_isolation)

    def test_sweep_endpoint_deletes_old_terminal_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._bind(directory)
            with TestClient(main.app) as client:
                # Seed the old job + retention limit after startup so the
                # startup sweep (which runs once at app start) doesn't win the race.
                old = _job()
                main.store.save(old)
                _age_job(main.store, old, days=40)
                main.settings_store.save(AppSettings(job_max_age_days=30))
                response = client.post("/api/settings/sweep")
            self.assertEqual(response.status_code, 200)
            self.assertIn(str(old.id), response.json()["deleted_jobs"])
            self.assertIsNone(main.store.get(old.id))

    def test_delete_job_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._bind(directory)
            with TestClient(main.app) as client:
                # Saved after startup so recover_incomplete_jobs leaves the
                # in-flight job non-terminal; pre-startup survivors get marked FAILED.
                failed = _job(JobStage.FAILED)
                running = _job(JobStage.GENERATING_SHAPE)
                main.store.save(failed)
                main.store.save(running)
                self.assertEqual(client.delete(f"/api/jobs/{running.id}").status_code, 409)
                self.assertEqual(client.delete(f"/api/jobs/{uuid4()}").status_code, 404)
                self.assertEqual(client.delete(f"/api/jobs/{failed.id}").status_code, 204)
                self.assertEqual(client.delete(f"/api/jobs/{failed.id}").status_code, 404)
            self.assertFalse(main.store.path(failed.id).exists())
            self.assertTrue(main.store.path(running.id).exists())


class WorkerSettingsTests(unittest.TestCase):
    def test_isolation_flag_reads_settings_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            SettingsStore(Path(directory)).save(AppSettings(stage_isolation=False))
            with patch.dict(os.environ, {"HUNYFORGE_DATA_ROOT": directory, "HUNYFORGE_STAGE_ISOLATION": "1"}, clear=False):
                self.assertFalse(hunyuan_worker.stage_isolation_enabled())

    def test_isolation_falls_back_to_env_without_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"HUNYFORGE_DATA_ROOT": directory, "HUNYFORGE_STAGE_ISOLATION": "0"}, clear=False):
                self.assertFalse(hunyuan_worker.stage_isolation_enabled())
            with patch.dict(os.environ, {"HUNYFORGE_DATA_ROOT": directory, "HUNYFORGE_STAGE_ISOLATION": "1"}, clear=False):
                self.assertTrue(hunyuan_worker.stage_isolation_enabled())

    def test_admission_threshold_reads_settings_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            SettingsStore(Path(directory)).save(AppSettings(min_available_mb=4096))
            with patch.dict(os.environ, {"HUNYFORGE_DATA_ROOT": directory, "HUNYFORGE_MIN_AVAILABLE_MB": "0"}, clear=False):
                with patch.object(hunyuan_worker, "memory_snapshot", return_value={"available_mb": 1024}):
                    self.assertIsNotNone(hunyuan_worker.admission_error())
                with patch.object(hunyuan_worker, "memory_snapshot", return_value={"available_mb": 8192}):
                    self.assertIsNone(hunyuan_worker.admission_error())


if __name__ == "__main__":
    unittest.main()
