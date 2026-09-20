"""Retention sweeper for job artifacts.

Deletes terminal jobs past their age limits, evicts oldest-first when the data
directory exceeds a size cap, and reaps crash leftovers (atomic-write *.tmp
files and orphaned stage-runner temp dirs). Running jobs are never touched:
only COMPLETE/FAILED/CANCELLED jobs are eligible for deletion.
"""

import logging
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import AppSettings, JobStage, JobStatus
from .storage import JobStore, directory_size

logger = logging.getLogger(__name__)

_FAILED_STAGES = {JobStage.FAILED, JobStage.CANCELLED}
_TMP_MAX_AGE_SECONDS = 3600
_SLOT_MAX_AGE_SECONDS = 6 * 3600


def _job_size(store: JobStore, job: JobStatus) -> int:
    return directory_size(store.path(job.id))


def _delete(store: JobStore, job: JobStatus, report: dict, reason: str) -> None:
    size = _job_size(store, job)
    if store.delete_job(job.id):
        report["deleted_jobs"].append(str(job.id))
        report["freed_bytes"] += size
        logger.info("retention: deleted job %s (%s, %d bytes)", job.id, reason, size)


def sweep_jobs(store: JobStore, settings: AppSettings, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    report: dict = {"deleted_jobs": [], "freed_bytes": 0, "tmp_files_removed": 0, "slot_dirs_removed": 0}

    terminal = [job for job in store.list_jobs() if job.stage in {JobStage.COMPLETE, JobStage.FAILED, JobStage.CANCELLED}]

    for job in terminal:
        limit = settings.failed_job_max_age_days if job.stage in _FAILED_STAGES else settings.job_max_age_days
        if limit <= 0:
            continue
        age = now - (job.finished_at or job.updated_at)
        if age > timedelta(days=limit):
            _delete(store, job, report, f"{job.stage.value} older than {limit}d")

    if settings.data_max_gb > 0:
        cap_bytes = int(settings.data_max_gb * 1024**3)
        remaining = [job for job in terminal if str(job.id) not in set(report["deleted_jobs"])]
        remaining.sort(key=lambda job: job.updated_at)
        for job in remaining:
            if store.disk_usage()["total_bytes"] <= cap_bytes:
                break
            _delete(store, job, report, "data cap eviction")

    cutoff = time.time()
    if store.jobs.is_dir():
        for tmp in store.jobs.rglob("*.tmp"):
            try:
                if tmp.is_file() and cutoff - tmp.stat().st_mtime > _TMP_MAX_AGE_SECONDS:
                    tmp.unlink()
                    report["tmp_files_removed"] += 1
            except OSError:
                continue

    for slot in Path(tempfile.gettempdir()).glob("hunyforge-*"):
        try:
            if slot.is_dir() and cutoff - slot.stat().st_mtime > _SLOT_MAX_AGE_SECONDS:
                shutil.rmtree(slot, ignore_errors=True)
                report["slot_dirs_removed"] += 1
        except OSError:
            continue

    return report
