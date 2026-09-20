import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

from .models import (
    AppSettings,
    Checkpoint,
    GenerationSettings,
    JobStage,
    JobStatus,
    PIPELINE_STAGES,
    Project,
    TERMINAL_STAGES,
    current_model_revision,
    current_runtime_config,
    now,
)


def _retry_transient(operation, attempts: int = 25, delay: float = 0.02):
    for attempt in range(attempts):
        try:
            return operation()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def _replace_atomic(source: Path, target: Path) -> None:
    _retry_transient(lambda: os.replace(source, target))


def sha256_file(path: Path) -> str:
    def digest_file():
        with open(path, "rb") as handle:
            if hasattr(hashlib, "file_digest"):
                return hashlib.file_digest(handle, "sha256").hexdigest()
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()

    return _retry_transient(digest_file)


def directory_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


class SettingsStore:
    """settings.json persistence. Missing or corrupt files fall back to
    env-seeded defaults; a corrupt file is quarantined, not overwritten."""

    def __init__(self, root: Path):
        self.path = Path(root) / "settings.json"

    def load(self) -> AppSettings:
        if not self.path.is_file():
            return AppSettings()
        try:
            return AppSettings.model_validate(json.loads(self.path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError):
            try:
                _replace_atomic(self.path, self.path.with_suffix(".json.corrupt"))
            except OSError:
                pass
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
        _replace_atomic(temporary, self.path)


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.jobs = root / "jobs"
        self.projects = root / "projects"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.projects.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def path(self, job_id: UUID) -> Path:
        return self.jobs / str(job_id)

    def save(self, job: JobStatus) -> None:
        directory = self.path(job.id)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            job.updated_at = now()
            temporary = directory / f"job.{uuid4().hex}.tmp"
            temporary.write_text(job.model_dump_json(indent=2), encoding="utf-8")
            _replace_atomic(temporary, directory / "job.json")

    def get(self, job_id: UUID) -> JobStatus | None:
        path = self.path(job_id) / "job.json"
        if not path.exists():
            return None
        job = JobStatus.model_validate(json.loads(_retry_transient(lambda: path.read_text(encoding="utf-8"))))
        return self._decorate_actions(job)

    def list_jobs(self, project_id: UUID | None = None) -> list[JobStatus]:
        jobs = []
        for path in self.jobs.glob("*/job.json"):
            job = JobStatus.model_validate(json.loads(_retry_transient(lambda: path.read_text(encoding="utf-8"))))
            if project_id is None or job.project_id == project_id:
                jobs.append(job)
        jobs.sort(key=lambda job: job.updated_at, reverse=True)
        return [self._decorate_actions(job) for job in jobs]

    def _decorate_actions(self, job: JobStatus) -> JobStatus:
        job.resume_stage = None
        if job.stage == JobStage.COMPLETE:
            job.available_actions = []
            return job
        if job.stage in (JobStage.CANCELLATION_REQUESTED, JobStage.CANCELLING):
            job.available_actions = []
            return job
        if job.stage not in TERMINAL_STAGES:
            job.available_actions = ["cancel"]
            return job
        actions = ["restart"]
        next_stage = self.earliest_incomplete_stage(job)
        job.resume_stage = next_stage
        if next_stage != "shape":
            actions.append("resume")
            if next_stage == "texture":
                actions.append("resume_texture")
            elif next_stage in ("unity", "validation"):
                actions.append("resume_unity")
        job.available_actions = actions
        return job

    def delete_job(self, job_id: UUID) -> bool:
        path = self.path(job_id)
        if not path.is_dir():
            return False
        shutil.rmtree(path)
        return True

    def disk_usage(self) -> dict:
        job_dirs = [path for path in self.jobs.iterdir() if path.is_dir()] if self.jobs.is_dir() else []
        jobs_bytes = sum(directory_size(path) for path in job_dirs)
        projects_bytes = directory_size(self.projects) if self.projects.is_dir() else 0
        return {"total_bytes": jobs_bytes + projects_bytes, "jobs_bytes": jobs_bytes, "projects_bytes": projects_bytes, "job_count": len(job_dirs)}

    def recover_incomplete_jobs(self) -> int:
        """Mark jobs interrupted by an API/container restart as failed."""
        recovered = 0
        terminal = {JobStage.COMPLETE, JobStage.FAILED, JobStage.CANCELLED}
        for path in self.jobs.glob("*/job.json"):
            job = JobStatus.model_validate_json(_retry_transient(lambda: path.read_text(encoding="utf-8")))
            if job.stage in terminal:
                continue
            finished = now()
            for record in job.stage_status.values():
                if record.state == "running":
                    record.state = "failed"
                    record.finished_at = finished
                    if record.started_at is not None:
                        record.elapsed_seconds = (finished - record.started_at).total_seconds()
            job.error_code = "SERVICE_RESTARTED"
            job.error_message = "Generation was interrupted because HunyForge restarted. Retry this job to resume."
            job.stage = JobStage.FAILED
            job.finished_at = finished
            job.current_operation = None
            self.save(job)
            recovered += 1
        return recovered

    @staticmethod
    def _safe_artifact_name(name: str) -> str:
        if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(f"Invalid artifact name: {name!r}")
        return name

    def _record_artifact(self, job: JobStatus, name: str) -> str:
        relative = f"artifacts/{name}"
        if relative not in job.artifacts:
            job.artifacts.append(relative)
        job.artifact_readiness[name] = True
        self.save(job)
        return relative

    def add_artifact(self, job: JobStatus, name: str, content: bytes) -> str:
        safe = self._safe_artifact_name(name)
        directory = self.path(job.id) / "artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / safe
        with self._lock:
            if target.exists():
                if sha256_file(target) != hashlib.sha256(content).hexdigest():
                    raise ValueError(f"Artifact {safe} is finalized and cannot be overwritten with different content")
            else:
                temporary = directory / f"{safe}.{uuid4().hex}.tmp"
                temporary.write_bytes(content)
                _replace_atomic(temporary, target)
        return self._record_artifact(job, safe)

    def commit_artifact(self, job: JobStatus, name: str, staging: Path) -> str:
        safe = self._safe_artifact_name(name)
        directory = self.path(job.id) / "artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / safe
        incoming = sha256_file(staging)
        with self._lock:
            if target.exists():
                if sha256_file(target) != incoming:
                    raise ValueError(f"Artifact {safe} is finalized and cannot be overwritten with different content")
                staging.unlink()
            else:
                _replace_atomic(staging, target)
        return self._record_artifact(job, safe)

    def checkpoint_signature(self, job: JobStatus) -> str:
        settings = {key: job.parameters.get(key) for key in GenerationSettings.model_fields}
        payload = {
            "schema_version": job.schema_version,
            "backend": job.backend,
            "seed": job.seed,
            "preset": job.preset,
            "texture": job.texture,
            "control_type": job.control_type,
            "control_data_sha256": hashlib.sha256(str(job.parameters.get("control_data") or "").encode("utf-8")).hexdigest(),
            "settings": settings,
            "image_sha256": hashlib.sha256((job.image or "").encode("utf-8")).hexdigest(),
            "model_revision": job.model_revision,
            "runtime_config": job.runtime_config,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def make_checkpoint(self, job: JobStatus, artifact_names: list[str]) -> Checkpoint:
        artifacts = {}
        for name in artifact_names:
            safe = self._safe_artifact_name(name)
            artifacts[f"artifacts/{safe}"] = sha256_file(self.path(job.id) / "artifacts" / safe)
        return Checkpoint(signature=self.checkpoint_signature(job), artifacts=artifacts, source_job_id=job.id)

    def _required_checkpoint_artifacts(self, job: JobStatus, stage: str) -> set[str] | None:
        if stage == "shape":
            return {"artifacts/white-mesh.glb"}
        if stage == "texture":
            return {"artifacts/textured-mesh.glb"} if job.texture else None
        if stage == "rig":
            return {"artifacts/vehicle-rigged.glb"} if job.rig_spec else None
        if stage == "unity":
            required = {"artifacts/unity-lod0.glb", "artifacts/hunyforge-manifest.json"}
            if job.parameters.get("generate_lods"):
                required.add("artifacts/unity-lod1.glb")
            if job.parameters.get("generate_collision"):
                required.add("artifacts/unity-collision.glb")
            return required
        if stage == "validation":
            return {"artifacts/validation-report.json"}
        return None

    def valid_checkpoint(self, job: JobStatus, stage: str) -> bool:
        checkpoint = getattr(job, f"{stage}_checkpoint", None)
        if checkpoint is None:
            return False
        if job.model_revision != current_model_revision():
            return False
        if job.runtime_config != current_runtime_config():
            return False
        if checkpoint.source_job_id != job.id:
            return False
        if not checkpoint.artifacts:
            return False
        required = self._required_checkpoint_artifacts(job, stage)
        if required is None or not required.issubset(set(checkpoint.artifacts)):
            return False
        if checkpoint.signature != self.checkpoint_signature(job):
            return False
        job_root = Path(os.path.realpath(self.path(job.id)))
        artifacts_root = Path(os.path.realpath(self.path(job.id) / "artifacts"))
        for relative, expected in checkpoint.artifacts.items():
            if not relative.startswith("artifacts/") or ".." in Path(relative).parts:
                return False
            target = Path(os.path.realpath(self.path(job.id) / relative))
            if target != artifacts_root and artifacts_root not in target.parents:
                return False
            if job_root not in target.parents and target != job_root:
                return False
            if not target.is_file():
                return False
            if sha256_file(target) != expected:
                return False
        return True

    def earliest_incomplete_stage(self, job: JobStatus) -> str:
        if not self.valid_checkpoint(job, "shape"):
            return "shape"
        if job.texture and not self.valid_checkpoint(job, "texture"):
            return "texture"
        if job.rig_spec and not self.valid_checkpoint(job, "rig"):
            return "rig"
        if not self.valid_checkpoint(job, "unity"):
            return "unity"
        package = self.path(job.id) / "artifacts" / "unity-package.zip"
        if not self.valid_checkpoint(job, "validation") or not package.is_file():
            return "validation"
        return "validation"

    def prepare_resume(self, source: JobStatus, child: JobStatus, resume_stage: str) -> None:
        start = PIPELINE_STAGES.index(resume_stage)
        for stage in PIPELINE_STAGES[:start]:
            if stage == "texture" and not source.texture:
                continue
            if stage == "rig" and not source.rig_spec:
                continue
            checkpoint = getattr(source, f"{stage}_checkpoint", None)
            if checkpoint is None or not self.valid_checkpoint(source, stage):
                raise ValueError(f"Cannot resume at {resume_stage}: required {stage} checkpoint is missing or invalid")
            for relative in checkpoint.artifacts:
                origin = self.path(source.id) / relative
                target = self.path(child.id) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    _retry_transient(lambda: shutil.copyfile(origin, target))
                if relative not in child.artifacts:
                    child.artifacts.append(relative)
                child.artifact_readiness[Path(relative).name] = True
            setattr(child, f"{stage}_checkpoint", Checkpoint(signature=self.checkpoint_signature(child), artifacts=dict(checkpoint.artifacts), source_job_id=child.id))
        child.resume_from = resume_stage
        child.parent_job_id = source.id

    def save_project(self, project: Project) -> None:
        directory = self.projects / str(project.id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "project.json").write_text(project.model_dump_json(indent=2), encoding="utf-8")

    def get_project(self, project_id) -> Project | None:
        path = self.projects / str(project_id) / "project.json"
        return Project.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    def list_projects(self) -> list[Project]:
        return [project for path in self.projects.glob("*/project.json") if (project := Project.model_validate_json(path.read_text(encoding="utf-8")))]
