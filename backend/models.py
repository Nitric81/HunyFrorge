import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def now() -> datetime:
    return datetime.now(timezone.utc)


DEFAULT_SOURCE_REVISION = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
DEFAULT_MODEL_REVISION = "0b94677654c57bb9a6b6845cd7b704ccf551d327"
PIPELINE_STAGES = ("shape", "texture", "rig", "unity", "validation")
ReferenceView = Literal["front", "rear", "left", "right", "front_left", "rear_right", "top", "bottom"]
REQUIRED_REFERENCE_VIEWS = frozenset({"front", "rear", "left", "right"})
MAX_REFERENCE_SET_BYTES = 40 * 1024 * 1024
GENERATION_PRESETS_PATH = Path(__file__).resolve().parent / "generation-presets.json"


class JobStage(str, Enum):
    QUEUED = "queued"
    LOADING_SHAPE_MODEL = "loading_shape_model"
    GENERATING_SHAPE = "generating_shape"
    PROCESSING_MESH = "processing_mesh"
    RELEASING_SHAPE_MODEL = "releasing_shape_model"
    LOADING_TEXTURE_MODEL = "loading_texture_model"
    GENERATING_TEXTURES = "generating_textures"
    RIGGING = "rigging"
    PREPARING_UNITY_EXPORT = "preparing_unity_export"
    VALIDATING_UNITY_EXPORT = "validating_unity_export"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


TERMINAL_STAGES = {JobStage.COMPLETE, JobStage.FAILED, JobStage.CANCELLED}


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    num_inference_steps: int = Field(ge=1, le=100)
    guidance_scale: float = Field(ge=0.1, le=20)
    octree_resolution: Literal[256, 384, 512]
    num_chunks: int = Field(ge=1000, le=20000)
    remove_background: bool
    render_size: Literal[512, 768, 1024]
    texture_size: Literal[1024, 2048]
    max_num_view: Literal[4, 6]
    texture_inference_steps: int = Field(ge=1, le=50)
    texture_guidance_scale: float = Field(ge=0.1, le=20)
    multiview_resolution: Literal[512, 768]
    face_count: int = Field(ge=1000, le=100000)
    generate_lods: bool
    generate_collision: bool
    collision_mode: Literal["box", "convex_hull"]
    unity_mode: Literal["fast", "full"]

    @model_validator(mode="after")
    def validate_combinations(self):
        if self.texture_size < self.render_size:
            raise ValueError("texture_size must be greater than or equal to render_size")
        if self.unity_mode == "fast" and self.generate_collision and self.collision_mode == "convex_hull":
            raise ValueError("fast Unity mode does not support convex_hull collision; use collision_mode=box or unity_mode=full")
        return self


GENERATION_SETTINGS_FIELDS = frozenset(GenerationSettings.model_fields)
INTERNAL_PARAMETERS = frozenset({"image", "seed", "control_type", "control_data", "job_id"})


def load_generation_presets() -> dict:
    return json.loads(GENERATION_PRESETS_PATH.read_text(encoding="utf-8"))


def resolve_generation_settings(preset: str, parameters: dict | None, strict: bool = True) -> GenerationSettings:
    contract = load_generation_presets()
    presets = contract.get("presets", {})
    if preset not in presets:
        raise ValueError(f"Unknown generation preset: {preset}")
    overrides = dict(parameters or {})
    steps = overrides.pop("steps", None)
    if steps is not None:
        if "num_inference_steps" in overrides and overrides["num_inference_steps"] != steps:
            raise ValueError('Conflicting legacy "steps" alias and "num_inference_steps" values')
        overrides["num_inference_steps"] = steps
    for internal in INTERNAL_PARAMETERS:
        overrides.pop(internal, None)
    unknown = sorted(set(overrides) - GENERATION_SETTINGS_FIELDS)
    if strict and unknown:
        raise ValueError(f"Unknown generation settings: {', '.join(unknown)}")
    merged = dict(presets[preset].get("settings", {}))
    merged.update({key: value for key, value in overrides.items() if key in GENERATION_SETTINGS_FIELDS})
    return GenerationSettings(**merged)


def current_runtime_config() -> dict:
    return {
        "source_revision": os.getenv("HUNYUAN_SOURCE_REVISION", DEFAULT_SOURCE_REVISION),
        "model_revision": current_model_revision(),
        "dino_device": os.getenv("HUNYUAN_DINO_DEVICE", "cpu"),
        "flashvdm": os.getenv("HUNYUAN_FLASHVDM", "0"),
        "torch_compile": os.getenv("HUNYUAN_COMPILE", "0"),
        "cpu_threads": os.getenv("HUNYUAN_CPU_THREADS", "8"),
        "model_path": os.getenv("HUNYUAN_MODEL_PATH", ""),
        "request_timeout_seconds": int(os.getenv("HUNYUAN_REQUEST_TIMEOUT_SECONDS", "3600")),
        "pipeline_revision": "staged-v2-view-limit-20260913",
        "multi_view_url": os.getenv("HUNYFORGE_MULTI_VIEW_URL", ""),
    }


def current_model_revision() -> str:
    return os.getenv("HUNYUAN_MODEL_REVISION", DEFAULT_MODEL_REVISION)


T2I_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
T2I_SCAFFOLD_GENERATE = "single object, centered, entire object visible, plain neutral studio background, even lighting"
T2I_SCAFFOLD_EDIT = "keep the same object, shape, pose, and composition; change only the surface material, colors, and finish"


def t2i_config() -> dict:
    return {
        "enabled": os.getenv("HUNYFORGE_T2I_ENABLED", "0") == "1",
        "model": T2I_MODEL_ID,
        "model_path": os.getenv("HUNYFORGE_T2I_MODEL_PATH", ""),
        "max_prompt_chars": int(os.getenv("HUNYFORGE_T2I_MAX_PROMPT_CHARS", "2000")),
    }


def scaffold_prompt(prompt: str, edit: bool = False) -> str:
    suffix = T2I_SCAFFOLD_EDIT if edit else T2I_SCAFFOLD_GENERATE
    return f"{prompt.strip()}, {suffix}"


class StageState(BaseModel):
    state: Literal["waiting", "running", "complete", "failed", "skipped", "cancelled"] = "waiting"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_seconds: float = 0
    error: str | None = None


class Checkpoint(BaseModel):
    signature: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=now)
    source_job_id: UUID


class VehicleWheelSpec(BaseModel):
    """One wheel cut region: a cylinder centered on the axle."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1, max_length=40)
    center: tuple[float, float, float]
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    radius: float = Field(gt=0)
    half_width: float = Field(gt=0)
    steer: bool = False


class VehicleRigSpec(BaseModel):
    """Wheel cut regions in model space; wheels are partitioned out of the mesh."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    wheels: list[VehicleWheelSpec] = Field(min_length=1, max_length=8)
    strip_ground: bool = True

    @model_validator(mode="after")
    def validate_unique_names(self):
        names = [wheel.name for wheel in self.wheels]
        if len(set(names)) != len(names):
            raise ValueError("wheel names must be unique")
        return self


class ReferenceImage(BaseModel):
    """One locally persisted, canonically-labelled source view."""

    model_config = ConfigDict(extra="forbid")

    view: ReferenceView
    image: str = Field(min_length=1, max_length=14_000_000)
    filename: str = Field(min_length=1, max_length=255)
    content_type: Literal["image/png", "image/jpeg"]
    byte_size: int = Field(gt=0, le=10 * 1024 * 1024)
    sha256: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")


class JobStatus(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID | None = None
    stage: JobStage = JobStage.QUEUED
    progress: int = Field(default=0, ge=0, le=100)
    backend: str = "demo"
    seed: int = 48291
    texture: bool = True
    image: str | None = None
    reference_images: list[ReferenceImage] = Field(default_factory=list)
    control_type: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    peak_vram_mb: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    schema_version: int = 2
    preset: Literal["draft", "standard", "final"] = "standard"
    parent_job_id: UUID | None = None
    resume_from: Literal["shape", "texture", "rig", "unity", "validation"] | None = None
    input_mode: Literal["image", "multi-image", "text", "retexture", "vehicle-rig"] = "image"
    prompt: str | None = None
    t2i_seed: int | None = None
    t2i_model: str | None = None
    asset_type: Literal["generic", "character", "vehicle"] = "generic"
    rig_spec: VehicleRigSpec | None = None
    model_revision: str = ""
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    stage_status: dict[str, StageState] = Field(default_factory=lambda: {name: StageState() for name in PIPELINE_STAGES})
    shape_checkpoint: Checkpoint | None = None
    texture_checkpoint: Checkpoint | None = None
    rig_checkpoint: Checkpoint | None = None
    unity_checkpoint: Checkpoint | None = None
    validation_checkpoint: Checkpoint | None = None
    artifact_readiness: dict[str, bool] = Field(default_factory=dict)
    timings: dict[str, Any] = Field(default_factory=dict)
    current_operation: str | None = None
    operation_progress: dict[str, Any] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failed_stage: str | None = None
    available_actions: list[str] = Field(default_factory=list)
    resume_stage: str | None = None


class JobCreate(BaseModel):
    project_id: UUID | None = None
    backend: str = Field(default="demo", pattern="^(demo|hunyuan3d-2.1|hunyuan3d-omni)$")
    preset: Literal["draft", "standard", "final"] = "standard"
    seed: int = Field(default=48291, ge=0, le=2**32 - 1)
    texture: bool = True
    image: str | None = None
    reference_images: list[ReferenceImage] = Field(default_factory=list, max_length=8)
    control_type: str | None = Field(default=None, pattern="^(point|voxel|pose|bbox)$")
    control_data: str | None = None
    input_mode: Literal["image", "multi-image", "text", "retexture", "vehicle-rig"] = "image"
    prompt: str | None = Field(default=None, max_length=2000)
    t2i_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    asset_type: Literal["generic", "character", "vehicle"] = "generic"
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_backend_controls(self):
        if len({item.view for item in self.reference_images}) != len(self.reference_images):
            raise ValueError("Each reference view can be uploaded only once")
        total_reference_bytes = sum(item.byte_size for item in self.reference_images)
        if total_reference_bytes > MAX_REFERENCE_SET_BYTES:
            raise ValueError("Reference set exceeds the 40 MB total upload limit")
        if self.input_mode == "multi-image":
            present = {item.view for item in self.reference_images}
            missing = sorted(REQUIRED_REFERENCE_VIEWS - present)
            if missing:
                raise ValueError(f"Multi-view jobs require: {', '.join(missing)}")
            # Keep the old image field populated for lineage and legacy consumers. The
            # multi-view adapter receives every item in reference_images.
            front = next(item for item in self.reference_images if item.view == "front")
            self.image = front.image
        elif self.reference_images:
            raise ValueError("reference_images are only valid with input_mode=multi-image")
        # A base64 data URL is roughly 4/3 the source size; allow a small envelope for its header.
        if self.image and len(self.image) > 14_000_000:
            raise ValueError("Reference image exceeds the 10 MB upload limit")
        if self.control_data and len(self.control_data) > 14_000_000:
            raise ValueError("Omni control data exceeds the 10 MB upload limit")
        if self.backend != "demo" and not self.image:
            raise ValueError("Real Hunyuan jobs require a reference image")
        if self.backend == "hunyuan3d-omni" and (not self.control_type or not self.control_data):
            raise ValueError("Omni jobs require both control_type and control_data")
        if self.backend == "hunyuan3d-omni" and self.texture:
            raise ValueError("Omni jobs are shape-only; disable PBR textures before running Omni")
        if self.input_mode == "multi-image" and self.backend == "hunyuan3d-omni":
            raise ValueError("Multi-view jobs require the configured local multi-view adapter; Hunyuan3D-Omni controls are not multi-reference inputs")
        if self.input_mode == "retexture":
            raise ValueError("Retexture jobs must be created via POST /api/jobs/{job_id}/retexture")
        if self.input_mode == "vehicle-rig":
            raise ValueError("Vehicle rig jobs must be created via POST /api/jobs/{job_id}/rig")
        if self.input_mode == "text" and not (self.prompt and self.prompt.strip()):
            raise ValueError("Text-mode jobs require a prompt for provenance")
        if self.input_mode == "image" and self.prompt is not None:
            raise ValueError("prompt is only valid with input_mode=text")
        resolved = resolve_generation_settings(self.preset, self.parameters, strict=True)
        self.parameters = resolved.model_dump()
        return self


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "hunyforge-api"
    inference_mode: str
    gpu_available: bool
    model_paths_configured: bool
    model_snapshot_present: bool = False
    model_snapshot_path: str | None = None
    hunyuan_service_ready: bool = False
    runtime_ready: bool = False
    workers: dict[str, Any] = Field(default_factory=dict)
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    t2i: dict[str, Any] = Field(default_factory=dict)
    multi_view: dict[str, Any] = Field(default_factory=dict)


class Project(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
