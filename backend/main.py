import asyncio
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal
from uuid import UUID
import importlib.util

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    AppSettings,
    GenerationSettings,
    HealthResponse,
    JobCreate,
    JobStage,
    JobStatus,
    Project,
    ProjectCreate,
    StageState,
    VehicleRigSpec,
    TERMINAL_STAGES,
    current_model_revision,
    current_runtime_config,
    load_generation_presets,
    now,
    qwen_edit_config,
    scaffold_prompt,
    t2i_config,
    T2I_MODEL_ID,
)
import json
from .pipeline import Pipeline
from .retention import sweep_jobs
from .storage import JobStore, SettingsStore
from . import vehicle_rig

logger = logging.getLogger(__name__)

ROOT = Path(os.getenv("HUNYFORGE_DATA_ROOT", Path.cwd() / "data"))
store = JobStore(ROOT)
settings_store = SettingsStore(ROOT)
pipeline = Pipeline(store)
app = FastAPI(title="HunyForge API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174", "http://localhost:4173", "http://127.0.0.1:4173"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def recover_interrupted_jobs() -> None:
    store.recover_incomplete_jobs()


@app.on_event("startup")
async def start_retention_sweeper() -> None:
    async def _loop() -> None:
        while True:
            try:
                report = await asyncio.to_thread(sweep_jobs, store, settings_store.load())
                if report["deleted_jobs"] or report["tmp_files_removed"] or report["slot_dirs_removed"]:
                    logger.info("retention sweep: %s", report)
            except Exception:
                logger.exception("retention sweep failed")
            await asyncio.sleep(3600)

    app.state.retention_sweeper = asyncio.create_task(_loop())

DIST_ROOT = Path(__file__).resolve().parent.parent / "dist"
if DIST_ROOT.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST_ROOT / "assets"), name="assets")

_worker_health_cache: dict = {"expires": 0.0, "data": None}

PUBLIC_JOB_EXCLUDE = {
    "image": True,
    "reference_images": {"__all__": {"image": True}},
    "parameters": {"image": True, "reference_images": True, "control_data": True},
}


def hunyuan_service_ready() -> bool:
    if os.getenv("HUNYFORGE_DEMO", "1") == "1":
        return True
    try:
        with urllib.request.urlopen(
            os.getenv("HUNYUAN_API_URL", "http://127.0.0.1:8082") + "/health",
            timeout=1,
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def worker_health() -> dict:
    if os.getenv("HUNYFORGE_DEMO", "1") == "1":
        return {"demo": {"ready": True, "state": "embedded"}}
    cached = _worker_health_cache
    if time.monotonic() < cached["expires"] and cached["data"] is not None:
        return cached["data"]
    try:
        with urllib.request.urlopen(
            os.getenv("HUNYUAN_API_URL", "http://127.0.0.1:8082") + "/health",
            timeout=1,
        ) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
            data = {"hunyuan": body if isinstance(body, dict) else {"ready": response.status == 200}}
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        data = {"hunyuan": {"ready": False}}
    cached["data"] = data
    cached["expires"] = time.monotonic() + 1.0
    return data


def multi_view_service_ready() -> bool:
    url = os.getenv("HUNYFORGE_MULTI_VIEW_URL", "").strip()
    if not url:
        return False
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _build_health() -> HealthResponse:
    torch_spec = importlib.util.find_spec("torch")
    gpu = False
    if torch_spec:
        try:
            import torch
            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
    hunyuan_root = os.getenv("HUNYUAN_ROOT")
    model_configured = bool(hunyuan_root and (Path(hunyuan_root) / "api_server.py").is_file())
    snapshot_root = os.getenv("HUNYUAN_MODEL_PATH")
    snapshot_present = bool(snapshot_root and (Path(snapshot_root) / "hunyuan3d-dit-v2-1" / "model.fp16.ckpt").is_file())
    mode = "demo" if os.getenv("HUNYFORGE_DEMO", "1") == "1" else "hunyuan"
    service_ready = hunyuan_service_ready()
    multi_view_enabled = multi_view_service_ready()
    return HealthResponse(inference_mode=mode, gpu_available=gpu, model_paths_configured=model_configured, model_snapshot_present=snapshot_present, model_snapshot_path=snapshot_root if snapshot_present else None, hunyuan_service_ready=service_ready, runtime_ready=mode == "demo" or (gpu and model_configured and snapshot_present and service_ready), workers=worker_health(), runtime_config=current_runtime_config(), t2i=t2i_config(), qwen_edit=qwen_edit_config(), multi_view={"enabled": multi_view_enabled or mode == "demo", "adapter_configured": bool(os.getenv("HUNYFORGE_MULTI_VIEW_URL", "").strip()), "reason": None if multi_view_enabled or mode == "demo" else "Start the local Hunyuan3D-2mv worker and set HUNYFORGE_MULTI_VIEW_URL."})


@app.get("/", include_in_schema=False)
async def frontend_index():
    if (DIST_ROOT / "index.html").is_file():
        return FileResponse(DIST_ROOT / "index.html")
    raise HTTPException(status_code=404, detail="Frontend build is not present")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return await asyncio.to_thread(_build_health)


@app.get("/api/presets")
async def presets() -> dict:
    contract = await asyncio.to_thread(load_generation_presets)
    return {**contract, "settings_schema": GenerationSettings.model_json_schema()}


class ReferencePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=2000)
    seed: int = Field(default=48291, ge=0, le=2**32 - 1)
    image: str | None = Field(default=None, max_length=14_000_000)
    scaffold: bool = True
    size: Literal[512, 768, 1024] = 1024
    parent_job_id: UUID | None = None


class SpriteCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str | None = Field(default=None, max_length=2000)
    image: str | None = Field(default=None, max_length=14_000_000)
    seed: int = Field(default=48291, ge=0, le=2**32 - 1)
    size: Literal[256, 512, 1024, 2048] = 1024
    width: int | None = Field(default=None, ge=16, le=4096)
    height: int | None = Field(default=None, ge=16, le=4096)
    padding_percent: int = Field(default=8, ge=0, le=40)
    scaffold: bool = True

    @model_validator(mode="after")
    def require_source(self):
        if bool(self.prompt and self.prompt.strip()) == bool(self.image):
            raise ValueError("Provide exactly one of prompt or image")
        return self


class SpriteJobCreateRequest(SpriteCreateRequest):
    project_id: UUID | None = None


class QwenEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str = Field(min_length=1, max_length=14_000_000)
    prompt: str = Field(min_length=1, max_length=2000)
    seed: int = Field(default=48291, ge=0, le=2**32 - 1)
    size: Literal[512, 768, 1024] = 1024
    num_inference_steps: int = Field(default=40, ge=10, le=80)


class SpriteSheetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: list[str] = Field(min_length=1, max_length=64)
    frame_width: int = Field(ge=16, le=4096)
    frame_height: int = Field(ge=16, le=4096)
    columns: int = Field(ge=1, le=16)
    directions: int = Field(default=1, ge=1, le=8)
    animation: str = Field(default="idle", min_length=1, max_length=64)


VEHICLE_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


class VehicleStateFrames(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["intact", "damaged", "wrecked", "empty", "loaded"]
    frames: list[str] = Field(min_length=8, max_length=8)


class VehicleSpriteSetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    states: list[VehicleStateFrames] = Field(min_length=3, max_length=5)
    frame_width: int = Field(ge=16, le=4096)
    frame_height: int = Field(ge=16, le=4096)

    @model_validator(mode="after")
    def require_core_states(self):
        names = [entry.state for entry in self.states]
        if len(set(names)) != len(names):
            raise ValueError("Each vehicle state may appear only once")
        required = {"intact", "damaged", "wrecked"}
        missing = required.difference(names)
        if missing:
            raise ValueError(f"Missing required vehicle states: {', '.join(sorted(missing))}")
        return self


class RetextureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str = Field(min_length=1, max_length=14_000_000)
    prompt: str | None = Field(default=None, max_length=2000)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    t2i_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    parameters: dict[str, Any] = Field(default_factory=dict)


def _worker_url() -> str:
    return os.getenv("HUNYUAN_API_URL", "http://127.0.0.1:8082")


def _worker_preview(payload: dict) -> dict:
    request = urllib.request.Request(_worker_url() + "/preview", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("HUNYFORGE_T2I_TIMEOUT_SECONDS", "300"))) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as error:
        detail_body = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail_body).get("detail") or detail_body
        except json.JSONDecodeError:
            detail = detail_body
        raise HTTPException(status_code=error.code if 400 <= error.code < 600 else 502, detail=f"Worker preview failed: {detail}") from error
    except urllib.error.URLError as error:
        raise HTTPException(status_code=503, detail="Worker is unavailable or still loading; wait for runtime readiness and retry.") from error


def _worker_sprite(payload: dict) -> dict:
    request = urllib.request.Request(_worker_url() + "/sprite", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("HUNYFORGE_T2I_TIMEOUT_SECONDS", "300"))) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as error:
        detail_body = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail_body).get("detail") or detail_body
        except json.JSONDecodeError:
            detail = detail_body
        raise HTTPException(status_code=error.code if 400 <= error.code < 600 else 502, detail=f"Worker sprite processing failed: {detail}") from error
    except urllib.error.URLError as error:
        raise HTTPException(status_code=503, detail="Worker is unavailable or still loading; wait for runtime readiness and retry.") from error


def _worker_qwen_edit(payload: dict) -> dict:
    request = urllib.request.Request(_worker_url() + "/qwen-edit", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("HUNYFORGE_QWEN_EDIT_TIMEOUT_SECONDS", "1800"))) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as error:
        detail_body = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail_body).get("detail") or detail_body
        except json.JSONDecodeError:
            detail = detail_body
        raise HTTPException(status_code=error.code if 400 <= error.code < 600 else 502, detail=f"Qwen edit worker failed: {detail}") from error
    except urllib.error.URLError as error:
        raise HTTPException(status_code=503, detail="Qwen edit worker is unavailable or still loading") from error


def _demo_preview_image(request: ReferencePreviewRequest) -> str:
    try:
        import base64
        import random
        from io import BytesIO
        from PIL import Image, ImageDraw
    except ImportError:
        raise HTTPException(status_code=503, detail="Demo preview requires Pillow")
    rng = random.Random(request.seed)
    image = Image.new("RGB", (request.size, request.size), tuple(rng.randint(90, 200) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    draw.rectangle((request.size // 4, request.size // 4, request.size * 3 // 4, request.size * 3 // 4), fill=tuple(rng.randint(20, 120) for _ in range(3)))
    draw.text((20, 20), f"demo preview: {request.prompt[:60]}", fill=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _demo_sprite_image(request: SpriteCreateRequest) -> str:
    try:
        import base64
        import random
        from io import BytesIO
        from PIL import Image, ImageDraw
    except ImportError:
        raise HTTPException(status_code=503, detail="Demo sprite generation requires Pillow")
    rng = random.Random(request.seed)
    width, height = request.width or request.size, request.height or request.size
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    inset = round(min(width, height) * (0.16 + request.padding_percent / 250))
    draw.ellipse((inset, inset, width - inset, height - inset), fill=tuple(rng.randint(30, 200) for _ in range(3)) + (255,))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@app.post("/api/reference-preview")
async def reference_preview(request: ReferencePreviewRequest):
    if request.parent_job_id is not None:
        parent = store.get(request.parent_job_id)
        if not parent:
            raise HTTPException(status_code=404, detail="Parent job not found")
        if not store.valid_checkpoint(parent, "shape"):
            raise HTTPException(status_code=409, detail="Parent job has no valid white-mesh checkpoint to edit")
    prompt = scaffold_prompt(request.prompt, edit=bool(request.image)) if request.scaffold else request.prompt
    if os.getenv("HUNYFORGE_DEMO", "1") == "1":
        return {"image": _demo_preview_image(request), "timings": {"t2i_inference": 0.0}, "prompt_effective": prompt, "seed": request.seed}
    if not t2i_config()["enabled"]:
        raise HTTPException(status_code=503, detail="T2I reference generation is disabled on this runtime")
    payload = {"prompt": prompt, "seed": request.seed, "width": request.size, "height": request.size}
    if request.image:
        payload["image"] = request.image.split(",", 1)[1] if request.image.startswith("data:") and "," in request.image else request.image
    result = await asyncio.to_thread(_worker_preview, payload)
    return {"image": result["image"], "timings": result.get("timings", {}), "prompt_effective": prompt, "seed": request.seed}


@app.post("/api/sprites")
async def create_sprite(request: SpriteCreateRequest):
    width, height = request.width or request.size, request.height or request.size
    if os.getenv("HUNYFORGE_DEMO", "1") == "1":
        return {"image": _demo_sprite_image(request), "timings": {"sprite": 0.0}, "seed": request.seed, "size": request.size, "width": width, "height": height, "has_alpha": True, "prompt_effective": request.prompt}
    source = request.image
    prompt_effective = None
    if request.prompt:
        if not t2i_config()["enabled"]:
            raise HTTPException(status_code=503, detail="Text-to-sprite requires the local FLUX T2I worker")
        prompt_effective = scaffold_prompt(request.prompt) if request.scaffold else request.prompt
        preview = await asyncio.to_thread(_worker_preview, {"prompt": prompt_effective, "seed": request.seed, "width": 1024, "height": 1024})
        source = preview["image"]
    if not source:
        raise HTTPException(status_code=422, detail="Sprite source image is missing")
    result = await asyncio.to_thread(_worker_sprite, {"image": source, "width": width, "height": height, "padding_percent": request.padding_percent})
    return {"image": result["image"], "timings": result.get("timings", {}), "seed": request.seed, "size": request.size, "width": width, "height": height, "has_alpha": True, "prompt_effective": prompt_effective}


async def _run_sprite_job(job: JobStatus, request: SpriteCreateRequest) -> None:
    record = job.stage_status.setdefault("shape", StageState())
    started = now()
    record.state = "running"
    record.started_at = started
    job.stage = JobStage.GENERATING_SHAPE
    job.progress = 15
    job.current_operation = "sprite_source"
    store.save(job)
    try:
        result = await create_sprite(request)
        job.progress = 82
        job.current_operation = "sprite_packaging"
        store.save(job)
        image = result["image"]
        raw = image.split(",", 1)[1] if image.startswith("data:") and "," in image else image
        import base64
        store.add_artifact(job, "sprite.png", base64.b64decode(raw, validate=True))
        metadata = {key: value for key, value in result.items() if key != "image"}
        store.add_artifact(job, "sprite-metadata.json", json.dumps(metadata, indent=2).encode("utf-8"))
        record.state = "complete"
        record.finished_at = now()
        record.elapsed_seconds = (record.finished_at - started).total_seconds()
        for stage in ("texture", "rig", "unity", "validation"):
            skipped = job.stage_status.setdefault(stage, StageState())
            skipped.state = "skipped"
            skipped.finished_at = record.finished_at
        job.stage = JobStage.COMPLETE
        job.progress = 100
        job.current_operation = None
        job.finished_at = record.finished_at
        store.save(job)
    except Exception as error:
        record.state = "failed"
        record.finished_at = now()
        record.elapsed_seconds = (record.finished_at - started).total_seconds()
        record.error = str(error)
        job.stage = JobStage.FAILED
        job.error_code = "SPRITE_ERROR"
        job.error_message = str(error)
        job.current_operation = None
        job.finished_at = record.finished_at
        store.save(job)


@app.post("/api/sprite-jobs", response_model=JobStatus, status_code=202, response_model_exclude=PUBLIC_JOB_EXCLUDE)
async def create_sprite_job(request: SpriteJobCreateRequest) -> JobStatus:
    if request.project_id and not store.get_project(request.project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    payload = request.model_dump(mode="json", exclude={"project_id"})
    job = JobStatus(project_id=request.project_id, backend="sprite", seed=request.seed, texture=False, preset="draft", input_mode="sprite", prompt=request.prompt, t2i_seed=request.seed if request.prompt else None, t2i_model=T2I_MODEL_ID if request.prompt else None, parameters={"sprite": payload}, model_revision=current_model_revision(), runtime_config=current_runtime_config())
    store.save(job)
    asyncio.create_task(_run_sprite_job(job, request))
    return _fresh(job)


@app.post("/api/qwen-edit")
async def qwen_edit_image(request: QwenEditRequest):
    if not qwen_edit_config()["enabled"]:
        raise HTTPException(status_code=503, detail="Qwen image editing is disabled on this runtime")
    result = await asyncio.to_thread(_worker_qwen_edit, {"image": request.image, "prompt": request.prompt, "seed": request.seed, "width": request.size, "height": request.size, "num_inference_steps": request.num_inference_steps})
    return {"image": result["image"], "timings": result.get("timings", {}), "seed": request.seed, "model": qwen_edit_config()["model"]}


@app.post("/api/sprite-sheets")
async def create_sprite_sheet(request: SpriteSheetRequest):
    try:
        import base64
        from io import BytesIO
        from PIL import Image
    except ImportError:
        raise HTTPException(status_code=503, detail="Sprite sheet assembly requires Pillow")
    if request.directions > request.columns:
        raise HTTPException(status_code=422, detail="Columns must accommodate the requested directional frames")
    rows = (len(request.frames) + request.columns - 1) // request.columns
    atlas = Image.new("RGBA", (request.columns * request.frame_width, rows * request.frame_height), (0, 0, 0, 0))
    for index, source in enumerate(request.frames):
        if not source.startswith("data:image/") or "," not in source:
            raise HTTPException(status_code=422, detail=f"Frame {index + 1} must be a PNG or JPEG data URL")
        try:
            frame = Image.open(BytesIO(base64.b64decode(source.split(",", 1)[1], validate=True))).convert("RGBA")
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"Frame {index + 1} is not a readable image: {error}") from error
        frame.thumbnail((request.frame_width, request.frame_height))
        x = (index % request.columns) * request.frame_width + (request.frame_width - frame.width) // 2
        y = (index // request.columns) * request.frame_height + (request.frame_height - frame.height) // 2
        atlas.alpha_composite(frame, (x, y))
    output = BytesIO()
    atlas.save(output, format="PNG")
    encoded = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    return {"image": encoded, "has_alpha": True, "frame_count": len(request.frames), "columns": request.columns, "rows": rows, "frame_width": request.frame_width, "frame_height": request.frame_height, "unity": {"sprite_mode": "Multiple", "pixels_per_unit": request.frame_height, "directions": request.directions, "animation": request.animation}}


@app.post("/api/vehicle-sprite-sets")
async def create_vehicle_sprite_set(request: VehicleSpriteSetRequest):
    try:
        import base64
        from io import BytesIO
        from PIL import Image
    except ImportError:
        raise HTTPException(status_code=503, detail="Vehicle sprite packing requires Pillow")

    def decode_frame(source: str, state: str, direction: str):
        if not source.startswith("data:image/") or "," not in source:
            raise HTTPException(status_code=422, detail=f"{state}/{direction} must be a PNG or JPEG data URL")
        try:
            return Image.open(BytesIO(base64.b64decode(source.split(",", 1)[1], validate=True))).convert("RGBA")
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"{state}/{direction} is not a readable image: {error}") from error

    atlas = Image.new("RGBA", (8 * request.frame_width, len(request.states) * request.frame_height), (0, 0, 0, 0))
    row_images: dict[str, str] = {}
    mappings: list[dict] = []
    for row, entry in enumerate(request.states):
        row_image = Image.new("RGBA", (8 * request.frame_width, request.frame_height), (0, 0, 0, 0))
        for column, (direction, source) in enumerate(zip(VEHICLE_DIRECTIONS, entry.frames)):
            frame = decode_frame(source, entry.state, direction)
            frame.thumbnail((request.frame_width, request.frame_height))
            x = column * request.frame_width + (request.frame_width - frame.width) // 2
            y = (request.frame_height - frame.height) // 2
            row_image.alpha_composite(frame, (x, y))
            mappings.append({"state": entry.state, "direction": direction, "rect": {"x": column * request.frame_width, "y": row * request.frame_height, "width": request.frame_width, "height": request.frame_height}})
        atlas.alpha_composite(row_image, (0, row * request.frame_height))
        row_bytes = BytesIO(); row_image.save(row_bytes, format="PNG")
        row_images[entry.state] = "data:image/png;base64," + base64.b64encode(row_bytes.getvalue()).decode("ascii")
    output = BytesIO(); atlas.save(output, format="PNG")
    unity = {"sprite_mode": "Multiple", "pixels_per_unit": request.frame_height, "pivot": "Bottom", "directions": list(VEHICLE_DIRECTIONS), "states": [entry.state for entry in request.states], "sprites": mappings}
    return {"image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii"), "rows": row_images, "has_alpha": True, "columns": 8, "row_count": len(request.states), "frame_width": request.frame_width, "frame_height": request.frame_height, "unity": unity}


@app.post("/api/jobs", response_model=JobStatus, status_code=202, response_model_exclude=PUBLIC_JOB_EXCLUDE)
async def create_job(request: JobCreate) -> JobStatus:
    if request.backend != "demo" and not await asyncio.to_thread(hunyuan_service_ready):
        raise HTTPException(status_code=503, detail="Hunyuan worker is still loading. Wait for local runtime readiness and retry.")
    if request.project_id and not store.get_project(request.project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    if request.input_mode == "multi-image" and request.backend != "demo" and not await asyncio.to_thread(multi_view_service_ready):
        raise HTTPException(status_code=503, detail="Multi-view generation requires the local Hunyuan3D-2mv worker. Start it and set HUNYFORGE_MULTI_VIEW_URL; Hunyuan3D 2.1's stock image endpoint does not fuse multiple references.")
    parameters = {**request.parameters, "image": request.image, "reference_images": [item.model_dump() for item in request.reference_images], "seed": request.seed, "control_type": request.control_type, "control_data": request.control_data}
    job = JobStatus(project_id=request.project_id, backend=request.backend, seed=request.seed, texture=request.texture, image=request.image, reference_images=request.reference_images, control_type=request.control_type, parameters=parameters, preset=request.preset, input_mode=request.input_mode, prompt=request.prompt, t2i_seed=request.t2i_seed, t2i_model=T2I_MODEL_ID if request.t2i_seed is not None else None, asset_type=request.asset_type, unreal_export=request.unreal_export, model_revision=current_model_revision(), runtime_config=current_runtime_config())
    if not pipeline.reserve(job):
        raise HTTPException(status_code=409, detail="GPU worker is busy; wait for the active job to finish or cancel it")
    try:
        store.save(job)
    except Exception:
        pipeline.release(job)
        raise
    asyncio.create_task(pipeline.run(job))
    return _fresh(job)


@app.post("/api/jobs/{job_id}/retexture", response_model=JobStatus, status_code=202, response_model_exclude=PUBLIC_JOB_EXCLUDE)
async def retexture_job(job_id: UUID, request: RetextureRequest) -> JobStatus:
    source = store.get(job_id)
    if not source:
        raise HTTPException(status_code=404, detail="Job not found")
    if not store.valid_checkpoint(source, "shape"):
        raise HTTPException(status_code=409, detail="Job has no valid white-mesh checkpoint to retexture")
    if source.backend != "demo" and not await asyncio.to_thread(hunyuan_service_ready):
        raise HTTPException(status_code=503, detail="Hunyuan worker is still loading. Wait for local runtime readiness and retry.")
    overrides = dict(request.parameters)
    unknown = sorted(set(overrides) - set(GenerationSettings.model_fields))
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown generation settings: {', '.join(unknown)}")
    merged = {key: source.parameters.get(key) for key in GenerationSettings.model_fields}
    merged.update(overrides)
    try:
        settings = GenerationSettings(**merged)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid retexture settings: {e}") from e
    child_seed = request.seed if request.seed is not None else source.seed
    parameters = {**source.parameters, "image": request.image, "seed": child_seed, **settings.model_dump()}
    child = JobStatus(project_id=source.project_id, backend=source.backend, seed=child_seed, texture=True, image=request.image, reference_images=source.reference_images, control_type=source.control_type, parameters=parameters, preset=source.preset, parent_job_id=source.id, resume_from="texture", input_mode="retexture", prompt=request.prompt, t2i_seed=request.t2i_seed, t2i_model=T2I_MODEL_ID if request.t2i_seed is not None else None, unreal_export=source.unreal_export, model_revision=current_model_revision(), runtime_config=current_runtime_config())
    if not pipeline.reserve(child):
        raise HTTPException(status_code=409, detail="GPU worker is busy; wait for the active job to finish or cancel it")
    try:
        await asyncio.to_thread(store.prepare_resume, source, child, "texture")
        store.save(child)
    except Exception:
        pipeline.release(child)
        raise
    asyncio.create_task(pipeline.run(child))
    return _fresh(child)


def _mesh_artifact(job_id: UUID) -> tuple[Path, str] | tuple[None, None]:
    directory = store.path(job_id) / "artifacts"
    for name in ("textured-mesh.glb", "white-mesh.glb"):
        path = directory / name
        if path.is_file():
            return path, name
    return None, None


class RigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    rig_spec: VehicleRigSpec


def _suggest_wheels_payload(data: bytes, source_name: str) -> dict:
    mesh = vehicle_rig.load_single_mesh(data)
    stripped, removed = vehicle_rig.strip_ground_plane(mesh)
    wheels = vehicle_rig.suggest_wheel_regions(stripped)
    return {"wheels": [wheel.model_dump() for wheel in wheels], "ground_faces_removed": removed, "source_mesh": source_name}


@app.post("/api/jobs/{job_id}/wheel-suggest")
async def wheel_suggest(job_id: UUID):
    source = store.get(job_id)
    if not source:
        raise HTTPException(status_code=404, detail="Job not found")
    path, name = _mesh_artifact(job_id)
    if path is None:
        raise HTTPException(status_code=409, detail="Job has no mesh artifact to analyze")
    try:
        return await asyncio.to_thread(_suggest_wheels_payload, path.read_bytes(), name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/rig", response_model=JobStatus, status_code=202, response_model_exclude=PUBLIC_JOB_EXCLUDE)
async def rig_job(job_id: UUID, request: RigRequest) -> JobStatus:
    source = store.get(job_id)
    if not source:
        raise HTTPException(status_code=404, detail="Job not found")
    if source.backend != "demo" and not await asyncio.to_thread(hunyuan_service_ready):
        raise HTTPException(status_code=503, detail="Hunyuan worker is still loading. Wait for local runtime readiness and retry.")
    path, _name = _mesh_artifact(job_id)
    if path is None:
        raise HTTPException(status_code=409, detail="Job has no mesh artifact to rig — finish the shape stage first")
    bounds = await asyncio.to_thread(lambda: vehicle_rig.load_single_mesh(path.read_bytes()).bounds)
    if bounds is None:
        raise HTTPException(status_code=422, detail="Mesh artifact is unreadable")
    low, high = bounds
    extent = high - low
    pad = extent * 0.15
    for wheel in request.rig_spec.wheels:
        center = wheel.center
        for i in range(3):
            if not (low[i] - pad[i] <= center[i] <= high[i] + pad[i]):
                raise HTTPException(status_code=422, detail=f"Wheel {wheel.name} center is outside the mesh bounds")
        max_extent = float(extent.max())
        if wheel.radius > max_extent * 0.6 or wheel.half_width > wheel.radius * 1.5:
            raise HTTPException(status_code=422, detail=f"Wheel {wheel.name} cylinder is implausibly large for this mesh")
    parameters = dict(source.parameters)
    child = JobStatus(project_id=source.project_id, backend=source.backend, seed=source.seed, texture=source.texture, image=source.image, reference_images=source.reference_images, control_type=source.control_type, parameters=parameters, preset=source.preset, parent_job_id=source.id, resume_from="rig", input_mode="vehicle-rig", asset_type="vehicle", rig_spec=request.rig_spec, unreal_export=source.unreal_export, model_revision=current_model_revision(), runtime_config=current_runtime_config())
    if not pipeline.reserve(child):
        raise HTTPException(status_code=409, detail="GPU worker is busy; wait for the active job to finish or cancel it")
    try:
        await asyncio.to_thread(store.prepare_resume, source, child, "rig")
        store.save(child)
    except ValueError as exc:
        pipeline.release(child)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        pipeline.release(child)
        raise
    asyncio.create_task(pipeline.run(child))
    return _fresh(child)


def _child_job(source: JobStatus, resume_from: str) -> JobStatus:
    return JobStatus(project_id=source.project_id, backend=source.backend, seed=source.seed, texture=source.texture, image=source.image, reference_images=source.reference_images, control_type=source.control_type, parameters=dict(source.parameters), preset=source.preset, parent_job_id=source.id, resume_from=resume_from, input_mode=source.input_mode, prompt=source.prompt, t2i_seed=source.t2i_seed, t2i_model=source.t2i_model, asset_type=source.asset_type, rig_spec=source.rig_spec, unreal_export=source.unreal_export, model_revision=current_model_revision(), runtime_config=current_runtime_config())


async def _start_attempt(source: JobStatus, resume_stage: str) -> JobStatus:
    if resume_stage in ("shape", "texture") and source.backend != "demo" and not await asyncio.to_thread(hunyuan_service_ready):
        raise HTTPException(status_code=503, detail="Hunyuan worker is still loading. Wait for local runtime readiness and retry.")
    child = _child_job(source, resume_stage)
    if not pipeline.reserve(child):
        raise HTTPException(status_code=409, detail="GPU worker is busy; wait for the active job to finish or cancel it")
    try:
        if resume_stage != "shape":
            await asyncio.to_thread(store.prepare_resume, source, child, resume_stage)
        store.save(child)
    except Exception:
        pipeline.release(child)
        raise
    asyncio.create_task(pipeline.run(child))
    return child


def _fresh(job: JobStatus) -> JobStatus:
    return store.get(job.id) or job


def _terminal_source(job_id: UUID) -> JobStatus:
    source = store.get(job_id)
    if not source:
        raise HTTPException(status_code=404, detail="Job not found")
    if source.stage not in {JobStage.FAILED, JobStage.CANCELLED}:
        raise HTTPException(status_code=409, detail="Only failed or cancelled jobs can be retried")
    return source


@app.post("/api/jobs/{job_id}/retry", response_model=JobStatus, status_code=202, response_model_exclude=PUBLIC_JOB_EXCLUDE)
async def retry_job(job_id: UUID) -> JobStatus:
    source = _terminal_source(job_id)
    return _fresh(await _start_attempt(source, store.earliest_incomplete_stage(source)))


@app.post("/api/jobs/{job_id}/resume", response_model=JobStatus, status_code=202, response_model_exclude=PUBLIC_JOB_EXCLUDE)
async def resume_job(job_id: UUID) -> JobStatus:
    source = _terminal_source(job_id)
    next_stage = store.earliest_incomplete_stage(source)
    if next_stage == "shape":
        raise HTTPException(status_code=409, detail="Job has no valid checkpoint to resume from; use restart or retry")
    return _fresh(await _start_attempt(source, next_stage))


@app.post("/api/jobs/{job_id}/restart", response_model=JobStatus, status_code=202, response_model_exclude=PUBLIC_JOB_EXCLUDE)
async def restart_job(job_id: UUID) -> JobStatus:
    source = _terminal_source(job_id)
    return _fresh(await _start_attempt(source, "shape"))


@app.post("/api/projects", response_model=Project, status_code=201)
async def create_project(request: ProjectCreate) -> Project:
    project = Project(name=request.name, description=request.description)
    store.save_project(project)
    return project


@app.get("/api/projects", response_model=list[Project])
async def list_projects() -> list[Project]:
    return store.list_projects()


@app.get("/api/projects/{project_id}", response_model=Project)
async def get_project(project_id: UUID) -> Project:
    project = store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.get("/api/jobs/{job_id}", response_model=JobStatus, response_model_exclude=PUBLIC_JOB_EXCLUDE)
async def get_job(job_id: UUID) -> JobStatus:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/input")
async def job_input(job_id: UUID) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"image": job.image, "reference_images": [item.model_dump() for item in job.reference_images], "control_data": job.parameters.get("control_data")}


@app.get("/api/jobs", response_model=list[JobStatus])
async def list_jobs(project_id: UUID | None = None) -> JSONResponse:
    jobs = store.list_jobs(project_id)
    return JSONResponse([job.model_dump(mode="json", exclude=PUBLIC_JOB_EXCLUDE) for job in jobs])


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: UUID, request: Request):
    if not store.get(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    async def stream():
        last = None
        while True:
            if await request.is_disconnected():
                break
            job = store.get(job_id)
            if not job:
                break
            encoded = json.dumps(job.model_dump(mode="json", exclude=PUBLIC_JOB_EXCLUDE), separators=(",", ":"))
            if encoded != last:
                yield f"event: job\ndata: {encoded}\n\n"
                last = encoded
            if job.stage.value in {"complete", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/jobs/{job_id}/cancel", response_model=JobStatus, response_model_exclude=PUBLIC_JOB_EXCLUDE)
async def cancel_job(job_id: UUID) -> JobStatus:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.stage not in {JobStage.COMPLETE, JobStage.FAILED, JobStage.CANCELLED}:
        pipeline.request_cancel(str(job_id))
        refreshed = store.get(job_id)
        if refreshed is not None:
            job = refreshed
    return job


@app.get("/api/jobs/{job_id}/artifacts")
async def artifacts(job_id: UUID) -> dict[str, list[str]]:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"artifacts": job.artifacts}


@app.get("/api/jobs/{job_id}/validation")
async def validation(job_id: UUID) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if "validation-report.json" not in {path.split("/", 1)[-1] for path in job.artifacts}:
        raise HTTPException(status_code=409, detail="Unity validation is not available until the job completes")
    report_path = store.path(job_id) / "artifacts" / "validation-report.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
async def artifact(job_id: UUID, artifact_name: str) -> FileResponse:
    job = store.get(job_id)
    if not job or artifact_name not in {path.split("/", 1)[-1] for path in job.artifacts}:
        raise HTTPException(status_code=404, detail="Artifact not found")
    target = (store.path(job_id) / "artifacts" / artifact_name).resolve()
    if store.path(job_id).resolve() not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(target)


async def _settings_payload() -> dict:
    return {"settings": settings_store.load().model_dump(mode="json"), "usage": await asyncio.to_thread(store.disk_usage)}


@app.get("/api/settings")
async def get_settings() -> dict:
    return await _settings_payload()


@app.put("/api/settings")
async def put_settings(payload: AppSettings) -> dict:
    presets = load_generation_presets().get("presets", {})
    if payload.default_preset is not None and payload.default_preset not in presets:
        raise HTTPException(status_code=422, detail=f"Unknown preset: {payload.default_preset}")
    settings_store.save(payload)
    return await _settings_payload()


@app.post("/api/settings/sweep")
async def sweep_now() -> dict:
    return await asyncio.to_thread(sweep_jobs, store, settings_store.load())


@app.delete("/api/jobs/{job_id}", status_code=204)
async def delete_job(job_id: UUID) -> None:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.stage not in TERMINAL_STAGES:
        raise HTTPException(status_code=409, detail="Only terminal jobs can be deleted")
    store.delete_job(job_id)
