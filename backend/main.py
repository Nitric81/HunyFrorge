import asyncio
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal
from uuid import UUID
import importlib.util

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    GenerationSettings,
    HealthResponse,
    JobCreate,
    JobStage,
    JobStatus,
    Project,
    ProjectCreate,
    VehicleRigSpec,
    current_model_revision,
    current_runtime_config,
    load_generation_presets,
    scaffold_prompt,
    t2i_config,
    T2I_MODEL_ID,
)
import json
from .pipeline import Pipeline
from .storage import JobStore
from . import vehicle_rig

ROOT = Path(os.getenv("HUNYFORGE_DATA_ROOT", Path.cwd() / "data"))
store = JobStore(ROOT)
pipeline = Pipeline(store)
app = FastAPI(title="HunyForge API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174", "http://localhost:4173", "http://127.0.0.1:4173"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def recover_interrupted_jobs() -> None:
    store.recover_incomplete_jobs()

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
    return HealthResponse(inference_mode=mode, gpu_available=gpu, model_paths_configured=model_configured, model_snapshot_present=snapshot_present, model_snapshot_path=snapshot_root if snapshot_present else None, hunyuan_service_ready=service_ready, runtime_ready=mode == "demo" or (gpu and model_configured and snapshot_present and service_ready), workers=worker_health(), runtime_config=current_runtime_config(), t2i=t2i_config(), multi_view={"enabled": multi_view_enabled or mode == "demo", "adapter_configured": bool(os.getenv("HUNYFORGE_MULTI_VIEW_URL", "").strip()), "reason": None if multi_view_enabled or mode == "demo" else "Start the local Hunyuan3D-2mv worker and set HUNYFORGE_MULTI_VIEW_URL."})


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
