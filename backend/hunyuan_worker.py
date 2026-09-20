import asyncio
import base64
import json
import os
import shutil
import signal
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.background import BackgroundTask

from .models import AppSettings, GenerationSettings
from .storage import SettingsStore
from .worker_runtime import RuntimeEngine, memory_snapshot


def _runtime_settings() -> AppSettings:
    """Live user settings; falls back to env-seeded defaults when the data
    root has no settings.json or it cannot be read."""
    try:
        root = Path(os.environ.get("HUNYFORGE_DATA_ROOT", Path.cwd() / "data"))
        return SettingsStore(root).load()
    except Exception:
        return AppSettings()


class MeshToken(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: UUID
    artifact: Literal["white-mesh.glb"]
    sha256: str = Field(pattern="^[0-9a-f]{64}$")


class WorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: UUID
    stage: Literal["shape", "texture"]
    seed: int = Field(ge=0, le=2**32 - 1)
    image: str = Field(min_length=1, max_length=14000000)
    settings: GenerationSettings
    mesh_token: MeshToken | None = None

    @model_validator(mode="after")
    def _check_mesh_token(self):
        if self.stage == "shape" and self.mesh_token is not None:
            raise ValueError("Shape stage must not include a mesh token")
        if self.stage == "texture":
            if self.mesh_token is None:
                raise ValueError("Texture stage requires a mesh token")
            if self.mesh_token.job_id != self.job_id:
                raise ValueError("Mesh token job_id does not match request job_id")
        return self


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=2000)
    seed: int = Field(ge=0, le=2**32 - 1)
    image: str | None = Field(default=None, max_length=14000000)
    width: Literal[512, 768, 1024] = 1024
    height: Literal[512, 768, 1024] = 1024
    num_inference_steps: int = Field(default=4, ge=1, le=28)


def stage_isolation_enabled() -> bool:
    """Run each stage in a fresh subprocess so exit reclaims all model RSS."""
    return _runtime_settings().stage_isolation


def admission_error() -> str | None:
    """Reject work when the VM lacks headroom instead of OOMing mid-stage."""
    required = _runtime_settings().min_available_mb
    if required <= 0:
        return None
    available = memory_snapshot().get("available_mb")
    if available is not None and available < required:
        return f"insufficient_memory: {available}MiB available < {required}MiB required (min_available_mb setting)"
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Hunyuan3D Worker", version="2.1.0", lifespan=lifespan)


def get_engine() -> RuntimeEngine:
    if getattr(app.state, "engine", None) is None:
        root = Path(os.environ.get("HUNYFORGE_DATA_ROOT", Path.cwd() / "data"))
        app.state.engine = RuntimeEngine(root)
    return app.state.engine


_STAGE_ERROR_TYPES = {"ValueError": ValueError, "MemoryError": MemoryError}


def _stage_error(result: dict) -> Exception:
    message = result.get("error") or "stage failed"
    exc_type = _STAGE_ERROR_TYPES.get(result.get("error_type"), RuntimeError)
    return exc_type(message)


async def _spawn_and_collect(kind: str, payload: dict, slot: Path) -> dict:
    request_path = slot / "request.json"
    result_path = slot / "result.json"
    request_path.write_text(json.dumps({"kind": kind, "payload": payload}), encoding="utf-8")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "backend.stage_runner",
        "--request", str(request_path), "--result", str(result_path),
        stderr=asyncio.subprocess.PIPE,
    )
    app.state.active_stage = {"pid": proc.pid, "kind": kind}
    try:
        _, stderr = await proc.communicate()
    except asyncio.CancelledError:
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                proc.kill()
                await proc.wait()
        raise
    finally:
        app.state.active_stage = None
    result = None
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = None
    if result and result.get("status") == "ok":
        return result
    if result and result.get("status") == "error":
        raise _stage_error(result)
    tail = (stderr or b"").decode("utf-8", errors="replace")[-1500:].strip()
    if proc.returncode == -getattr(signal, "SIGKILL", 9):
        raise RuntimeError(f"worker_oom_killed: stage process was SIGKILLed (kernel OOM killer); check docker logs for the Killed line")
    raise RuntimeError(f"stage process exited with code {proc.returncode}: {tail or 'no stderr captured'}")


async def _isolated_stage(kind: str, payload: dict) -> tuple[dict, Path]:
    slot = Path(tempfile.mkdtemp(prefix=f"hunyforge-{kind}-"))
    try:
        return await _spawn_and_collect(kind, payload, slot), slot
    except BaseException:
        shutil.rmtree(slot, ignore_errors=True)
        raise


def _map_stage_exception(e: Exception):
    if isinstance(e, ValueError):
        return HTTPException(status_code=422, detail=str(e))
    msg = str(e)
    lowered = msg.lower()
    if isinstance(e, MemoryError) or "out of memory" in lowered or "worker_oom_killed" in lowered:
        return HTTPException(status_code=503, detail=msg or "Worker out of memory")
    return HTTPException(status_code=500, detail=msg)


@app.get("/health")
async def health():
    engine = get_engine()
    status = await asyncio.to_thread(engine.health)
    status["memory"] = await asyncio.to_thread(memory_snapshot)
    status["stage_isolation"] = stage_isolation_enabled()
    status["busy"] = status["busy"] or engine._worker_lock.locked()
    active = getattr(app.state, "active_stage", None)
    if active:
        status["active_stage"] = active
    if status["ready"]:
        return JSONResponse(content=status)
    raise HTTPException(status_code=503, detail=status)


@app.post("/generate")
async def generate(request: WorkerRequest):
    engine = get_engine()
    if not await asyncio.to_thread(engine._worker_lock.acquire, blocking=False):
        raise HTTPException(status_code=409, detail="Worker busy")
    task = None
    try:
        reason = admission_error()
        if reason:
            raise HTTPException(status_code=503, detail=reason)
        if stage_isolation_enabled():
            result, slot = await _isolated_stage("generate", request.model_dump(mode="json"))
            artifact = Path(result["artifact"])
            if not artifact.is_file():
                raise RuntimeError(f"stage artifact missing: {artifact}")
            return FileResponse(artifact, filename=artifact.name, media_type="model/gltf-binary", background=BackgroundTask(shutil.rmtree, slot, ignore_errors=True))
        task = asyncio.ensure_future(asyncio.to_thread(engine.generate, request))
        shielded = asyncio.shield(task)
        result = await shielded
        return FileResponse(result, filename=result.name, media_type="model/gltf-binary")
    except asyncio.CancelledError:
        if task is not None:
            try:
                await task
            except Exception:
                pass
        raise
    except HTTPException:
        raise
    except (ValueError, RuntimeError, MemoryError) as e:
        raise _map_stage_exception(e) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if engine._worker_lock.locked():
            engine._worker_lock.release()


@app.post("/preview")
async def preview(request: PreviewRequest):
    engine = get_engine()
    if not await asyncio.to_thread(engine._worker_lock.acquire, blocking=False):
        raise HTTPException(status_code=409, detail="Worker busy")
    task = None
    try:
        reason = admission_error()
        if reason:
            raise HTTPException(status_code=503, detail=reason)
        if stage_isolation_enabled():
            result, slot = await _isolated_stage("preview", request.model_dump(mode="json"))
            artifact = Path(result["artifact"])
            png = await asyncio.to_thread(artifact.read_bytes)
            timings = result.get("timings", {})
            content = {"image": "data:image/png;base64," + base64.b64encode(png).decode("ascii"), "timings": timings}
            return JSONResponse(content=content, background=BackgroundTask(shutil.rmtree, slot, ignore_errors=True))
        task = asyncio.ensure_future(asyncio.to_thread(engine.generate_preview, request))
        png, records = await asyncio.shield(task)
        timings = {name: round(record.get("elapsed_seconds") or 0, 2) for name, record in records.items()}
        return JSONResponse(content={"image": "data:image/png;base64," + base64.b64encode(png).decode("ascii"), "timings": timings})
    except asyncio.CancelledError:
        if task is not None:
            try:
                await task
            except Exception:
                pass
        raise
    except HTTPException:
        raise
    except (ValueError, RuntimeError, MemoryError) as e:
        raise _map_stage_exception(e) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if engine._worker_lock.locked():
            engine._worker_lock.release()
