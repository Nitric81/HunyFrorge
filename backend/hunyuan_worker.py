import asyncio
import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import GenerationSettings
from .worker_runtime import RuntimeEngine


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Hunyuan3D Worker", version="2.1.0", lifespan=lifespan)


def get_engine() -> RuntimeEngine:
    if getattr(app.state, "engine", None) is None:
        root = Path(os.environ.get("HUNYFORGE_DATA_ROOT", Path.cwd() / "data"))
        app.state.engine = RuntimeEngine(root)
    return app.state.engine


@app.get("/health")
async def health():
    engine = get_engine()
    status = await asyncio.to_thread(engine.health)
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
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            raise HTTPException(status_code=503, detail="GPU out of memory; reduce settings or wait.") from e
        raise HTTPException(status_code=500, detail=msg) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
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
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            raise HTTPException(status_code=503, detail="GPU out of memory; wait for the active job to finish and retry.") from e
        raise HTTPException(status_code=500, detail=msg) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if engine._worker_lock.locked():
            engine._worker_lock.release()
