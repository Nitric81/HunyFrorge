"""Local Hunyuan3D-2mv shape worker for HunyForge.

The official 2mv checkpoint fuses named canonical views. It is deliberately a
separate process from HunyForge's Hunyuan3D-2.1 worker because their Python
packages and checkpoints are different generations.
"""

import asyncio
import base64
import os
import tempfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


CANONICAL_VIEWS = {"front", "left", "back", "right"}


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    view: Literal["front", "rear", "left", "right"]
    image: str = Field(min_length=1, max_length=14_000_000)


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    num_inference_steps: int = Field(default=5, ge=1, le=100)
    guidance_scale: float = Field(default=5.0, ge=0.1, le=20)
    octree_resolution: int = Field(default=380, ge=16, le=512)
    num_chunks: int = Field(default=20000, ge=1000, le=5_000_000)
    remove_background: bool = True


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(ge=0, le=2**32 - 1)
    references: list[Reference] = Field(min_length=1, max_length=4)
    settings: Settings

    @model_validator(mode="after")
    def validate_views(self):
        views = [reference.view for reference in self.references]
        if len(set(views)) != len(views):
            raise ValueError("Duplicate multi-view reference labels are not allowed")
        return self


class Engine:
    def __init__(self):
        self.pipeline = None
        self.remover = None
        self.lock = Lock()

    def ready(self):
        root = Path(os.environ.get("HUNYUAN2MV_MODEL_PATH", "/models/Hunyuan3D-2mv"))
        return root.is_dir() and any(root.iterdir())

    @staticmethod
    def _decode(value: str) -> bytes:
        payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        try:
            return base64.b64decode(payload, validate=True)
        except Exception as error:
            raise ValueError("Invalid base64 reference image") from error

    def _load_pipeline(self):
        if self.pipeline is not None:
            return
        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        path = os.environ.get("HUNYUAN2MV_MODEL_PATH", "/models/Hunyuan3D-2mv")
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            path,
            subfolder=os.environ.get("HUNYUAN2MV_SUBFOLDER", "hunyuan3d-dit-v2-mv-turbo"),
            variant="fp16",
        )
        if os.environ.get("HUNYUAN2MV_FLASHVDM", "1") == "1":
            self.pipeline.enable_flashvdm()
        self.torch = torch

    def generate(self, request: GenerateRequest) -> Path:
        if not self.ready():
            raise RuntimeError("Hunyuan3D-2mv weights are not mounted at HUNYUAN2MV_MODEL_PATH")
        from PIL import Image
        from hy3dgen.rembg import BackgroundRemover

        self._load_pipeline()
        images = {}
        for reference in request.references:
            view = "back" if reference.view == "rear" else reference.view
            image = Image.open(BytesIO(self._decode(reference.image))).convert("RGBA")
            if request.settings.remove_background:
                self.remover = self.remover or BackgroundRemover()
                image = self.remover(image)
            images[view] = image
        with self.torch.inference_mode():
            mesh = self.pipeline(
                image=images,
                num_inference_steps=request.settings.num_inference_steps,
                guidance_scale=request.settings.guidance_scale,
                octree_resolution=request.settings.octree_resolution,
                num_chunks=request.settings.num_chunks,
                generator=self.torch.Generator(device="cuda").manual_seed(request.seed),
                output_type="trimesh",
            )[0]
        directory = Path(tempfile.mkdtemp(prefix="hunyforge-2mv-"))
        output = directory / "shape.glb"
        mesh.export(output)
        return output


engine = Engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="HunyForge Hunyuan3D-2mv Worker", version="2.0", lifespan=lifespan)


@app.get("/health")
async def health():
    if not engine.ready():
        raise HTTPException(status_code=503, detail={"ready": False, "reason": "Hunyuan3D-2mv weights unavailable"})
    return JSONResponse({"ready": True, "model": "Hunyuan3D-2mv", "loaded": engine.pipeline is not None})


@app.post("/generate")
async def generate(request: GenerateRequest):
    if not engine.lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Hunyuan3D-2mv worker is busy")
    try:
        output = await asyncio.to_thread(engine.generate, request)
        return FileResponse(output, filename="shape.glb", media_type="model/gltf-binary")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    finally:
        engine.lock.release()
