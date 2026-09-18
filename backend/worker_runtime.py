import base64
import ctypes
import gc
import json
import logging
import os
import shutil
import threading
import time
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

from .models import DEFAULT_MODEL_REVISION, DEFAULT_SOURCE_REVISION, GenerationSettings, JobStatus, current_runtime_config
from .storage import sha256_file
from .telemetry import StageTelemetry, rss_mb


logger = logging.getLogger(__name__)

MAX_IMAGE_PIXELS = 16_777_216
WHITELISTED_ARTIFACTS = frozenset({"white-mesh.glb"})


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.02)


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        resolved = Path(os.path.realpath(path))
        root_resolved = Path(os.path.realpath(root))
        resolved.relative_to(root_resolved)
        return True
    except ValueError:
        return False


def memory_snapshot() -> dict:
    """Process RSS plus VM headroom; available/total only on Linux."""
    snap = {}
    rss = rss_mb()
    if rss is not None:
        snap["rss_mb"] = int(rss)
    try:
        meminfo = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            fields = rest.split()
            if fields:
                meminfo[key] = int(fields[0])
        snap["available_mb"] = meminfo.get("MemAvailable", 0) // 1024
        snap["total_mb"] = meminfo.get("MemTotal", 0) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return snap


def _malloc_trim() -> None:
    """Return free glibc arena pages to the kernel; no-op elsewhere."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class _TimedModelProxy:
    __slots__ = ("_model", "_prefix", "_meter", "_counter")

    def __init__(self, model, prefix, meter):
        self._model = model
        self._prefix = prefix
        self._meter = meter
        self._counter = 0

    def __getattr__(self, name):
        if name in self.__slots__:
            raise AttributeError(name)
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        n = self._counter
        self._counter += 1
        with self._meter.measure(f"{self._prefix}_{n}"):
            return self._model(*args, **kwargs)


class RuntimeEngine:
    def __init__(self, root: Path, dependencies=None):
        self.root = Path(root).resolve()
        self._dependencies = dependencies
        self.shape_pipeline = None
        self.paint_pipeline = None
        self.remover = None
        self.t2i_pipeline = None
        self.states = {"shape": "unloaded", "texture": "unloaded", "reference": "unloaded"}
        self.error = None
        self.active_job_id = None
        self.last_runtime = None
        self._worker_lock = threading.Lock()
        self._deps_lock = threading.Lock()

    @classmethod
    def default_dependencies(cls):
        import os
        import torch
        from PIL import Image
        from model_worker import (
            Hunyuan3DDiTFlowMatchingPipeline,
            BackgroundRemover,
            Hunyuan3DPaintConfig,
            Hunyuan3DPaintPipeline,
            quick_convert_with_obj2gltf,
        )
        deps = {
            "shape_factory": Hunyuan3DDiTFlowMatchingPipeline,
            "paint_factory": Hunyuan3DPaintPipeline,
            "remover_factory": BackgroundRemover,
            "paint_config": Hunyuan3DPaintConfig,
            "convert": quick_convert_with_obj2gltf,
            "torch": torch,
            "Image": Image,
            "t2i_factory": None,
        }
        if os.getenv("HUNYFORGE_T2I_ENABLED", "0") == "1":
            try:
                from diffusers import Flux2KleinPipeline
                deps["t2i_factory"] = Flux2KleinPipeline
            except ImportError:
                pass
        return deps

    def _get_dep(self, name: str):
        if self._dependencies is None:
            with self._deps_lock:
                if self._dependencies is None:
                    self._dependencies = self.default_dependencies()
        if name not in self._dependencies:
            raise RuntimeError(f"Missing dependency: {name}")
        return self._dependencies[name]

    def _validate_config(self, set_threads: bool = True):
        cpu_threads = os.getenv("HUNYUAN_CPU_THREADS", "8")
        try:
            n = int(cpu_threads)
            if not 1 <= n <= 32:
                raise ValueError
        except ValueError:
            raise ValueError(f"HUNYUAN_CPU_THREADS must be an integer between 1 and 32, got {cpu_threads!r}")
        flash = os.getenv("HUNYUAN_FLASHVDM", "0")
        if flash not in ("0", "1"):
            raise ValueError(f"HUNYUAN_FLASHVDM must be 0 or 1, got {flash!r}")
        compile = os.getenv("HUNYUAN_COMPILE", "0")
        if compile not in ("0", "1"):
            raise ValueError(f"HUNYUAN_COMPILE must be 0 or 1, got {compile!r}")
        dino = os.getenv("HUNYUAN_DINO_DEVICE", "cpu")
        if dino not in ("cpu", "cuda"):
            raise ValueError(f"HUNYUAN_DINO_DEVICE must be cpu or cuda, got {dino!r}")
        if set_threads:
            torch = self._get_dep("torch")
            torch.set_num_threads(n)
        return {"cpu_threads": n, "flashvdm": flash, "compile": compile, "dino_device": dino}

    def _ready(self) -> tuple[bool, str | None]:
        try:
            self._validate_config(set_threads=False)
        except ValueError as e:
            return False, str(e)
        model_path = os.environ.get("HUNYUAN_MODEL_PATH", "")
        if not model_path:
            return False, "HUNYUAN_MODEL_PATH is not configured"
        model_root = Path(model_path)
        if not model_root.is_dir():
            return False, f"Model path does not exist: {model_path}"
        if not (model_root / "hunyuan3d-dit-v2-1" / "model.fp16.ckpt").is_file():
            return False, f"Shape checkpoint missing under {model_path}/hunyuan3d-dit-v2-1"
        if not (model_root / "hunyuan3d-paintpbr-v2-1").is_dir():
            return False, f"Paint model missing under {model_path}/hunyuan3d-paintpbr-v2-1"
        try:
            torch = self._get_dep("torch")
            self._get_dep("Image")
        except Exception as e:
            return False, f"Runtime dependency import failed: {e}"
        try:
            if not torch.cuda.is_available():
                return False, "CUDA is not available"
        except Exception as e:
            return False, f"CUDA availability check failed: {e}"
        return True, None

    def health(self) -> dict:
        ready, ready_error = self._ready()
        return {
            "workers": {
                "shape": {"state": self.states["shape"]},
                "texture": {"state": self.states["texture"]},
                "reference": {"state": self.states["reference"]},
            },
            "ready": ready,
            "busy": self.active_job_id is not None or self.states["reference"] in ("loading", "running"),
            "active_job_id": self.active_job_id,
            "runtime_config": current_runtime_config(),
            "last_error": self.error or ready_error,
            "memory": memory_snapshot(),
        }

    def _raw_base64(self, value: str) -> str:
        if not value:
            raise ValueError("Image missing")
        if value.startswith(("http://", "https://")):
            raise ValueError("Remote image URLs are not allowed")
        if value.startswith("data:"):
            if "," not in value:
                raise ValueError("Malformed data URL")
            return value.split(",", 1)[1]
        return value

    def _decode_base64(self, value: str) -> bytes:
        payload = self._raw_base64(value)
        if not payload:
            raise ValueError("Empty image payload")
        try:
            return base64.b64decode(payload, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {e}") from e

    def _load_image(self, raw_bytes: bytes):
        Image = self._get_dep("Image")
        try:
            img = Image.open(BytesIO(raw_bytes))
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}") from e
        fmt = getattr(img, "format", None)
        if fmt not in ("PNG", "JPEG"):
            raise ValueError(f"Unsupported image format: {fmt}")
        if img.width * img.height > MAX_IMAGE_PIXELS:
            raise ValueError(f"Image dimensions {img.width}x{img.height} exceed maximum pixel count")
        orig_max = None
        if hasattr(Image, "MAX_IMAGE_PIXELS"):
            orig_max = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        try:
            img.load()
        except Exception as e:
            raise ValueError(f"Image load failed: {e}") from e
        finally:
            if orig_max is not None:
                Image.MAX_IMAGE_PIXELS = orig_max
        try:
            return img.convert("RGBA")
        except Exception as e:
            raise ValueError(f"Image convert failed: {e}") from e

    def _load_job(self, request):
        job_dir = self.root / "jobs" / str(request.job_id)
        if not job_dir.exists():
            raise ValueError("Job directory not found")
        if not _is_path_within(job_dir, self.root):
            raise ValueError("Job directory escapes root")
        job_path = job_dir / "job.json"
        if not job_path.is_file():
            raise ValueError("Job metadata missing")
        if not _is_path_within(job_path, self.root):
            raise ValueError("Job metadata escapes root")
        job = JobStatus.model_validate(json.loads(job_path.read_text(encoding="utf-8")))
        if job.id != request.job_id:
            raise ValueError("Persisted job id does not match request job_id")
        if job.backend != "hunyuan3d-2.1":
            raise ValueError(f"Worker only accepts hunyuan3d-2.1 jobs, got {job.backend}")
        if request.stage == "texture" and not job.texture:
            raise ValueError("Texture stage requested for a shape-only job")
        if job.stage == "complete":
            raise ValueError("Completed jobs are immutable; worker rerun is not permitted")
        return job_dir, job

    def _validate_request(self, request, job, raw_image: bytes):
        if request.seed != job.seed:
            raise ValueError("Seed does not match persisted job seed")
        expected = {k: job.parameters.get(k) for k in GenerationSettings.model_fields}
        actual = request.settings.model_dump()
        if actual != expected:
            raise ValueError("Settings do not match persisted job settings")
        job_bytes = self._decode_base64(job.image or "")
        if raw_image != job_bytes:
            raise ValueError("Image does not match persisted job image")
        if request.stage == "shape":
            if request.mesh_token is not None:
                raise ValueError("Shape stage must not include a mesh token")
        elif request.stage == "texture":
            if request.mesh_token is None:
                raise ValueError("Texture stage requires a mesh token")
            if request.mesh_token.job_id != request.job_id:
                raise ValueError("Mesh token job_id does not match request job_id")
            if request.mesh_token.artifact not in WHITELISTED_ARTIFACTS:
                raise ValueError(f"Artifact {request.mesh_token.artifact} is not whitelisted")
        else:
            raise ValueError(f"Unknown stage: {request.stage}")

    def _verify_mesh_token(self, token, job_dir: Path):
        artifact_path = job_dir / "artifacts" / token.artifact
        if not artifact_path.is_file():
            raise ValueError(f"Mesh artifact not found: {token.artifact}")
        if not _is_path_within(artifact_path, self.root) or not _is_path_within(artifact_path, job_dir):
            raise ValueError("Mesh artifact path escapes root")
        actual_hash = sha256_file(artifact_path)
        if actual_hash != token.sha256:
            raise ValueError("Mesh token hash mismatch")
        return artifact_path

    def _checked_output_path(self, path: Path, job_dir: Path) -> Path:
        if path.is_symlink():
            raise ValueError("Worker output path is a symlink")
        if not _is_path_within(path, self.root) or not _is_path_within(path, job_dir):
            raise ValueError("Worker output path escapes job directory")
        return path

    def _ensure_stage_dir(self, job_dir: Path, stage: str) -> Path:
        stage_dir = self._checked_output_path(job_dir / "worker" / stage, job_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        invocation_dir = self._checked_output_path(stage_dir / uuid4().hex, job_dir)
        invocation_dir.mkdir(parents=True, exist_ok=True)
        return invocation_dir

    def _release_models(self):
        self.shape_pipeline = None
        self.paint_pipeline = None
        self.remover = None
        self.t2i_pipeline = None
        gc.collect()
        try:
            torch = self._get_dep("torch")
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        _malloc_trim()

    def _instrument_paint_pipeline(self, pipeline, meter):
        vp = pipeline.view_processor
        names = ("bake_view_selection", "render_normal_multiview", "render_position_multiview", "bake_from_multiview", "texture_inpaint")
        for name in names:
            if not hasattr(vp, name):
                continue
            orig = getattr(vp, name)
            counter = [0]
            def make_wrapper(o, n, c):
                def wrapper(*args, **kwargs):
                    i = c[0]
                    c[0] = i + 1
                    with meter.measure(f"{n}_{i}"):
                        return o(*args, **kwargs)
                return wrapper
            setattr(vp, name, make_wrapper(orig, name, counter))
        for key, prefix in (("multiview_model", "multiview_model"), ("super_model", "super_model")):
            if key in pipeline.models:
                orig = pipeline.models[key]
                pipeline.models[key] = _TimedModelProxy(orig, prefix, meter)

    def _write_runtime_provenance(self, job_dir: Path, stage: str, request, image, settings, error=None):
        try:
            torch = self._get_dep("torch")
        except Exception:
            torch = None
        provenance = {
            "source_revision": os.getenv("HUNYUAN_SOURCE_REVISION", DEFAULT_SOURCE_REVISION),
            "model_revision": os.getenv("HUNYUAN_MODEL_REVISION") or DEFAULT_MODEL_REVISION,
            "torch_version": getattr(torch, "__version__", None) if torch else None,
            "cuda_version": None,
            "gpu_name": None,
            "input_dimensions": (image.width, image.height) if image else None,
            "seed": request.seed,
            "settings": settings.model_dump(),
            "shape_config": {"box_v": 1.01, "mc_level": 0.0, "mc_algo": "mc"},
            "process_id": os.getpid(),
        }
        if torch is not None:
            try:
                provenance["cuda_version"] = getattr(torch.version, "cuda", None)
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    provenance["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
        if error is not None:
            provenance["error"] = str(error)
        path = self._checked_output_path(job_dir / "telemetry" / f"{stage}-runtime.json", job_dir)
        _atomic_write_json(path, provenance)
        self.last_runtime = provenance

    def _run_shape(self, request, job_dir: Path, stage_dir: Path, image, meter):
        stage = "shape"
        output_path = stage_dir / "white-mesh.glb"
        self.states["shape"] = "loading"
        torch = self._get_dep("torch")
        latents = None
        mesh = None
        with meter.measure(stage):
            try:
                with meter.measure("background_removal"):
                    if request.settings.remove_background:
                        self.remover = self._get_dep("remover_factory")()
                        image = self.remover(image)
                    meter.update("background_removal", {"remove_background": request.settings.remove_background})
                with torch.inference_mode():
                    with meter.measure("shape_model_loading"):
                        self.shape_pipeline = self._get_dep("shape_factory").from_pretrained(
                            os.environ["HUNYUAN_MODEL_PATH"],
                            device="cuda",
                            dtype=torch.float16,
                            use_safetensors=False,
                            variant="fp16",
                            subfolder="hunyuan3d-dit-v2-1",
                        )
                        if os.getenv("HUNYUAN_FLASHVDM", "0") == "1":
                            self.shape_pipeline.enable_flashvdm(mc_algo="mc", replace_vae=False)
                        if os.getenv("HUNYUAN_COMPILE", "0") == "1":
                            self.shape_pipeline.compile()
                        self.states["shape"] = "running"
                    with meter.measure("shape_inference"):
                        latents = self.shape_pipeline(
                            image=image,
                            num_inference_steps=request.settings.num_inference_steps,
                            guidance_scale=request.settings.guidance_scale,
                            octree_resolution=request.settings.octree_resolution,
                            num_chunks=request.settings.num_chunks,
                            generator=torch.Generator(device="cuda").manual_seed(request.seed),
                            box_v=1.01,
                            mc_level=0.0,
                            mc_algo="mc",
                            output_type="latent",
                            enable_pbar=False,
                            callback=lambda step, t, outputs: meter.update("shape_diffusion", {"step": step + 1, "total": request.settings.num_inference_steps}),
                            callback_steps=1,
                        )
                    with meter.measure("mesh_extraction"):
                        mesh = self.shape_pipeline._export(
                            latents,
                            output_type="trimesh",
                            box_v=1.01,
                            mc_level=0.0,
                            num_chunks=request.settings.num_chunks,
                            octree_resolution=request.settings.octree_resolution,
                            mc_algo="mc",
                            enable_pbar=False,
                        )[0]
                    latents = None
                    with meter.measure("glb_conversion"):
                        mesh.export(output_path)
                    mesh = None
            except Exception as e:
                self.states["shape"] = "failed"
                self.error = str(e)
                raise
            finally:
                latents = None
                mesh = None
                stage_failed = self.states.get(stage) == "failed"
                self.states[stage] = "unloading"
                with meter.measure("model_release"):
                    self._release_models()
                if stage_failed:
                    self.states[stage] = "failed"
                else:
                    self.states[stage] = "unloaded"
        return output_path

    def _run_texture(self, request, job_dir: Path, stage_dir: Path, image, meter):
        stage = "texture"
        output_path = stage_dir / "textured-mesh.glb"
        self.states["texture"] = "loading"
        torch = self._get_dep("torch")
        token = request.mesh_token
        source_path = self._verify_mesh_token(token, job_dir)
        verified_white_path = stage_dir / "white-mesh.glb"
        with meter.measure(stage):
            try:
                with meter.measure("artifact_transfer"):
                    shutil.copyfile(source_path, verified_white_path)
                copied_hash = sha256_file(verified_white_path)
                if copied_hash != token.sha256:
                    raise ValueError("Copied white mesh hash does not match token")
                with meter.measure("background_removal"):
                    if request.settings.remove_background:
                        self.remover = self._get_dep("remover_factory")()
                        image = self.remover(image)
                    meter.update("background_removal", {"remove_background": request.settings.remove_background})
                conf = self._get_dep("paint_config")(request.settings.max_num_view, request.settings.multiview_resolution)
                conf.render_size = request.settings.render_size
                conf.texture_size = request.settings.texture_size
                conf.max_selected_view_num = request.settings.max_num_view
                conf.resolution = request.settings.multiview_resolution
                conf.texture_inference_steps = request.settings.texture_inference_steps
                conf.texture_guidance_scale = request.settings.texture_guidance_scale
                conf.face_count = request.settings.face_count
                conf.seed = request.seed
                conf.telemetry = meter
                conf.dino_device = os.getenv("HUNYUAN_DINO_DEVICE", "cpu")
                conf.multiview_pretrained_path = os.environ["HUNYUAN_MODEL_PATH"]
                conf.dino_ckpt_path = os.getenv("HUNYUAN_DINO_PATH", "facebook/dinov2-giant")
                conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
                conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
                conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
                with torch.inference_mode():
                    with meter.measure("texture_model_loading"):
                        self.paint_pipeline = self._get_dep("paint_factory")(conf)
                        if os.getenv("HUNYUAN_COMPILE", "0") == "1":
                            self.paint_pipeline.models["multiview_model"].pipeline.unet = torch.compile(self.paint_pipeline.models["multiview_model"].pipeline.unet)
                        self.states["texture"] = "running"
                    self._instrument_paint_pipeline(self.paint_pipeline, meter)
                    with meter.measure("texture_inference"):
                        obj = self.paint_pipeline(
                            mesh_path=str(verified_white_path),
                            image_path=image,
                            output_mesh_path=str(stage_dir / "textured.obj"),
                            use_remesh=True,
                            save_glb=False,
                        )
                    with meter.measure("glb_conversion"):
                        self._get_dep("convert")(str(obj), str(output_path))
            except Exception as e:
                self.states["texture"] = "failed"
                self.error = str(e)
                raise
            finally:
                stage_failed = self.states.get(stage) == "failed"
                self.states[stage] = "unloading"
                with meter.measure("model_release"):
                    self._release_models()
                if stage_failed:
                    self.states[stage] = "failed"
                else:
                    self.states[stage] = "unloaded"
        return output_path

    def _t2i_ready(self) -> str | None:
        if os.getenv("HUNYFORGE_T2I_ENABLED", "0") != "1":
            return "T2I reference generation is disabled (HUNYFORGE_T2I_ENABLED)"
        model_path = os.environ.get("HUNYFORGE_T2I_MODEL_PATH", "")
        if not model_path or not Path(model_path).is_dir():
            return "T2I model path is not configured or missing"
        try:
            factory = self._get_dep("t2i_factory")
        except RuntimeError as e:
            return str(e)
        if factory is None:
            return "Flux2KleinPipeline is unavailable in this runtime"
        return None

    def generate_preview(self, request) -> tuple[bytes, dict]:
        self.error = None
        t2i_error = self._t2i_ready()
        if t2i_error:
            self.error = t2i_error
            raise ValueError(t2i_error)
        image = self._load_image(self._decode_base64(request.image)) if request.image else None
        telemetry_dir = self.root / "telemetry" / "previews"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = telemetry_dir / f"{uuid4().hex}.json"
        with StageTelemetry(telemetry_path, gpu=True) as meter:
            with meter.measure("reference"):
                try:
                    self.states["reference"] = "loading"
                    torch = self._get_dep("torch")
                    with torch.inference_mode():
                        with meter.measure("t2i_model_loading"):
                            self.t2i_pipeline = self._get_dep("t2i_factory").from_pretrained(
                                os.environ["HUNYFORGE_T2I_MODEL_PATH"],
                                torch_dtype=torch.bfloat16,
                            )
                            self.t2i_pipeline.enable_model_cpu_offload()
                            self.states["reference"] = "running"
                        with meter.measure("t2i_inference"):
                            result = self.t2i_pipeline(
                                prompt=request.prompt,
                                image=[image] if image is not None else None,
                                width=request.width,
                                height=request.height,
                                num_inference_steps=request.num_inference_steps,
                                guidance_scale=1.0,
                                generator=torch.Generator(device="cuda").manual_seed(request.seed),
                                output_type="pil",
                            )
                    output = result.images[0]
                    buffer = BytesIO()
                    output.save(buffer, format="PNG")
                except Exception as e:
                    self.states["reference"] = "failed"
                    self.error = str(e)
                    raise
                finally:
                    stage_failed = self.states["reference"] == "failed"
                    self.states["reference"] = "unloading"
                    with meter.measure("model_release"):
                        self._release_models()
                    self.states["reference"] = "failed" if stage_failed else "unloaded"
        return buffer.getvalue(), dict(meter.records)

    def generate(self, request) -> Path:
        self.error = None
        self.active_job_id = str(request.job_id)
        try:
            self._validate_config()
            model_path = os.environ.get("HUNYUAN_MODEL_PATH", "")
            if not model_path or not Path(model_path).is_dir():
                raise ValueError("HUNYUAN_MODEL_PATH not configured or missing")
            job_dir, job = self._load_job(request)
            raw_image = self._decode_base64(request.image)
            image = self._load_image(raw_image)
            self._validate_request(request, job, raw_image)
            stage = request.stage
            stage_dir = self._ensure_stage_dir(job_dir, stage)
            telemetry_path = self._checked_output_path(job_dir / "telemetry" / f"{stage}.json", job_dir)
            provenance_error = None
            try:
                with StageTelemetry(telemetry_path, gpu=True) as meter:
                    if stage == "shape":
                        result = self._run_shape(request, job_dir, stage_dir, image, meter)
                    else:
                        result = self._run_texture(request, job_dir, stage_dir, image, meter)
            except Exception as e:
                provenance_error = e
                raise
            finally:
                try:
                    self._write_runtime_provenance(job_dir, stage, request, image, request.settings, error=provenance_error)
                except Exception as e:
                    logger.warning("Runtime provenance write failed for job %s stage %s: %s", request.job_id, stage, e)
                    note = f"Runtime provenance write failed: {e}"
                    self.error = f"{self.error}; {note}" if self.error else note
            return result
        except Exception as e:
            self.error = str(e)
            raise
        finally:
            self.active_job_id = None
