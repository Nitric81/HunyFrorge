import asyncio
import json
import base64
import os
import time
import urllib.request
import urllib.error
import struct
import zipfile
import shlex
import shutil
import subprocess
import tempfile
import math
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from threading import Lock
import numpy as np
import trimesh

from .models import (
    GenerationSettings,
    JobStage,
    JobStatus,
    PIPELINE_STAGES,
    StageState,
    TERMINAL_STAGES,
    current_model_revision,
    current_runtime_config,
    now,
    resolve_generation_settings,
)
from .storage import JobStore, sha256_file
from .telemetry import StageTelemetry
from . import vehicle_rig

UNITY_DIR = Path(__file__).resolve().parent / "unity"


class InferenceAdapter:
    name = "base"

    async def generate_shape(self, seed: int, parameters: dict, output: Path) -> None:
        raise NotImplementedError

    async def generate_textures(self, mesh: Path, parameters: dict, output: Path) -> None:
        raise NotImplementedError


class DemoAdapter(InferenceAdapter):
    name = "demo"

    async def generate_shape(self, seed: int, parameters: dict, output: Path) -> None:
        await asyncio.sleep(0.05)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(make_demo_glb())

    async def generate_textures(self, mesh: Path, parameters: dict, output: Path) -> None:
        await asyncio.sleep(0.05)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(make_demo_glb(textured=True))


def make_demo_glb(textured: bool = False) -> bytes:
    """Create a small valid triangle GLB for offline UI/export testing."""
    positions = struct.pack("<9f", -0.8, 0, 0, 0.8, 0, 0, 0, 1.4, 0)
    indices = struct.pack("<3H", 0, 1, 2) + b"\x00\x00"
    binary = positions + indices
    document = {"asset": {"version": "2.0", "generator": "HunyForge demo"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}], "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.33, 0.52, 0.47, 1] if not textured else [0.78, 0.48, 0.18, 1], "metallicFactor": 0.05, "roughnessFactor": 0.82}}], "buffers": [{"byteLength": len(binary)}], "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962}, {"buffer": 0, "byteOffset": len(positions), "byteLength": 6, "target": 34963}], "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [-0.8, 0, 0], "max": [0.8, 1.4, 0]}, {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}]}
    encoded = json.dumps(document, separators=(",", ":")).encode(); encoded += b" " * ((4 - len(encoded) % 4) % 4)
    binary += b"\x00" * ((4 - len(binary) % 4) % 4)
    return struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(encoded) + 8 + len(binary)) + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded + struct.pack("<II", len(binary), 0x004E4942) + binary


def scene_to_mesh(loaded: trimesh.Scene | trimesh.Trimesh) -> trimesh.Trimesh:
    """Flatten a Trimesh scene without relying on version-specific Scene APIs."""
    if not isinstance(loaded, trimesh.Scene):
        return loaded
    if not loaded.geometry:
        raise ValueError("Generated artifact contains no mesh geometry")
    to_geometry = getattr(loaded, "to_geometry", None)
    if callable(to_geometry):
        flattened = to_geometry()
        if isinstance(flattened, trimesh.Trimesh):
            return flattened
        geometries = tuple(getattr(flattened, "geometry", {}).values())
        if geometries:
            return trimesh.util.concatenate(geometries)
    parts = []
    for node in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node]
        geometry = loaded.geometry.get(geometry_name)
        if geometry is None:
            continue
        part = geometry.copy()
        part.apply_transform(transform)
        parts.append(part)
    if not parts:
        raise ValueError("Generated artifact contains no mesh geometry")
    return trimesh.util.concatenate(parts)


def validate_glb(data: bytes) -> list[str]:
    if len(data) < 20 or data[:4] != b"glTF":
        return ["artifact-unreadable"]
    try:
        loaded = trimesh.load(BytesIO(data), file_type="glb", force="scene")
        mesh = scene_to_mesh(loaded)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
            return ["artifact-readable", "valid-glb-header", "empty-mesh"]
        if not all(math.isfinite(float(value)) for value in mesh.vertices.reshape(-1)):
            return ["artifact-readable", "valid-glb-header", "non-finite-vertices"]
        topology = "mesh-topology-review" if not mesh.is_watertight else "mesh-topology-valid"
        return ["artifact-readable", "valid-glb-header", "non-empty-mesh", "finite-vertices", topology, "scale-known", "unity-manifest"]
    except Exception:
        return ["artifact-readable", "invalid-glb-geometry"]


def texture_quality_checks(data: bytes) -> list[str]:
    try:
        loaded = trimesh.load(BytesIO(data), file_type="glb", force="scene")
        mesh = scene_to_mesh(loaded)
        visual = getattr(mesh, "visual", None)
        uv = getattr(visual, "uv", None)
        material = getattr(visual, "material", None)
        return [
            "uv-coordinates-present" if uv is not None and len(uv) else "uv-coordinates-missing",
            "material-present" if material is not None else "material-missing",
        ]
    except Exception:
        return ["uv-coordinates-missing", "material-missing"]


def _transfer_source_visuals(source: trimesh.Trimesh, reduced: trimesh.Trimesh) -> trimesh.Trimesh | None:
    """Reattach the source mesh's UVs/material to a decimated mesh via nearest-vertex lookup."""
    try:
        uv = np.asarray(source.visual.uv)
        _, nearest = source.kdtree.query(reduced.vertices)
        reduced.visual = trimesh.visual.TextureVisuals(uv=uv[nearest], material=source.visual.material)
        return reduced
    except Exception:
        return None


def _prepare_rigged_unity_geometry(loaded: trimesh.Scene, data: bytes, settings: GenerationSettings, notes: list[str]) -> tuple[dict[str, bytes], list[str]]:
    """LOD/collision for a rigged scene: decimate the Chassis node only; wheel
    nodes pass through with their pivots so the hierarchy survives."""
    produced = {"unity-lod0.glb": data}
    chassis = loaded.geometry.get("Chassis")
    if not isinstance(chassis, trimesh.Trimesh):
        notes.append("rigged-scene-missing-chassis")
        return produced, notes
    if settings.generate_lods:
        lod1_target = max(4, int(settings.face_count) // 2)
        if chassis.faces.shape[0] > lod1_target:
            try:
                reduced = chassis.simplify_quadric_decimation(face_count=lod1_target)
                if isinstance(reduced, trimesh.Trimesh) and reduced.faces.shape[0] > 0:
                    source_visual = getattr(chassis, "visual", None)
                    if getattr(source_visual, "uv", None) is not None and getattr(reduced.visual, "uv", None) is None:
                        transferred = _transfer_source_visuals(chassis, reduced)
                        if transferred is not None:
                            reduced = transferred
                    scene = trimesh.Scene()
                    scene.add_geometry(reduced, node_name="Chassis", geom_name="Chassis", transform=np.eye(4))
                    for node in loaded.graph.nodes_geometry:
                        if node == "Chassis":
                            continue
                        transform, geom_name = loaded.graph.get(node)
                        scene.add_geometry(loaded.geometry[geom_name], node_name=node, geom_name=geom_name, transform=transform)
                    produced["unity-lod1.glb"] = bytes(scene.export(file_type="glb"))
                    notes.append(f"lod1-chassis-faces-{reduced.faces.shape[0]}-wheels-preserved")
                else:
                    produced["unity-lod1.glb"] = data
                    notes.append("lod1-decimator-empty-fallback-source")
            except Exception:
                produced["unity-lod1.glb"] = data
                notes.append("lod1-decimator-unavailable-fallback-source")
        else:
            produced["unity-lod1.glb"] = data
            notes.append(f"lod1-chassis-faces-{chassis.faces.shape[0]}-within-budget")
    if settings.generate_collision:
        collision_mesh = chassis.bounding_box if settings.collision_mode == "box" else chassis.convex_hull
        produced["unity-collision.glb"] = bytes(collision_mesh.export(file_type="glb"))
        notes.append(f"collision-chassis-{settings.collision_mode}")
    return produced, notes


def prepare_unity_geometry(data: bytes, settings: GenerationSettings, telemetry: StageTelemetry | None = None) -> tuple[dict[str, bytes], list[str]]:
    """Normalize a GLB and derive LOD/collision geometry through trimesh."""
    loaded = trimesh.load(BytesIO(data), file_type="glb", force="scene")
    if isinstance(loaded, trimesh.Scene) and "Chassis" in getattr(loaded, "geometry", {}) and len(loaded.geometry) > 1:
        return _prepare_rigged_unity_geometry(loaded, data, settings, ["rigged-scene"])
    mesh = scene_to_mesh(loaded)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
        raise ValueError("Generated artifact contains no usable triangle mesh")
    working = mesh.copy()
    working.remove_unreferenced_vertices()
    source_visual = getattr(mesh, "visual", None)
    source_uv = getattr(source_visual, "uv", None)
    source_has_pbr = source_uv is not None and len(source_uv) > 0 and getattr(source_visual, "material", None) is not None
    produced = {"unity-lod0.glb": data}
    notes = ["lod0-source-bytes-preserved"]
    if settings.generate_lods:
        with telemetry.measure("lod_generation") if telemetry is not None else nullcontext():
            target = max(4, int(settings.face_count))
            if working.faces.shape[0] > target:
                try:
                    reduced = working.simplify_quadric_decimation(face_count=target)
                    if not isinstance(reduced, trimesh.Trimesh) or reduced.faces.shape[0] == 0:
                        raise ValueError("decimator returned no geometry")
                    working = reduced
                    notes.append(f"lod-working-faces-{working.faces.shape[0]}-budget-{target}")
                    if working.faces.shape[0] > target:
                        notes.append("lod-budget-missed")
                except Exception:
                    notes.append(f"lod-decimator-unavailable-actual-{working.faces.shape[0]}-budget-{target}-missed")
            else:
                notes.append(f"lod-working-faces-{working.faces.shape[0]}-within-budget-{target}")
            lod1_target = max(4, min(int(settings.face_count), working.faces.shape[0]) // 2)
            if working.faces.shape[0] > lod1_target:
                try:
                    lod1 = working.simplify_quadric_decimation(face_count=lod1_target)
                    if not isinstance(lod1, trimesh.Trimesh) or lod1.faces.shape[0] == 0:
                        raise ValueError("decimator returned no geometry")
                    notes.append(f"lod1-faces-{lod1.faces.shape[0]}-budget-{lod1_target}")
                    if lod1.faces.shape[0] > lod1_target:
                        notes.append("lod1-budget-missed")
                except Exception:
                    lod1 = working
                    notes.append(f"lod1-fallback-working-actual-{working.faces.shape[0]}-budget-{lod1_target}-missed")
            else:
                lod1 = working
                notes.append(f"lod1-faces-{lod1.faces.shape[0]}-within-budget-{lod1_target}")
            lod1_visual = getattr(lod1, "visual", None)
            lod1_uv = getattr(lod1_visual, "uv", None)
            if source_has_pbr and (lod1_uv is None or len(lod1_uv) == 0):
                transferred = _transfer_source_visuals(mesh, lod1)
                if transferred is not None:
                    produced["unity-lod1.glb"] = bytes(transferred.export(file_type="glb"))
                    notes.append("lod1-uv-transferred-nearest-source")
                else:
                    produced["unity-lod1.glb"] = data
                    notes.append("lod1-fallback-source-preserve-pbr-no-uv-decimator")
            else:
                produced["unity-lod1.glb"] = bytes(lod1.export(file_type="glb"))
    if settings.generate_collision:
        with telemetry.measure("collision_generation") if telemetry is not None else nullcontext():
            if settings.collision_mode == "box":
                collision_mesh = working.bounding_box
                notes.append("collision-bounding-box")
            else:
                collision_mesh = working.convex_hull
                notes.append("collision-convex-hull")
            produced["unity-collision.glb"] = bytes(collision_mesh.export(file_type="glb"))
    return produced, notes


class HunyuanShapeAdapter(InferenceAdapter):
    def __init__(self, backend: str):
        self.name = backend
        self.url = os.getenv("HUNYUAN_API_URL", "http://127.0.0.1:8082")

    async def generate_shape(self, seed: int, parameters: dict, output: Path) -> None:
        image = parameters.get("image")
        if not image:
            raise RuntimeError("A reference image is required for real Hunyuan generation.")
        payload = {"job_id": parameters["job_id"], "stage": "shape", "seed": seed, "image": self._image_payload(image), "settings": {key: parameters[key] for key in GenerationSettings.model_fields}}
        await asyncio.to_thread(self._request, "/generate", payload, output)

    async def generate_textures(self, mesh: Path, parameters: dict, output: Path) -> None:
        image = parameters.get("image")
        if not image:
            raise RuntimeError("A reference image is required for real Hunyuan texturing.")
        payload = {"job_id": parameters["job_id"], "stage": "texture", "seed": parameters["seed"], "image": self._image_payload(image), "settings": {key: parameters[key] for key in GenerationSettings.model_fields}, "mesh_token": {"job_id": parameters["job_id"], "artifact": "white-mesh.glb", "sha256": sha256_file(mesh)}}
        await asyncio.to_thread(self._request, "/generate", payload, output)

    @staticmethod
    def _raw_base64(value: str) -> str:
        return value.split(",", 1)[1] if value.startswith("data:") and "," in value else value

    @classmethod
    def _image_payload(cls, value: str) -> str:
        # The official Hunyuan3D API decodes this field directly with base64.b64decode,
        # so data-URL prefixes must be removed for the local service.
        return cls._raw_base64(value) if os.getenv("HUNYUAN_STRIP_DATA_URL", "1") == "1" else value

    def _request(self, path: str, payload: dict, output: Path) -> None:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            timeout = int(os.getenv("HUNYUAN_REQUEST_TIMEOUT_SECONDS", "3600"))
            output.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with open(output, "wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
        except urllib.error.HTTPError as error:
            detail_body = error.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(detail_body)
                detail = parsed.get("detail") or parsed.get("text") or detail_body
            except json.JSONDecodeError:
                detail = detail_body
            raise RuntimeError(f"Hunyuan service returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError("Hunyuan worker is unavailable or still loading; wait for local runtime readiness and retry.") from error


class HunyuanMultiViewAdapter(HunyuanShapeAdapter):
    """Native HTTP bridge to the local Hunyuan3D-2mv worker."""

    name = "hunyuan3d-2mv"

    async def generate_shape(self, seed: int, parameters: dict, output: Path) -> None:
        references = parameters.get("reference_images") or []
        if not references:
            raise ValueError("Multi-view job has no persisted reference images")
        url = os.getenv("HUNYFORGE_MULTI_VIEW_URL", "").strip()
        if not url:
            raise RuntimeError("Multi-view generation requires HUNYFORGE_MULTI_VIEW_URL and the local Hunyuan3D-2mv worker.")
        payload = {"seed": seed, "references": [{"view": item["view"], "image": item["image"]} for item in references], "settings": {key: parameters[key] for key in GenerationSettings.model_fields}}
        await asyncio.to_thread(self._request_url, url.rstrip("/") + "/generate", payload, output)

    async def generate_textures(self, mesh: Path, parameters: dict, output: Path) -> None:
        # The 2.1 paint pipeline remains single-reference. The canonical front
        # reference is deliberately retained as image and recorded in provenance.
        await super().generate_textures(mesh, parameters, output)

    def _request_url(self, url: str, payload: dict, output: Path) -> None:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(request, timeout=int(os.getenv("HUNYFORGE_MULTI_VIEW_TIMEOUT_SECONDS", "1800"))) as response:
                with open(output, "wb") as handle:
                    shutil.copyfileobj(response, handle, 1024 * 1024)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Hunyuan3D-2mv worker returned HTTP {error.code}: {detail[-500:]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError("Hunyuan3D-2mv worker is unavailable or still loading.") from error


class HunyuanOmniAdapter(InferenceAdapter):
    """Run an explicitly configured local wrapper around the official Omni CLI."""
    name = "hunyuan3d-omni"

    async def generate_shape(self, seed: int, parameters: dict, output: Path) -> None:
        command = os.getenv("HUNYUAN_OMNI_COMMAND")
        if not command:
            raise RuntimeError("Hunyuan3D-Omni uses the official inference.py CLI; set HUNYUAN_OMNI_COMMAND to a local wrapper before selecting this backend.")
        await asyncio.to_thread(self._run, command, seed, parameters, output)

    async def generate_textures(self, mesh: Path, parameters: dict, output: Path) -> None:
        raise RuntimeError("Hunyuan3D-Omni does not provide the 2.1 texture API; use the Hunyuan3D 2.1 texture stage after configuring a handoff.")

    def _run(self, command: str, seed: int, parameters: dict, output: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="hunyforge-omni-") as directory:
            root = Path(directory)
            image_path = root / "reference-image.png"
            control_path = root / "control-data"
            output_path = root / "output.glb"
            image_value = HunyuanShapeAdapter._raw_base64(parameters["image"])
            try:
                image_path.write_bytes(base64.b64decode(image_value, validate=True))
            except Exception as error:
                raise ValueError("Omni reference image is not valid base64 data") from error
            control_value = parameters.get("control_data", "")
            control_path.write_text(HunyuanShapeAdapter._raw_base64(control_value) if control_value.startswith("data:") else control_value, encoding="utf-8")
            values = {"image": str(image_path), "control_type": parameters.get("control_type", ""), "control_data": str(control_path), "output": str(output_path), "seed": str(seed)}
            args = [part.format(**values) for part in shlex.split(command)]
            completed = subprocess.run(args, capture_output=True, text=True, timeout=180, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"Omni runner failed ({completed.returncode}): {completed.stderr[-500:]}")
            if not output_path.is_file():
                raise RuntimeError(f"Omni runner completed without producing the configured output GLB: {output_path}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "rb") as source, open(output, "wb") as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)


class Pipeline:
    def __init__(self, store: JobStore, adapter: InferenceAdapter | None = None):
        self.store = store
        self.adapter_override = adapter
        self.running: set[str] = set()
        self.active: dict[str, JobStatus] = {}
        self.cancel_requests: set[str] = set()
        self._reservation_lock = Lock()
        self._job_monotonic: dict[str, float] = {}
        self._stage_monotonic: dict[str, float] = {}

    def reserve(self, job: JobStatus) -> bool:
        key = str(job.id)
        with self._reservation_lock:
            held = set(self.active) | self.running
            if held and key not in held:
                return False
            self.active[key] = job
            self.running.add(key)
            return True

    def release(self, job: JobStatus) -> None:
        key = str(job.id)
        with self._reservation_lock:
            self.active.pop(key, None)
            self.running.discard(key)
            self.cancel_requests.discard(key)
            self._job_monotonic.pop(key, None)
            for held in [held for held in self._stage_monotonic if held.startswith(f"{key}:")]:
                self._stage_monotonic.pop(held, None)

    def is_busy(self) -> bool:
        return bool(self.running or self.active)

    def request_cancel(self, job_id: str) -> None:
        self.cancel_requests.add(job_id)
        job = self.active.get(job_id)
        if job is not None and job.stage not in TERMINAL_STAGES:
            job.stage = JobStage.CANCELLATION_REQUESTED
            self.store.save(job)

    def _cancel_pending(self, job: JobStatus) -> bool:
        return str(job.id) in self.cancel_requests

    def _mark_cancelling(self, job: JobStatus) -> None:
        if job.stage != JobStage.CANCELLING:
            job.stage = JobStage.CANCELLING
            self.store.save(job)

    def _adapter_for(self, job: JobStatus) -> InferenceAdapter:
        if self.adapter_override is not None:
            return self.adapter_override
        if job.backend == "demo":
            return DemoAdapter()
        if job.backend == "hunyuan3d-omni":
            return HunyuanOmniAdapter()
        if job.input_mode == "multi-image":
            return HunyuanMultiViewAdapter(job.backend)
        return HunyuanShapeAdapter(job.backend)

    def _touch_elapsed(self, job: JobStatus) -> None:
        base = self._job_monotonic.get(str(job.id))
        if base is not None:
            job.timings["elapsed_seconds_total"] = round(time.perf_counter() - base, 3)

    def _stage_elapsed(self, job: JobStatus, name: str, record: StageState, finished) -> None:
        base = self._stage_monotonic.get(f"{job.id}:{name}")
        if base is not None:
            record.elapsed_seconds = round(time.perf_counter() - base, 3)
        elif record.started_at is not None:
            record.elapsed_seconds = (finished - record.started_at).total_seconds()

    def _prepare(self, job: JobStatus) -> GenerationSettings:
        settings = resolve_generation_settings(job.preset, job.parameters, strict=False)
        job.runtime_config = job.runtime_config or current_runtime_config()
        job.model_revision = job.model_revision or current_model_revision()
        job.parameters.update(settings.model_dump())
        job.parameters["seed"] = job.seed
        job.parameters["job_id"] = str(job.id)
        if job.image is not None:
            job.parameters["image"] = job.image
        if job.reference_images:
            job.parameters["reference_images"] = [item.model_dump() for item in job.reference_images]
        if job.control_type is not None:
            job.parameters["control_type"] = job.control_type
        job.started_at = job.started_at or now()
        self._job_monotonic[str(job.id)] = time.perf_counter()
        start = PIPELINE_STAGES.index(job.resume_from) if job.resume_from else 0
        for stage in PIPELINE_STAGES[:start]:
            record = job.stage_status.setdefault(stage, StageState())
            if record.state in ("waiting", "skipped"):
                record.state = "skipped" if stage == "texture" and not job.texture else "complete"
        self.store.save(job)
        return settings

    def _stage_begin(self, job: JobStatus, name: str, stage: JobStage, progress: int, operation: str) -> None:
        record = job.stage_status.setdefault(name, StageState())
        record.state = "running"
        record.started_at = now()
        record.finished_at = None
        record.error = None
        self._stage_monotonic[f"{job.id}:{name}"] = time.perf_counter()
        job.stage = stage
        job.progress = progress
        job.current_operation = operation
        job.failed_stage = None
        self.store.save(job)

    def _stage_finish(self, job: JobStatus, name: str, state: str, error: str | None = None) -> None:
        record = job.stage_status.setdefault(name, StageState())
        finished = now()
        record.state = state
        record.finished_at = finished
        record.error = error
        self._stage_elapsed(job, name, record, finished)
        job.current_operation = None
        job.operation_progress = None
        self._touch_elapsed(job)
        self.store.save(job)

    def _finalize_cancel(self, job: JobStatus, stage_name: str) -> None:
        record = job.stage_status.get(stage_name)
        if record is not None and record.state in ("waiting", "running"):
            record.state = "cancelled"
            record.finished_at = now()
            self._stage_elapsed(job, stage_name, record, record.finished_at)
        job.stage = JobStage.CANCELLED
        job.current_operation = None
        job.operation_progress = None
        job.finished_at = now()
        self._collect_api_timings(job)
        self._touch_elapsed(job)
        self.store.save(job)

    def _fail(self, job: JobStatus, exc: Exception) -> None:
        message = str(exc)
        lowered = message.lower()
        if "out of memory" in lowered:
            code = "CUDA_OUT_OF_MEMORY"
        elif "unavailable" in lowered or "still loading" in lowered:
            code = "WORKER_UNAVAILABLE"
        else:
            code = "PIPELINE_ERROR"
        job.error_code = code
        job.error_message = f"{message} Retry or resume reuses any preserved stage checkpoints."
        job.stage = JobStage.FAILED
        job.current_operation = None
        job.operation_progress = None
        job.finished_at = now()
        self._collect_api_timings(job)
        self._touch_elapsed(job)
        self.store.save(job)

    def _poll_worker_telemetry(self, job: JobStatus, stage_name: str, sidecar: Path) -> bool:
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        changed = False
        for name, record in (data.get("stages") or {}).items():
            key = f"{stage_name}.{name}"
            if job.timings.get(key) != record:
                job.timings[key] = record
                changed = True
        operation = data.get("operation")
        if operation and job.current_operation != operation:
            job.current_operation = operation
            changed = True
        progress = data.get("progress")
        if progress != job.operation_progress:
            job.operation_progress = progress
            changed = True
        peaks = [record["allocator_peak_vram_mb"] for record in (data.get("stages") or {}).values() if record.get("allocator_peak_vram_mb")]
        if peaks:
            peak = int(round(max(peaks)))
            if job.peak_vram_mb is None or peak > job.peak_vram_mb:
                job.peak_vram_mb = peak
                changed = True
        return changed

    def _poll_api_telemetry(self, job: JobStatus, sidecar: Path) -> bool:
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        changed = False
        for name, record in (data.get("stages") or {}).items():
            key = f"api.{name}"
            if job.timings.get(key) != record:
                job.timings[key] = record
                changed = True
        operation = data.get("operation")
        if operation and job.current_operation != operation:
            job.current_operation = operation
            changed = True
        progress = data.get("progress")
        if progress != job.operation_progress:
            job.operation_progress = progress
            changed = True
        return changed

    def _collect_api_timings(self, job: JobStatus) -> None:
        self._poll_api_telemetry(job, self.store.path(job.id) / "telemetry" / "api.json")

    def _heartbeat(self, job: JobStatus, stage_name: str) -> None:
        telemetry_dir = self.store.path(job.id) / "telemetry"
        changed = self._poll_api_telemetry(job, telemetry_dir / "api.json")
        changed = self._poll_worker_telemetry(job, stage_name, telemetry_dir / f"{stage_name}.json") or changed
        record = job.stage_status.get(stage_name)
        base = self._stage_monotonic.get(f"{job.id}:{stage_name}")
        if record is not None and base is not None and record.state == "running":
            record.elapsed_seconds = round(time.perf_counter() - base, 3)
            changed = True
        if changed:
            self._touch_elapsed(job)
            self.store.save(job)

    async def _await_operation(self, awaitable, job: JobStatus, stage_name: str):
        task = asyncio.ensure_future(awaitable)
        try:
            while not task.done():
                self._heartbeat(job, stage_name)
                if self._cancel_pending(job):
                    self._mark_cancelling(job)
                await asyncio.wait({task}, timeout=0.5)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.wait({task}, timeout=0.5)
                except asyncio.CancelledError:
                    pass
            if not task.cancelled():
                task.exception()
            raise
        self._heartbeat(job, stage_name)
        return await task

    async def run(self, job: JobStatus) -> None:
        if not self.reserve(job):
            job.error_code = "PIPELINE_ERROR"
            job.error_message = "GPU worker reservation was lost to a concurrent job."
            job.stage = JobStage.FAILED
            job.finished_at = now()
            self.store.save(job)
            return
        try:
            if self._cancel_pending(job):
                self._finalize_cancel(job, "shape")
                return
            await self._run(job)
        finally:
            self.release(job)

    async def _run(self, job: JobStatus) -> None:
        try:
            settings = self._prepare(job)
            adapter = self._adapter_for(job)
            start = PIPELINE_STAGES.index(job.resume_from) if job.resume_from else 0
            telemetry_path = self.store.path(job.id) / "telemetry" / "api.json"
            with StageTelemetry(telemetry_path) as telemetry:
                if start == 0:
                    job.stage = JobStage.LOADING_SHAPE_MODEL
                    job.progress = 5
                    self.store.save(job)
                proceed = True
                if proceed and start <= 0:
                    proceed = await self._shape_stage(job, adapter, telemetry)
                if proceed and start <= 1:
                    proceed = await self._texture_stage(job, adapter, telemetry)
                if proceed and start <= 2:
                    proceed = await self._rig_stage(job, telemetry)
                if proceed and start <= 3:
                    proceed = await self._unity_stage(job, settings, telemetry)
                if proceed:
                    proceed = await self._validation_stage(job, settings, telemetry)
                if not proceed:
                    return
            if job.backend == "demo" and job.peak_vram_mb is None:
                job.peak_vram_mb = 0
            job.stage = JobStage.COMPLETE
            job.progress = 100
            job.finished_at = now()
            self._collect_api_timings(job)
            self._touch_elapsed(job)
            self.store.save(job)
        except Exception as exc:
            self._fail(job, exc)

    def _validate_glb_file(self, path: Path) -> list[str]:
        return validate_glb(path.read_bytes())

    def _texture_file_checks(self, job: JobStatus, staging: Path, mesh: Path) -> list[str]:
        data = staging.read_bytes()
        checks = validate_glb(data)
        if job.backend != "demo":
            checks += texture_quality_checks(data)
            if data == mesh.read_bytes():
                checks.append("untextured-fallback")
        return checks

    async def _shape_stage(self, job: JobStatus, adapter: InferenceAdapter, telemetry: StageTelemetry) -> bool:
        if self._cancel_pending(job):
            self._finalize_cancel(job, "shape")
            return False
        self._stage_begin(job, "shape", JobStage.GENERATING_SHAPE, 25, "shape_request")
        staging = self.store.path(job.id) / "staging" / "shape.glb"
        staging.parent.mkdir(parents=True, exist_ok=True)
        try:
            with telemetry.measure("shape_request"):
                await self._await_operation(adapter.generate_shape(job.seed, job.parameters, staging), job, "shape")
            checks = await self._await_operation(asyncio.to_thread(self._validate_glb_file, staging), job, "shape")
            if "non-empty-mesh" not in checks or "finite-vertices" not in checks:
                raise ValueError("Shape stage produced an invalid GLB artifact")
            self.store.commit_artifact(job, "white-mesh.glb", staging)
            job.shape_checkpoint = self.store.make_checkpoint(job, ["white-mesh.glb"])
        except Exception as exc:
            staging.unlink(missing_ok=True)
            if self._cancel_pending(job):
                self._stage_finish(job, "shape", "cancelled")
                self._finalize_cancel(job, "shape")
                return False
            job.failed_stage = "shape"
            self._stage_finish(job, "shape", "failed", str(exc))
            raise
        if self._cancel_pending(job):
            self._stage_finish(job, "shape", "complete")
            self._finalize_cancel(job, "shape")
            return False
        self._stage_finish(job, "shape", "complete")
        job.stage = JobStage.PROCESSING_MESH
        job.progress = 40
        self.store.save(job)
        return True

    async def _texture_stage(self, job: JobStatus, adapter: InferenceAdapter, telemetry: StageTelemetry) -> bool:
        if not job.texture:
            record = job.stage_status.setdefault("texture", StageState())
            record.state = "skipped"
            self.store.save(job)
            return True
        if self._cancel_pending(job):
            self._finalize_cancel(job, "texture")
            return False
        self._stage_begin(job, "texture", JobStage.GENERATING_TEXTURES, 72, "texture_request")
        staging = self.store.path(job.id) / "staging" / "texture.glb"
        staging.parent.mkdir(parents=True, exist_ok=True)
        mesh = self.store.path(job.id) / "artifacts" / "white-mesh.glb"
        try:
            with telemetry.measure("texture_request"):
                await self._await_operation(adapter.generate_textures(mesh, job.parameters, staging), job, "texture")
            checks = await self._await_operation(asyncio.to_thread(self._texture_file_checks, job, staging, mesh), job, "texture")
            if "non-empty-mesh" not in checks or "finite-vertices" not in checks:
                raise ValueError("Texture stage produced an invalid GLB artifact")
            if job.backend != "demo" and ("uv-coordinates-present" not in checks or "material-present" not in checks or "untextured-fallback" in checks):
                raise ValueError("Texture stage returned a mesh without usable UV coordinates and PBR material; retry the texture stage")
            self.store.commit_artifact(job, "textured-mesh.glb", staging)
            job.texture_checkpoint = self.store.make_checkpoint(job, ["textured-mesh.glb"])
        except Exception as exc:
            staging.unlink(missing_ok=True)
            if self._cancel_pending(job):
                self._stage_finish(job, "texture", "cancelled")
                self._finalize_cancel(job, "texture")
                return False
            job.failed_stage = "texture"
            self._stage_finish(job, "texture", "failed", str(exc))
            raise
        if self._cancel_pending(job):
            self._stage_finish(job, "texture", "complete")
            self._finalize_cancel(job, "texture")
            return False
        self._stage_finish(job, "texture", "complete")
        return True

    def _export_checks(self, job: JobStatus) -> tuple[list[str], list[str], bool]:
        directory = self.store.path(job.id) / "artifacts"
        white = directory / "white-mesh.glb"
        shape_checks = validate_glb(white.read_bytes()) if white.is_file() else ["artifact-unreadable"]
        textured = directory / "textured-mesh.glb"
        texture_checks: list[str] = []
        if textured.is_file():
            texture_checks = validate_glb(textured.read_bytes()) + texture_quality_checks(textured.read_bytes())
        ok = "non-empty-mesh" in shape_checks and "finite-vertices" in shape_checks
        if job.texture:
            ok = ok and "non-empty-mesh" in texture_checks and "finite-vertices" in texture_checks
            if job.backend != "demo":
                ok = ok and "uv-coordinates-present" in texture_checks and "material-present" in texture_checks
        return shape_checks, texture_checks, ok

    async def _rig_stage(self, job: JobStatus, telemetry: StageTelemetry) -> bool:
        record = job.stage_status.setdefault("rig", StageState())
        if not job.rig_spec or not job.rig_spec.wheels:
            record.state = "skipped"
            record.finished_at = now()
            self.store.save(job)
            return True
        if self._cancel_pending(job):
            self._finalize_cancel(job, "rig")
            return False
        self._stage_begin(job, "rig", JobStage.RIGGING, 76, "vehicle_rig")
        directory = self.store.path(job.id) / "artifacts"
        source = directory / "textured-mesh.glb" if (directory / "textured-mesh.glb").is_file() else directory / "white-mesh.glb"
        try:
            data = await self._await_operation(asyncio.to_thread(source.read_bytes), job, "rig")
            with telemetry.measure("rig"):
                glb, report = await self._await_operation(asyncio.to_thread(vehicle_rig.build_rigged_glb, data, job.rig_spec), job, "rig")
            self.store.add_artifact(job, "vehicle-rigged.glb", glb)
            self.store.add_artifact(job, "vehicle-rig-report.json", json.dumps({"source_mesh": source.name, **report}, indent=2).encode())
            job.rig_checkpoint = self.store.make_checkpoint(job, ["vehicle-rigged.glb", "vehicle-rig-report.json"])
        except Exception as exc:
            if self._cancel_pending(job):
                self._stage_finish(job, "rig", "cancelled")
                self._finalize_cancel(job, "rig")
                return False
            job.failed_stage = "rig"
            self._stage_finish(job, "rig", "failed", str(exc))
            raise
        if self._cancel_pending(job):
            self._stage_finish(job, "rig", "complete")
            self._finalize_cancel(job, "rig")
            return False
        self._stage_finish(job, "rig", "complete")
        return True

    async def _unity_stage(self, job: JobStatus, settings: GenerationSettings, telemetry: StageTelemetry) -> bool:
        if self._cancel_pending(job):
            self._finalize_cancel(job, "unity")
            return False
        self._stage_begin(job, "unity", JobStage.PREPARING_UNITY_EXPORT, 82, "unity")
        directory = self.store.path(job.id) / "artifacts"
        source_name = "vehicle-rigged.glb" if (directory / "vehicle-rigged.glb").is_file() else "textured-mesh.glb" if (directory / "textured-mesh.glb").is_file() else "white-mesh.glb"
        try:
            data = await self._await_operation(asyncio.to_thread((directory / source_name).read_bytes), job, "unity")
            with telemetry.measure("unity"):
                produced, notes = await self._await_operation(asyncio.to_thread(prepare_unity_geometry, data, settings, telemetry), job, "unity")
            for name, content in produced.items():
                self.store.add_artifact(job, name, content)
            shape_checks, texture_checks, unity_ready = await self._await_operation(asyncio.to_thread(self._export_checks, job), job, "unity")
            manifest = {
                "job_id": str(job.id),
                "parent_job_id": str(job.parent_job_id) if job.parent_job_id else None,
                "schema_version": job.schema_version,
                "backend": job.backend,
                "seed": job.seed,
                "preset": job.preset,
                "settings": {key: job.parameters.get(key) for key in GenerationSettings.model_fields},
                "model_revision": job.model_revision,
                "runtime_config": job.runtime_config,
                "unity_ready": unity_ready,
                "format": "glb",
                "triangle_budget": settings.face_count,
                "scale": 1.0,
                "axis": "Y-up",
                "source_mesh": source_name,
                "lods": [name for name in ("unity-lod0.glb", "unity-lod1.glb") if name in produced],
                "collision": "unity-collision.glb" if "unity-collision.glb" in produced else None,
                "memory_policy": "sequential-shape-texture",
                "geometry_notes": notes,
                "artifacts": {name: sha256_file(directory / name) for name in list(produced) + (["vehicle-rigged.glb"] if (directory / "vehicle-rigged.glb").is_file() else [])},
                "telemetry": self._collect_telemetry(job),
                "checks": {"shape": shape_checks, "texture": texture_checks},
            }
            if job.rig_spec:
                manifest["vehicle"] = {
                    "source_mesh": source_name,
                    "wheels": [wheel.model_dump() for wheel in job.rig_spec.wheels],
                    "chassis_node": "Chassis",
                    "units": "model-space (Y-up)",
                    "suggested": {"suspension_distance": 0.1, "wheel_mass": 20.0, "chassis_mass": 1200.0},
                }
            self.store.add_artifact(job, "hunyforge-manifest.json", json.dumps(manifest, indent=2).encode())
            job.unity_checkpoint = self.store.make_checkpoint(job, list(produced) + ["hunyforge-manifest.json"])
        except Exception as exc:
            if self._cancel_pending(job):
                self._stage_finish(job, "unity", "cancelled")
                self._finalize_cancel(job, "unity")
                return False
            job.failed_stage = "unity"
            self._stage_finish(job, "unity", "failed", str(exc))
            raise
        if self._cancel_pending(job):
            self._stage_finish(job, "unity", "complete")
            self._finalize_cancel(job, "unity")
            return False
        self._stage_finish(job, "unity", "complete")
        return True

    def _collect_telemetry(self, job: JobStatus) -> dict:
        collected = {}
        errors: list[str] = []
        for name in ("api", "shape", "texture"):
            sidecar = self.store.path(job.id) / "telemetry" / f"{name}.json"
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = None
            collected[name] = data
            if isinstance(data, dict):
                for error in data.get("telemetry_errors") or []:
                    errors.append(f"{name}: {error}")
        runtime = {}
        for name in ("shape", "texture"):
            sidecar = self.store.path(job.id) / "telemetry" / f"{name}-runtime.json"
            try:
                runtime[name] = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                runtime[name] = None
        collected["runtime"] = runtime
        collected["telemetry_errors"] = errors
        return collected

    def _build_validation_report(self, job: JobStatus, settings: GenerationSettings) -> tuple[dict, bool]:
        directory = self.store.path(job.id) / "artifacts"
        shape_checks, texture_checks, geometry_ok = self._export_checks(job)
        manifest = {}
        manifest_path = directory / "hunyforge-manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        export_checks: list[str] = []
        blocking: list[str] = []
        if "non-empty-mesh" not in shape_checks or "finite-vertices" not in shape_checks:
            blocking.append("shape-geometry")
        if job.texture:
            if "non-empty-mesh" not in texture_checks or "finite-vertices" not in texture_checks:
                blocking.append("texture-geometry")
            elif job.backend != "demo" and ("uv-coordinates-present" not in texture_checks or "material-present" not in texture_checks):
                blocking.append("texture-uv-material")
        expected_lods = ["unity-lod0.glb"] + (["unity-lod1.glb"] if settings.generate_lods else [])
        expected_collision = "unity-collision.glb" if settings.generate_collision else None
        expected = expected_lods + ([expected_collision] if expected_collision else []) + ["hunyforge-manifest.json"]
        if job.rig_spec:
            expected = ["vehicle-rigged.glb"] + expected
            rigged = directory / "vehicle-rigged.glb"
            if rigged.is_file():
                rig_checks, rig_blocking = vehicle_rig.rig_quality_checks(rigged.read_bytes(), job.rig_spec)
                export_checks += rig_checks
                blocking += rig_blocking
            else:
                blocking.append("rig-artifact-missing")
        if manifest_path.is_file():
            if manifest.get("lods") != expected_lods:
                blocking.append("manifest-lod-mismatch")
            if manifest.get("collision") != expected_collision:
                blocking.append("manifest-collision-mismatch")
            recorded = manifest.get("artifacts") or {}
            for name in expected_lods + ([expected_collision] if expected_collision else []):
                path = directory / name
                if name not in recorded:
                    blocking.append(f"manifest-hash-missing-{name}")
                elif path.is_file() and recorded[name] != sha256_file(path):
                    blocking.append(f"manifest-hash-mismatch-{name}")
        else:
            blocking.append("manifest-missing")
        for name in expected:
            path = directory / name
            if not path.is_file():
                export_checks.append(f"{name}-missing")
                blocking.append(f"export-{name}")
            elif name.endswith(".glb"):
                file_bytes = path.read_bytes()
                checks = validate_glb(file_bytes)
                if "non-empty-mesh" in checks and "finite-vertices" in checks:
                    export_checks.append(f"{name}-present")
                else:
                    export_checks.append(f"{name}-invalid")
                    blocking.append(f"export-{name}")
                if job.texture and job.backend != "demo" and name in expected_lods:
                    quality = texture_quality_checks(file_bytes)
                    if "uv-coordinates-present" not in quality or "material-present" not in quality:
                        blocking.append(f"export-{name}-uv-material")
            else:
                export_checks.append(f"{name}-present")
        warnings = []
        if job.backend == "demo":
            warnings.append({"severity": "info", "message": "Demo geometry is a test artifact; replace with Hunyuan output before production use."})
        for note in manifest.get("geometry_notes", []):
            if "missed" in note or "unavailable" in note or "fallback" in note:
                warnings.append({"severity": "warning", "message": note})
        telemetry = manifest.get("telemetry") or {}
        for error in telemetry.get("telemetry_errors") or []:
            warnings.append({"severity": "warning", "message": f"Telemetry persistence issue: {error}"})
        ok = geometry_ok and not blocking
        report = {
            "status": "passed" if ok else "failed",
            "blocking_failures": [{"check": name, "severity": "blocking"} for name in blocking],
            "warnings": warnings,
            "demo": job.backend == "demo",
            "attempt_job_id": str(job.id),
            "source_manifest_job_id": manifest.get("job_id"),
            "geometry_processing": manifest.get("geometry_notes", []),
            "memory_policy": "sequential-shape-texture",
            "checks": {"shape": shape_checks, "texture": texture_checks, "export": export_checks},
        }
        return report, ok

    def _write_package(self, job: JobStatus, staging: Path) -> None:
        directory = self.store.path(job.id) / "artifacts"
        staging.parent.mkdir(parents=True, exist_ok=True)
        names = ["white-mesh.glb", "textured-mesh.glb", "vehicle-rigged.glb", "vehicle-rig-report.json", "unity-lod0.glb", "unity-lod1.glb", "unity-collision.glb", "hunyforge-manifest.json", "validation-report.json"]
        rigged = job.rig_spec is not None and (directory / "vehicle-rigged.glb").is_file()
        setup_script = UNITY_DIR / "HunyForgeVehicleSetup.cs"
        with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                source = directory / name
                if source.is_file():
                    archive.write(source, f"Assets/HunyForge/{name}")
            if rigged and setup_script.is_file():
                archive.write(setup_script, "Assets/HunyForge/Editor/HunyForgeVehicleSetup.cs")
            archive.writestr("Assets/HunyForge/README.md", "Import the GLB into Unity. Review the validation report before production use.\n" + ("For rigged vehicles run HunyForge → Setup Vehicle Colliders after import.\n" if rigged else ""))

    async def _validation_stage(self, job: JobStatus, settings: GenerationSettings, telemetry: StageTelemetry) -> bool:
        if self._cancel_pending(job):
            self._finalize_cancel(job, "validation")
            return False
        self._stage_begin(job, "validation", JobStage.VALIDATING_UNITY_EXPORT, 94, "validation")
        staging = self.store.path(job.id) / "staging" / "unity-package.zip"
        try:
            with telemetry.measure("validation"):
                report, ok = await self._await_operation(asyncio.to_thread(self._build_validation_report, job, settings), job, "validation")
            self.store.add_artifact(job, "validation-report.json", json.dumps(report, indent=2).encode())
            if not ok:
                raise ValueError("Unity validation failed: " + "; ".join(entry["check"] for entry in report["blocking_failures"]))
            job.validation_checkpoint = self.store.make_checkpoint(job, ["validation-report.json"])
            if self._cancel_pending(job):
                self._finalize_cancel(job, "validation")
                return False
            with telemetry.measure("packaging"):
                await self._await_operation(asyncio.to_thread(self._write_package, job, staging), job, "validation")
            self.store.commit_artifact(job, "unity-package.zip", staging)
        except Exception as exc:
            staging.unlink(missing_ok=True)
            if self._cancel_pending(job):
                self._stage_finish(job, "validation", "cancelled")
                self._finalize_cancel(job, "validation")
                return False
            job.failed_stage = "validation"
            self._stage_finish(job, "validation", "failed", str(exc))
            raise
        if self._cancel_pending(job):
            self._stage_finish(job, "validation", "complete")
            self._finalize_cancel(job, "validation")
            return False
        self._stage_finish(job, "validation", "complete")
        return True
