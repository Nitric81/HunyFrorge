# HunyForge AI Handoff

**Updated:** 2026-09-13  
**Workspace:** `D:\Projects\Software Dev\HunyForge`  
**Target:** Windows PC, Docker Desktop, NVIDIA RTX 4080 16 GB VRAM, 32 GB system RAM  
**Current state:** Dockerized local Hunyuan3D 2.1 runtime is deployed and healthy. Shape and PBR inference have both executed successfully on the target GPU, but a fresh end-to-end HunyForge job through the UI/API still needs to be completed and verified.

## Start here

1. Read this file.
2. Read `STATUS.md`, `docs/HUNYFORGE_DESIGN.md`, and `PERFORMANCE_AND_GUI_ENHANCEMENT_PLAN.md` for the broader product contract. Treat this handoff as newer than their 2026-09-11 verification claims.
3. Preserve unrelated workspace files. This directory is not currently a Git repository, so there is no commit or clean working-tree baseline.
4. Do not start another GPU job until `nvidia-smi` confirms the prior worker is idle.
5. Continue with the fresh end-to-end validation described under **Next work**.

## Product requirements that must not regress

- All inference and project data remain local; no cloud inference or telemetry.
- RTX 4080 16 GB VRAM and 32 GB RAM are the baseline machine.
- Shape and texture execution are staged to fit the 16 GB GPU.
- Unity preparation and validation are mandatory for every completed textured or shape-only asset. They are not optional export extras.
- The app must remain usable through both the visual UI and the local HTTP API.
- Completed artifacts must be preserved and jobs must eventually support checkpoint-aware stage resume.

## What was accomplished in this session

### Docker/runtime recovery

- Built and deployed a self-contained CUDA 12.4 Docker image containing:
  - the React/Vite production UI;
  - the FastAPI HunyForge API on port `8081`;
  - the official Hunyuan3D 2.1 API worker on loopback port `8082`;
  - PyTorch 2.5.1 CUDA 12.4 dependencies;
  - compiled Hunyuan PBR rasterizer and mesh-painter extensions;
  - the host model snapshot mounted read-only from `D:\AI_Models\Hunyuan3D-2.1`;
  - model/download caches mounted on `D:` so large cache data does not consume the nearly-full `C:` drive.
- Reorganized expensive dependency installation into cacheable Docker layers.
- Excluded the unavailable `bpy==4.0` CPython 3.10 Linux wheel and made the unused Blender import optional for the runtime path.
- The active container is named `hunyforge`; the image/Compose service is `hunyforge-hunyforge`.

### Corrected staged shape-to-PBR requests

The recurring UI error was:

```text
Hunyuan service returned HTTP 404: **NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**
```

This was not external network traffic. The official local API used that generic message to hide internal errors.

Implemented fixes:

- Added an optional base64 `mesh` field to the upstream `GenerationRequest` model during image construction.
- Changed the PBR request path to consume the supplied shape GLB instead of silently regenerating shape.
- Replaced unsupported `trimesh.Scene.to_geometry()` calls with `trimesh.util.concatenate(tuple(scene.geometry.values()))` in both the patched Hunyuan worker and HunyForge's own Unity/GLB processing.
- Added a clear failure for empty scenes.
- Changed the first upstream `ValueError` response from the misleading generic HTTP 404 to the actual local exception text with HTTP 500.
- Added a regression test for scene flattening on the installed Trimesh version.

### Runtime/UI resilience completed earlier in the same workstream

- HunyForge health now distinguishes model files, GPU availability, worker readiness, and overall runtime readiness.
- Job submission rejects a still-loading worker with an actionable HTTP 503.
- The frontend uses the persisted job as authority and reconnects/falls back to status checks when its event stream disconnects.
- Container startup marks interrupted non-terminal jobs failed rather than leaving them permanently displayed as running.
- Request timeout is configurable and currently set to 3,600 seconds per Hunyuan service call.
- Uvicorn startup messages written to stderr are classified as informational output by the entrypoint instead of being labeled as application errors.

### RTX 4080 PBR profile

The upstream defaults were effectively a 2K render and 4K output texture. They were changed to runtime-configurable values with these current Compose defaults:

```text
HUNYUAN_RENDER_SIZE=1024
HUNYUAN_TEXTURE_SIZE=2048
HUNYUAN_REQUEST_TIMEOUT_SECONDS=3600
```

The installed multiview pipeline uses `UniPCMultistepScheduler` at 15 inference steps. It was already using the upstream fast step count; no step-count reduction was made.

## Verification evidence

### Automated checks

These passed after the source changes and before the current deployment:

```powershell
python -m unittest backend.test_pipeline backend.test_api
npm run build
```

- Backend result: 16 tests passed.
- Frontend production build passed.
- Vite emitted only the existing warning that the Three.js bundle chunk exceeds 500 kB.
- Python compilation and a direct Trimesh scene-flattening smoke test passed.

### Real target-GPU inference

Shape generation had already completed successfully in prior jobs, producing a valid saved white mesh. This session focused on proving the repaired PBR handoff without spending another shape run.

#### Prior 4K PBR diagnostic

- Supplied saved white mesh: `/data/jobs/6b3910fa-2e78-437c-bded-7031d9e2b984/artifacts/white-mesh.glb`
- Worker logged `Using supplied mesh for texture generation`.
- Texture generation took about 3,608 seconds.
- The server completed with HTTP 200 and a textured GLB about 11 seconds after the direct client's 3,600-second timeout.
- Conclusion: the corrected mesh handoff worked, but the original 4K profile was too slow for the timeout and impractical as the RTX 4080 default.

#### Current 2K PBR diagnostic — passed

- Same saved white mesh and source image were sent directly to the internal local Hunyuan `/generate` endpoint with `texture=true`.
- Worker logged `Using supplied mesh for texture generation`.
- Texture stage duration: `2606.1705775260925` seconds (43 minutes 26 seconds).
- Total server duration: `2608.2741706371307` seconds.
- Server response: HTTP 200.
- Output: `/opt/hunyuan/gradio_cache/7e186cd9-f1e5-4138-9fba-49178204e73c_textured.glb` inside the current container.
- Size: 2,189,244 bytes.
- First four bytes: `67 6c 54 46` (`glTF`), confirming a binary GLB container.
- Observed GPU behavior during inference: approximately 15.6 GB VRAM in use, 100% GPU utilization, and roughly 49–51 C.
- The diagnostic output is in the container's non-volume `gradio_cache`; it will disappear when the container is recreated. It is evidence, not a persisted HunyForge job artifact.

### Current live runtime

At handoff, `GET http://127.0.0.1:8081/health` returned:

```json
{
  "status": "ok",
  "service": "hunyforge-api",
  "inference_mode": "hunyuan",
  "gpu_available": true,
  "model_paths_configured": true,
  "model_snapshot_present": true,
  "model_snapshot_path": "/models/Hunyuan3D-2.1",
  "hunyuan_service_ready": true,
  "runtime_ready": true
}
```

## Important current-state distinction

The successful 2K PBR diagnostic called the internal Hunyuan worker directly. It did **not** update the persisted HunyForge job.

Persisted job `6b3910fa-2e78-437c-bded-7031d9e2b984` therefore correctly remains:

```text
stage: failed
progress: 72
error_code: PIPELINE_ERROR
error_message: timed out
artifacts: artifacts/white-mesh.glb
```

Do not alter that historical record to claim success. The UI's existing **Retry** action creates a new job and currently restarts the entire pipeline from shape; it does not resume at PBR.

## Files changed in this workstream

Primary files changed during diagnosis and repair:

- `backend/pipeline.py` — staged Hunyuan request handling, long timeout, error mapping, Trimesh-compatible scene flattening, artifact validation, Unity preparation.
- `backend/test_pipeline.py` — regression coverage including the Trimesh scene compatibility case.
- `docker/hunyuan-local-models.patch` — local model paths, low-VRAM behavior, supplied-mesh PBR path, Trimesh compatibility, configurable render/texture sizes, worker logging behavior.
- `Dockerfile` — cacheable CUDA dependency build, upstream request schema/error-response adjustments, patched Hunyuan source, runtime assembly.
- `docker-compose.yml` — GPU runtime, `D:` model/cache mounts, 3,600-second timeout, 1024 render/2048 texture defaults.
- `.env.example` — corresponding runtime settings.

Other files changed earlier in the same recovery workstream and requiring preservation:

- `backend/models.py`
- `backend/main.py`
- `backend/storage.py`
- `src/App.tsx`

Because there is no Git metadata, inspect files and timestamps rather than assuming a diff can be reconstructed.

## Next work

### P0 — fresh full HunyForge workflow

Run one new textured Hunyuan3D 2.1 generation through the public HunyForge UI or `POST /api/jobs`, not directly against port 8082. Allow roughly 50–60 minutes based on measured shape plus 2K PBR time.

Verify all of the following before calling the app ready for complete workflows:

1. The UI shows the authoritative job progressing through shape, PBR, Unity preparation, validation, and complete.
2. Temporary SSE/network interruption does not mark a still-running job failed; polling recovers the true state.
3. The persisted job reaches `complete` rather than timing out.
4. `/data/jobs/<new-job-id>/artifacts/` contains the white mesh, textured GLB, Unity LOD/collision artifacts, manifest, validation report, and downloadable Unity package ZIP.
5. The final textured artifact has a valid `glTF` header and loads in the Three.js viewer.
6. Restarting the browser preserves and redisplays the job from backend state.
7. Restarting the container preserves completed job metadata/artifacts in the `hunyforge-data` volume.

Do not run competing jobs during this test. The app's normal public API has a single-GPU busy lock, but direct worker calls bypass it.

### P0 — Unity Editor proof

Unity preparation is implemented and mandatory in the backend, but the generated package has not yet been imported into a real Unity Editor project. Import a completed package and verify:

- scale, axes, and pivot;
- PBR material and texture assignment;
- LOD0/LOD1 switching;
- collision mesh behavior;
- manifest and validation-report accuracy.

Do not call Unity export production-ready until this editor test passes.

### P1 — checkpoint-aware history and resume

The current retry endpoint restarts from the beginning. Implement the existing design TODO in `docs/HUNYFORGE_DESIGN.md` and `PERFORMANCE_AND_GUI_ENHANCEMENT_PLAN.md`:

- persist immutable checkpoints after shape, mesh processing, PBR, Unity preparation, and validation;
- verify checkpoint integrity and pipeline/model compatibility;
- add `POST /api/jobs/{id}/resume`;
- resume from the latest valid checkpoint;
- support texture-only and export-only recovery;
- preserve the old job and create an auditable resumed attempt;
- make the GUI clearly distinguish **Resume from PBR/Unity** from **Restart from shape**;
- recover durable history after browser or container restart.

The saved white mesh in failed job `6b3910fa-2e78-437c-bded-7031d9e2b984` is a useful real fixture for implementing texture-only resume.

### P1 — progress and performance

- Add worker-side callbacks or structured phase logging so the UI can show meaningful progress within the 43-minute PBR stage instead of appearing frozen.
- Capture peak VRAM programmatically in the job record; current real observations were manual and `peak_vram_mb` remains unset on old jobs.
- Profile PBR substages before reducing quality further. UniPC is already at 15 steps.
- Consider selectable quality presets while keeping 1024 render/2048 texture as the tested 16 GB default. Any faster preset needs its own visual-quality and Unity validation evidence.
- Keep per-stage timeouts distinct from UI polling timeouts. A transient UI request timeout must not cancel or overwrite a still-running backend job.

### P1/P2 — remaining integrations and release gates

- Complete native Hunyuan3D-Omni execution for `point`, `voxel`, `pose`, and `bbox`, including an explicit Omni-shape-to-2.1-PBR handoff.
- Add OpenAPI examples/automation documentation for non-GUI workflows.
- Add browser/E2E coverage for reconnect, refresh, cancel, retry, resume, and artifact download.
- Reconcile `STATUS.md`, `context/project.md`, and `docs/VERIFICATION.md`; they still describe real Hunyuan execution as unverified and predate this session's GPU evidence.
- Initialize source control and make a baseline commit only if the user requests it; do not assume authorization or discard files.

## Exact resume commands

Run from `D:\Projects\Software Dev\HunyForge`.

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:8081/health | ConvertTo-Json -Depth 5
nvidia-smi
docker logs --tail 100 hunyforge
```

If the container is missing or an image rebuild is required:

```powershell
docker compose build
docker compose up -d --force-recreate
docker logs -f hunyforge
```

Model loading takes several minutes. Wait until `/health` reports `runtime_ready: true` before submitting a job.

Local test/build commands:

```powershell
python -m unittest backend.test_pipeline backend.test_api
npm test
npm run lint
npm run build
```

Open the app at `http://127.0.0.1:8081`. API documentation is available from FastAPI at `http://127.0.0.1:8081/docs`.

## Definition of ready for user testing

HunyForge should only be described as ready for complete 3D asset workflow testing after a new public-API/UI job has completed shape, PBR, mandatory Unity preparation, validation, persisted all deliverables, survived a browser refresh, and its package has passed a Unity Editor import smoke test. The repaired local model and PBR service are now proven; that complete application-level gate remains open.
