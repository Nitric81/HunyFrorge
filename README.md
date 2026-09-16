# HunyForge

HunyForge is a local-first 3D asset workbench: generate or reconstruct an asset, prepare it for Unity, inspect the result, and keep every job, artifact, and reference image on your machine.

It combines a React/Three.js UI with a FastAPI job service and local CUDA inference. The supported local workflows include image-to-3D, native multi-view reconstruction, text-to-3D, text-guided retexturing, Unity packaging, and vehicle wheel rigging.

## What it does

- Generate shape and PBR-textured GLB assets from an image or text prompt.
- Reconstruct a single asset from a labeled multi-view set using Hunyuan3D-2mv.
- Keep long-running jobs durable, with stage checkpoints, cancel/retry/resume, SSE progress, and polling fallback.
- Produce Unity-ready packages with LODs, collision geometry, materials, validation reports, and manifests.
- Detect and rig vehicle wheels into a GLB with Unity `WheelCollider` setup metadata.
- Run locally: input images, models, job data, and artifacts are not sent to a cloud inference service.

## Requirements

- Windows with Docker Desktop and an NVIDIA GPU configured for Docker GPU workloads.
- NVIDIA RTX 4080 (16 GB VRAM) is the tested target. Shape and texture run as separate stages to fit that budget.
- Node.js and Python for the developer workflow; Docker is the recommended real-inference workflow.
- Local model snapshots:
  - Hunyuan3D-2.1 for image shape generation and PBR texturing.
  - Hunyuan3D-2mv Turbo for native multi-view shape generation (optional unless using Multi view).
  - FLUX.2-klein-4B for text-to-3D and reference-preview features (optional unless using Text).

## Quick start

Start the production-style local stack:

```powershell
docker compose up -d --build
```

Open [http://127.0.0.1:8081](http://127.0.0.1:8081), then wait until the service is ready:

```powershell
Invoke-RestMethod http://127.0.0.1:8081/health | ConvertTo-Json -Depth 5
```

`runtime_ready` must be `true` before submitting a real Hunyuan job. Stop the base stack with:

```powershell
docker compose down
```

### Developer/demo workflow

The demo backend exercises the job, artifact, Unity-preparation, and validation contracts without local model weights:

```powershell
.\scripts\start-hunyforge.ps1
```

Alternatively, run the API and Vite UI separately:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8081

npm install
npm run dev
```

The developer UI is served at [http://127.0.0.1:5173](http://127.0.0.1:5173). API documentation is at [http://127.0.0.1:8081/docs](http://127.0.0.1:8081/docs).

## Multi-view reconstruction

In **Generate → Image**, choose **Multi view**. Supply images of the same asset in these required views:

| Required | Optional |
|---|---|
| Front, Rear, Left, Right | Front-left 3/4, Rear-right 3/4, Top, Underside |

HunyForge sends the complete labeled set in one request to a dedicated native Hunyuan3D-2mv worker. This is important: the stock Hunyuan3D-2.1 image endpoint can batch inputs but does not reconstruct one fused mesh from separate viewpoints.

Download the `tencent/Hunyuan3D-2mv` model into the directory configured by `HUNYUAN2MV_MODEL_HOST_PATH`, then start the optional worker:

```powershell
docker compose --profile multiview up -d --build
```

Confirm the integration is ready:

```powershell
Invoke-RestMethod http://127.0.0.1:8081/health |
  Select-Object -ExpandProperty multi_view
```

It should report `enabled: true`. The multi-view worker creates the shape; the existing 2.1 PBR stage uses the Front image as its texture reference. That limitation is recorded in job provenance.

## Configuration

Copy `.env.example` to a local `.env` file or set equivalent environment variables before starting Docker. Do not commit local paths, model directories, or credentials.

| Variable | Used for | Default in Compose |
|---|---|---|
| `HUNYFORGE_DEMO` | Demo contract backend instead of real inference | `0` |
| `HUNYUAN_MODEL_HOST_PATH` | Host directory containing Hunyuan3D-2.1 weights | Local model directory |
| `HUNYFORGE_T2I_MODEL_HOST_PATH` | Host directory containing the local FLUX model | Local model directory |
| `HUNYUAN_RENDER_SIZE` | Shape/PBR render resolution | `1024` |
| `HUNYUAN_TEXTURE_SIZE` | Texture resolution | `2048` |
| `HUNYUAN_DINO_DEVICE` | DINO device selection | `cuda` |
| `HUNYUAN_FLASHVDM` | Enable FlashVDM where supported | `1` |
| `HUNYFORGE_MULTI_VIEW_URL` | Internal URL for the 2mv worker | `http://hunyforge-multiview:8083` |
| `HUNYUAN2MV_MODEL_HOST_PATH` | Host directory containing Hunyuan3D-2mv weights | Local model directory |
| `HUNYUAN2MV_SUBFOLDER` | 2mv checkpoint variant | `hunyuan3d-dit-v2-mv-turbo` |

See [`.env.example`](.env.example) for the full supported configuration. The model directory mounted into Docker is read-only; job history and output artifacts are persisted in the `hunyforge-data` Docker volume.

## Working in the UI

1. Choose **Image** or **Text**, then select the asset type: Generic, Character, or Vehicle.
2. For image input, upload one image or select Multi view and add its required labeled references.
3. Select a quality preset. HunyForge resolves it into inference settings and records the result with the job.
4. Generate. Shape, texture, Unity preparation, validation, and vehicle rigging progress are streamed to the UI.
5. Inspect partial artifacts while a job runs. If a job is interrupted, use restart or resume; a child job retains its lineage and immutable source job.
6. For vehicle assets, mark/suggest wheels in the **Rig** tab and create a rigged child job. Auto-suggest begins with four conventional wheel markers; add and place markers for additional axles (up to 16 wheels). Download the Unity package when validation passes.

## Validation and checks

```powershell
npm run lint
npm test
npm run build
python -m compileall backend
python -m unittest backend.test_pipeline backend.test_api

# Requires a running API in demo mode.
.\scripts\verify-demo.ps1
```

The current implementation has real target-GPU evidence for text/image generation, PBR texturing, recovery, vehicle rigging, and headless Unity import. Formal visual-quality equivalence between presets and multi-image quality comparisons remain open work; do not infer them from a successful job alone.

## Project documentation

- [Product and technical design](docs/HUNYFORGE_DESIGN.md)
- [Verification ledger](docs/VERIFICATION.md)
- [Current delivery status](STATUS.md)

## Troubleshooting

- **Multi-view is unavailable:** start the `multiview` Compose profile and confirm `/health` reports `multi_view.enabled: true`.
- **A real job is rejected as not ready:** wait for `runtime_ready: true`; model initialization may take several minutes after a container rebuild.
- **GPU out of memory:** do not run competing GPU jobs. Keep the staged 1024 render / 2048 texture profile for the tested 16 GB target unless you have validated another profile.
- **Job connection is degraded:** the UI falls back from SSE to status polling. Refreshing the browser should recover persisted job state.
