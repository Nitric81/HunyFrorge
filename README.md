# HunyForge

HunyForge is a local-first 3D asset workbench: generate or reconstruct an asset, prepare it for Unity, inspect the result, and keep every job, artifact, and reference image on your machine.

It combines a React/Three.js UI with a FastAPI job service and local CUDA inference. The supported local workflows include image-to-3D, native multi-view reconstruction, text-to-3D, transparent 2D sprite creation, text-guided retexturing, Unity packaging, and vehicle wheel rigging.

## What it does

- Generate shape and PBR-textured GLB assets from an image or text prompt.
- Reconstruct a single asset from a labeled multi-view set using Hunyuan3D-2mv.
- Keep long-running jobs durable, with stage checkpoints, cancel/retry/resume, SSE progress, and polling fallback.
- Produce engine-ready packages with LODs, collision geometry, materials, validation reports, and manifests — Unity by default, Unreal Engine as a per-job opt-in.
- Detect and rig vehicle wheels into a GLB with Unity `WheelCollider` setup metadata.
- Create transparent RGBA PNG sprites from a local text prompt or an uploaded PNG/JPEG; HunyForge removes the background, centers the subject with configurable padding, and offers direct Unity-ready downloads. Sprite Studio also packs uploaded animation/directional frames into a sheet plus Unity `Multiple`-sprite metadata, and supports 128×64 or 256×128 2:1 isometric tile canvases.
- Run locally: input images, models, job data, and artifacts are not sent to a cloud inference service.

## Requirements

### Hardware

- **NVIDIA GPU with CUDA support.** The tested target is an RTX 4080 (16 GB VRAM); shape and texture run as separate stages to fit that budget. Smaller GPUs may work at lower presets but are unvalidated. AMD/ROCm is not supported.
- **~24 GB+ system RAM.** The worker admission gate rejects a stage when less than `HUNYFORGE_MIN_AVAILABLE_MB` (default 12 GB) is free.
- **Disk:** plan for tens of GB of model snapshots (the 2.1 weights alone are ~15 GB; FLUX and 2mv add more) plus a Docker volume for job artifacts.

### Software

- Windows 11 with Docker Desktop on WSL2 and the NVIDIA Container Toolkit configured for GPU workloads. This is the verified path; native Linux with the NVIDIA Container Toolkit should work but is untested.
- Node.js 22+ and Python 3.10+ for the developer workflow; Docker is the recommended real-inference workflow.
- Local model snapshots (download separately — see [Licenses](#licenses)):
  - Hunyuan3D-2.1 for image shape generation and PBR texturing.
  - Hunyuan3D-2mv Turbo for native multi-view shape generation (optional unless using Multi view).
  - FLUX.2-klein-4B for text-to-3D and reference-preview features (optional unless using Text).
  - Qwen-Image-Edit-2511 for reference-guided vehicle-frame editing (optional unless using Vehicle Set → Qwen Edit; approximately 54 GB).

### Engine targets (optional)

Engine installs are only needed to consume export packages; HunyForge runs without them.

- **Unity Editor** — imports `unity-package.zip` (GLB via the glTFast importer). Verified on Unity 6000.5.8f1 with glTFast 6.20.0; see `unity-smoke-test/`.
- **Unreal Engine 5.x** — imports `unreal-package.zip` (GLB via the Interchange pipeline). Verified on UE 5.8.2. The engine is auto-detected from the registry and `UE_*` install folders on any drive, or pass `-EnginePath`.
  - Required plugins are enabled automatically by the bootstrap script: Python Editor Script Plugin, Editor Scripting Utilities, and Interchange (glTF import on UE 5.8+).

## Security and privacy

HunyForge is a **single-user local tool**. The API has no authentication and no rate limiting; it is only safe because the stack binds to `127.0.0.1`.

- **Do not expose the service.** Do not publish port 8081/8082/8083 on `0.0.0.0`, a LAN interface, or behind a reverse proxy without adding your own authentication layer. Anyone who can reach the API can submit GPU jobs, create projects, and read or cancel jobs.
- **Local-first by default.** Input images, models, job data, and artifacts stay on your machine; `HF_HUB_OFFLINE=1` and Hugging Face telemetry is disabled in the container.
- **Report vulnerabilities privately** — see [SECURITY.md](SECURITY.md).

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
| `HUNYFORGE_STAGE_ISOLATION` | Run each stage in a fresh subprocess so the OS reclaims model RSS on exit | `1` |
| `HUNYFORGE_MIN_AVAILABLE_MB` | Reject a stage with `insufficient_memory` below this VM headroom | `12288` |
| `HUNYFORGE_JOB_MAX_AGE_DAYS` | Seed default: auto-delete completed jobs older than this | `0` (keep forever) |
| `HUNYFORGE_FAILED_JOB_MAX_AGE_DAYS` | Seed default: auto-delete failed/cancelled jobs older than this | `0` (keep forever) |
| `HUNYFORGE_DATA_MAX_GB` | Seed default: data-volume cap; oldest terminal jobs evicted first | `0` (unlimited) |

### Settings screen

Open **Settings** via the gear icon in the sidebar header. It covers:

- **Storage** — retention limits (completed-job age, failed/cancelled-job age, total disk cap), current disk usage, and a "Clean up now" sweep.
- **Generation defaults** — default quality preset and default Unreal-package opt-in.
- **Runtime** — memory admission threshold and stage isolation (applies to the next stage, not one in flight).
- **About** — version, inference mode, runtime/GPU status.

UI-editable values persist in `settings.json` under the data root and **win over the env seeds** above once saved; infra config (model paths, URLs, ports) remains env-only. Retention only ever deletes terminal (complete/failed/cancelled) jobs — a running job is never touched — and individual jobs can be deleted from the history list.

See [`.env.example`](.env.example) for the full supported configuration. The model directory mounted into Docker is read-only; job history and output artifacts are persisted in the `hunyforge-data` Docker volume.

## Working in the UI

1. Choose **Image** or **Text**, then select the asset type: Generic, Character, Vehicle, or Vegetation.
2. For image input, upload one image or select Multi view and add its required labeled references.
3. Select a quality preset. HunyForge resolves it into inference settings and records the result with the job.
4. Generate. Shape, texture, Unity preparation, validation, and vehicle rigging progress are streamed to the UI.
5. Inspect partial artifacts while a job runs. If a job is interrupted, use restart or resume; a child job retains its lineage and immutable source job.
6. For vehicle assets, mark/suggest wheels in the **Rig** tab and create a rigged child job. Auto-suggest begins with four conventional wheel markers; add and place markers for additional axles (up to 16 wheels). Download the Unity package when validation passes.
7. For vegetation assets (trees and large plants), the Unreal package additionally carries wind vertex colors (`COLOR_0`: R = sway weight, G = normalized height), a trunk-only collision capsule when `collision_mode` is `trunk`, and a FLUX-generated leaf-spray atlas (`T_Leaf_*.png`). The packaged setup script applies two-sided foliage materials with vertex-color-weighted wind.
8. For a 2D isometric asset, open **Sprites**. **Sprite** produces individual transparent props, buildings, vehicle angles, icons, masks, or overlays; **2:1 tile** produces a 128×64 or 256×128 canvas; **Sheet** packs uploaded animation/directional PNG frames into a transparent atlas and downloadable Unity metadata. **Vehicle set** requires eight ordered frames (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`) for intact, damaged, and wrecked states, with optional empty/loaded cargo states; it emits one combined atlas, an individual row PNG for every state, and bottom-pivot Unity mappings. Text sprites require the local FLUX worker; uploaded-image sprites use the local background-removal worker. PSD, Krita, and Aseprite remain the authoring sources: export their layers/frames to PNG before importing them into HunyForge.
   - In **Vehicle set**, the Qwen Edit section accepts an approved master vehicle image and a direction/state instruction. It produces a reference-guided candidate frame; inspect it, download it, then add it to the corresponding eight-frame state row. Qwen editing is intentionally one frame at a time so poor angle/state edits can be retried without discarding the rest of the set.

## Engine packages

Completed jobs produce engine-targeted ZIP packages as downloadable artifacts.

- `unity-package.zip` — extract into a Unity project's `Assets/` folder (or import the GLBs directly). Rigged vehicles include an editor script under **HunyForge > Setup Vehicle Colliders**.
- `unreal-package.zip` — opt in per job via the **Unreal package** toggle on the Generate form, then install into an Unreal project with the bootstrap script:

```powershell
.\scripts\setup-unreal.ps1 -ProjectPath D:\path\to\Game.uproject -PackageZip .\unreal-package.zip
```

The script resolves the UE install, enables the required plugins in the `.uproject` (backed up to `.uproject.hunyforge-bak`), runs the packaged Python setup headless — import, LOD chain, collision — and prints a JSON result report listing anything that remains manual. Run it with no arguments to probe an engine's capabilities without a project. For rigged vehicles the package preserves the `Chassis`/`Wheel_*` hierarchy and Chaos parameter hints in the manifest; assembling a Chaos vehicle pawn remains a documented manual step.

## Validation and checks

```powershell
npm run lint
npm test
npm run build
python -m compileall backend
python -m unittest discover -s backend -t .

# Requires a running API in demo mode.
.\scripts\verify-demo.ps1
```

The current implementation has real target-GPU evidence for text/image generation, PBR texturing, recovery, vehicle rigging, and headless Unity import. Formal visual-quality equivalence between presets and multi-image quality comparisons remain open work; do not infer them from a successful job alone.

## Project documentation

- [Product and technical design](docs/HUNYFORGE_DESIGN.md)
- [Verification ledger](docs/VERIFICATION.md)
- [Current delivery status](STATUS.md)

## Licenses

HunyForge's own source code is [Apache-2.0](LICENSE). **Model weights are not included** — you download each snapshot yourself, and each is governed by its own license. Review the model cards before use, especially for commercial use:

- **Hunyuan3D-2.1** and **Hunyuan3D-2mv** (`tencent/*` on Hugging Face) — Tencent Hunyuan Community License, which includes usage restrictions.
- **FLUX.2-klein-4B** (Black Forest Labs) — distributed under its own terms; check the model card.
- **RealESRGAN** (bundled in the Hunyuan3D texture pipeline) — BSD-3-Clause.

## Troubleshooting

- **Multi-view is unavailable:** start the `multiview` Compose profile and confirm `/health` reports `multi_view.enabled: true`.
- **A real job is rejected as not ready:** wait for `runtime_ready: true`; model initialization may take several minutes after a container rebuild.
- **GPU out of memory:** do not run competing GPU jobs. Keep the staged 1024 render / 2048 texture profile for the tested 16 GB target unless you have validated another profile.
- **`insufficient_memory` or `WORKER_OOM_KILLED` job errors:** stages run in isolated subprocesses so the failure is contained to that stage (the container stays up). Check `workers.hunyuan.memory` in `/health` for headroom, wait for it to recover, then resubmit.
- **Job connection is degraded:** the UI falls back from SSE to status polling. Refreshing the browser should recover persisted job state.
