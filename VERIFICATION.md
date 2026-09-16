# HunyForge Verification Record

**Updated:** 2026-09-14
**Target hardware:** NVIDIA RTX 4080 (16 GB VRAM), 32 GB RAM, Windows + Docker Desktop (NVIDIA GPU support)
**Runtime under test:** source `82920d643c0dc2f7bfd7255f45f62d386edfe60c`, model `0b94677654c57bb9a6b6845cd7b704ccf551d327`, pipeline revision `staged-v2-view-limit-20260913`

## Automated tests

| Suite | Result | Evidence |
|---|---|---|
| Backend (`python -m unittest`) | 136 passed, 10 skipped | Host run; includes 11 vehicle-rig tests and 19 T2I/text-mode tests; upstream contract cases skip on host because `HUNYUAN_ROOT` is unset |
| Upstream contract tests (in container) | 8 passed | Run inside the `hunyforge` container against patched `/opt/hunyuan`; covers view-limit fix (4 vs 6 views), configured steps/seed/progress callback, DINO device + CPU offload |
| Frontend (`npm test`) | 45 passed | Presets, safe artifact URLs, SSE/polling fallback, text-mode + retexture + vehicle-rig flows, mobile inert/focus, upload validation, partial preview |
| Frontend build (`npm run build`) | pass | Only warning: ~644 kB `ThreePreview` chunk (dynamic import; non-blocking) |

## Text-to-3D and text-guided retexture (FLUX.2-klein-4B, verified 2026-09-14)

- `POST /api/reference-preview` → FLUX loads in ~1.7 s warm, inference ~95 s cold / ~28 s warm edit-mode, peak ~9.5 GB VRAM, released after use. Server-side prompt scaffolding applied.
- Real text-to-3D job `d9ba5f4c` ("off-road buggy" prompt → FLUX reference → full pipeline, Draft preset): all 8 artifacts produced, `input_mode=text`, T2I provenance recorded.
- Dependency bump verified: diffusers 0.30.0→0.38.0, transformers 4.46→4.51.3, deepspeed removed (broken/unused under torch 2.5.1), `trust_remote_code=True` added for vendored hunyuanpaintpbr pipeline. 106/106 in-container tests + a real shape+texture regression job pass on the new deps.

## Vehicle rigging (verified 2026-09-14)

- `POST /api/jobs/d9ba5f4c/wheel-suggest` on the real buggy: ground strip removed 364 faces (textured LOD is 20k faces), suggested 4 wheels on the correct X axle — FL/FR steering flagged, radii 0.16–0.18, centers symmetric (±0.45 X, ±0.29/−0.33 Z).
- Rig child `8928f6fc` (input_mode `vehicle-rig`, parent `d9ba5f4c`): **complete**, all 5 stages green. Artifacts: `vehicle-rigged.glb` (Chassis + 4 wheel nodes, pivots verified on GLB reload), `vehicle-rig-report.json`, regenerated LOD0/LOD1/collision, manifest with `vehicle` block, `unity-package.zip` with `Editor/HunyForgeVehicleSetup.cs`.
- **Defects caught by the real run and fixed:** boundary-loop walker could spin forever on junction vertices (now consumes one edge per step); ground-strip dropped TextureVisuals (now UV-preserving); cap vertices lacked UVs → degenerate export (cap centers inherit mean-loop UV); axis scorer preferred a thin slab over the true disk (now penalized by axial-normal fraction, 0.22 vs 0.58 measured).
- **Recovery:** pre-fix failed child `efe65245` resumes at `validation` with preserved checkpoints and correctly re-fails on its stale rigged artifact — checkpoint immutability behaving as designed.
- Headless Unity 6000.5.8f1 smoke test on the rigged artifact: **10 checks, 0 failures** — hierarchy/pivots/spin-not-orbit pass; packaged setup script creates 4/4 WheelColliders + Rigidbody (mass 1200) with manifest-consistent radii.


## Real-GPU benchmark evidence (`data/benchmarks/`)

| Run | Job / artifact | Total | Shape inf | Texture inf | DINO | Mesh extract | Views | Peak alloc VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original runtime baseline | `e0a72b1a` / `pre-enhancement-public-api.json` | 3279.4 s | ~487 s (observed) | — | — | — | — | — |
| Draft, CPU DINO, forced 6 views | `enhanced-draft-cpu-dino.json` | 429.8 s | 6.0 s | 201.6 s | 151.1 s | 10.3 s | 6 | 11048 MB |
| Draft, CPU DINO, 4 views | `enhanced-draft-four-views-cpu-dino.json` | 479.7 s | 8.1 s | 172.0 s | 151.6 s | 9.8 s | 4 | 8952 MB |
| Draft, CUDA DINO | `enhanced-draft-cuda-dino.json` | 198.7 s | 6.4 s | 19.9 s | 0.55 s | 9.6 s | 4 | 8948 MB |
| Draft, CUDA DINO + FlashVDM | `enhanced-draft-flashvdm.json` | 181.9 s | 6.2 s | 19.7 s | 0.68 s | 1.4 s | 4 | 8945 MB |
| Draft, CUDA DINO + FlashVDM + compile | `enhanced-draft-compile.json` | 323.1 s | 38.5 s | 125.2 s | 0.75 s | 8.0 s | 4 | 8948 MB |
| Standard (optimal defaults) | `enhanced-standard-optimal.json` | 176.6 s | 15.4 s | 26.6 s | 0.84 s | 2.6 s | 4 | 8940 MB |
| Final (optimal defaults) | `enhanced-final-optimal.json` / job `504cd240` | 250.8 s | 23.3 s | 65.2 s | 0.59 s | 2.3 s | 6 | 11149 MB |
| Draft shape-only (post-restart recovery) | job `cc1b8ef8-446c-42b8-a8f9-d3f086a2d130` | 112.1 s | — | skipped | — | — | — | 7858 MB |

Interpretation notes:

- View count is proven by `super_model_*` call count in worker telemetry (2 calls per view): 8 calls = 4 views, 12 calls = 6 views.
- Whole-device sampled peaks exceed allocator peaks (Final ≈ 16015 MiB) because sampling includes desktop/other GPU usage; performance claims cite allocator peaks.
- Totals include cold model-load variance (shape load 68–101 s). Stage-level deltas are the reliable comparison.
- `torch.compile` pays ~140 s warmup per job that lazy per-stage release never amortizes — kept off by default.
- CUDA DINO offload returns memory to CPU immediately after extraction, so allocator peak is unchanged vs CPU DINO.
- `enhanced-draft-shape-only.json` is the observer snapshot of the crashed job (`status: failed`); the successful shape-only result is job `cc1b8ef8` recorded via the API.

## Recovery evidence

- **Service restart:** Docker engine restarted mid-shape on job `0041fefe`. Result: `stage=failed`, `error_code=PIPELINE_ERROR` ("Remote end closed connection without response"), `resume_stage=shape`, `available_actions=["restart"]`, no checkpoints (correct — no artifact had been committed). All 8 completed jobs retained artifacts. Restart child `cc1b8ef8` completed in 112 s.
- **Shape-only:** texture stage recorded `skipped`; `texture.texture_model_loading` absent from timings — texture model never loaded. Artifacts: `white-mesh.glb`, `unity-lod0.glb`, manifest, validation report, `unity-package.zip`.

## Unity Editor import (verified)

Headless Unity 6000.5.8f1 batchmode run against `unity-smoke-test/` (isolated project, embedded glTFast 6.20.0 + deps) on artifacts from job `79d76ae1` — report `unity-smoke-test/smoke-test-report.json`, **0 failures**:

- All five GLBs import as prefabs via glTFast: `white-mesh` (272619 verts), `textured-mesh` (26557, UVs+materials), `unity-lod0` (26557), `unity-lod1` (**11933 verts — real ~55% reduction, UVs+materials retained**), `unity-collision` (8 verts, box mode).
- LOD reduction check passes: lod0=26557, lod1=11933.

### LOD defect found and fixed during this pass

The first smoke test caught `unity-lod1.glb` identical to LOD0 (same SHA-256): the quadric decimator strips UVs, and the old code fell back to copying source bytes when the source had PBR. Fix in `backend/pipeline.py`: nearest-vertex UV+material transfer from the source mesh onto the decimated mesh (`_transfer_source_visuals`), keeping source-byte fallback only if transfer itself fails. Manifest now records `lod1-uv-transferred-nearest-source`. Note: `simplify_quadric_decimation` requires the optional `fast_simplification` package (present in the runtime image; absent on the dev host — tests mock the decimator).

## Browser verification

- Main workflows verified via Playwright against the Vite dev server + demo API: preset selection, generation submission, stage-aware activity, partial white-mesh preview, artifact download.
- Mobile accessibility verified at 390 px viewport: closed sidebar is `inert` and `visibility:hidden`; open sidebar makes `main` inert; focus moves between menu toggle and close button; Escape closes; close target is 44 px.
- Stale persisted job selection recovers (clears `hunyforge.selectedJob`, falls back); the 404 pair observed in dev is React StrictMode double-mount, not a defect.

## Standing caveats

- Visual-quality equivalence across presets is not claimed; all evidence is single-source-image runs (job `6b3910fa` image). LOD1 UVs are nearest-vertex transfers — adequate for distance rendering, not pixel-exact.
- The original-runtime baseline lacks worker-side stage telemetry (instrumented only in the enhanced runtime), so 3279 s is an end-to-end span.
- Unity verification covered GLB import/geometry/materials/LOD/collision — not gameplay integration or a full `.unitypackage` import flow.
