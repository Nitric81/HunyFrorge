# HunyForge Performance & GUI Enhancement Plan

## Objective

Reduce full-generation time while making the UI accurately represent staged shape, texture, and Unity-export work.

Freeze and measure the current deployed real Hunyuan runtime before deploying performance changes. Historical 2K PBR evidence from AI_HANDOFF.md is a partial baseline, not an end-to-end result. Telemetry instrumentation and workspace implementation may proceed while a new public-API baseline runs against the unchanged deployed image; optimized runtime deployment remains gated on capturing that baseline. Draft/Standard/Final and optimization variants must subsequently be measured with explicit settings before performance or quality claims are accepted.

## Reconciliation with 2026-09-13 handoff

- The enhancement plan takes priority. Preserve local-only inference, the repaired supplied-mesh PBR path, Trimesh-compatible scene handling, persistent artifacts, worker readiness checks, and the 3,600-second per-call timeout until benchmark evidence justifies a change.
- LOD generation and collision generation become selectable; fast preparation may omit them. Validation, manifest, and packaging remain required for a job to reach complete. Shape/texture readiness is independent of export completion.
- Retry and Resume create a new auditable attempt from the earliest incomplete compatible checkpoint. Restart from shape is a separate explicit action. Historical failed jobs are never rewritten as successful.
- Use the plan's permitted interim lifecycle: one serialized worker with lazy model loading and explicit model-reference release between shape and texture. Health and logs distinguish the two lifecycle states; separate OS workers remain an architectural follow-up rather than a prerequisite to the interim implementation.
- Keep the known 1024-render/2048-texture, CPU-DINO, UniPC-15 profile as the high-quality reference. Draft and Standard settings are provisional until measured; do not label unmeasured durations, VRAM, or quality as verified.
- Forward and validate both shape and texture inference steps explicitly. The old API schema and GUI did not prove upstream forwarding; verify actual pipeline calls, not just adapter payloads.
- Baselines and optimized runs use the same source-image hash. Record actual seed behavior: the original worker silently ignored the requested shape seed and fixed the texture seed to zero, so timing comparisons are possible but strict same-seed quality equivalence is not.
- The worker currently catches texture exceptions and returns an untextured mesh; this must become a real failed texture stage preserving the shape checkpoint, not false success.
- Unity Editor import remains an explicit quality gate and must not be inferred from GLB headers or demo tests.

## Phase 1: Establish performance baseline

**Status:** Verified.

Measure each stage independently on the RTX 4080:

- Model startup and warm-up time
- Background removal
- Shape inference
- Mesh extraction
- Texture model loading
- Texture inference
- GLB conversion
- LOD generation
- Collision generation
- ZIP packaging
- Peak VRAM and system RAM per stage

Record the model revision, CUDA/PyTorch versions, input image dimensions, inference settings, texture/render resolutions, total elapsed time, and peak VRAM.

Add structured timing and memory telemetry to the backend before optimizing.

Acceptance criteria:

- Every generation stage has a measurable duration. **Met** — worker-side telemetry records `background_removal`, `shape_model_loading`, `shape_inference`, `mesh_extraction`, `glb_conversion`, `texture_model_loading`, `dino_features`, `multiview_model_N`, `super_model_N`, `bake`, `inpaint`, `texture_inference`, plus API-side `unity`, `validation`, and `packaging` spans.
- Shape and texture peak VRAM are recorded separately. **Met** — per-operation `allocator_peak_vram_mb` plus sampled whole-device peaks; job-level `peak_vram_mb` kept distinct from device-sampled peaks.
- A baseline report exists for Draft, Standard, and Final configurations. **Met** — original-runtime end-to-end baseline (job `e0a72b1a`, 3279 s) plus measured optimized runs for all three presets under `data/benchmarks/`.

## Phase 2: Create explicit generation presets

**Status:** Verified — `backend/generation-presets.json` is the shared contract; resolved parameters are stored on the job and recorded in the manifest. Draft/Standard/Final now carry measured duration and peak-VRAM figures (182 s/8945 MB, 177 s/8940 MB, 251 s/11149 MB on the default runtime); quality remains provisional pending Unity/visual review.

Define shared backend/frontend presets:

### Draft

- Lower inference steps
- Octree resolution: 256
- Lower texture/render resolution
- Optional texture generation
- Fast preview-oriented output

### Standard

- Balanced shape quality
- Medium texture resolution
- Normal LOD/collision preparation

### Final

- Highest configured quality
- 2048 texture output
- Full validation and Unity packaging

Presets must resolve into explicit parameters rather than relying on upstream defaults.

Acceptance criteria:

- Presets are defined in one shared contract. **Met.**
- The selected preset is stored with the job. **Met** — `job.preset` + resolved `job.parameters`.
- The generated manifest records all resolved settings. **Met** — `hunyforge-manifest.json`.
- No performance-critical parameter remains silently defaulted by Hunyuan. **Met** — steps, guidance, octree, views, texture/render size, face count, seed, DINO device all flow explicitly; verified by container contract tests.

## Phase 3: Forward real inference parameters

**Status:** Verified — eight upstream contract tests pass inside the container against the patched `/opt/hunyuan` source; production telemetry confirms real effect (`super_model_0..7` = 4 views for Draft/Standard, `super_model_0..11` = 6 views for Final).

Update the HunyForge request contract to support:

- `num_inference_steps`
- `guidance_scale`
- `octree_resolution`
- `num_chunks`
- `remove_background`
- `render_size`
- `texture_size`
- `max_num_view`
- `face_count`
- `generate_lods`
- `generate_collision`

Forward these values through:

```text
GUI → HunyForge API → Hunyuan adapter → Hunyuan worker → actual pipeline
```

Acceptance criteria:

- API tests confirm every parameter reaches the adapter. **Met.**
- Worker tests confirm every parameter reaches the upstream pipeline. **Met** — contract tests assert configured steps, seed, progress callback, DINO device, and view count in real upstream call sites.
- The generated job JSON contains the resolved values. **Met.**
- UI settings are no longer decorative or disconnected. **Met.**

## Phase 4: Separate shape and texture model lifecycles

**Status:** Verified (interim single-worker design per reconciliation) — lazy per-stage load with explicit release; a shape-only job never loads the texture model (`texture.texture_model_loading` absent, job `cc1b8ef8`). Separate OS processes remain a documented follow-up.

Prefer two long-lived workers:

```text
HunyForge API
 ├── Shape worker
 └── Texture worker
```

The shape worker owns only shape inference. The texture worker owns only PBR texturing.

If separate processes are not practical initially, implement explicit model unload/offload between stages as an interim solution.

Acceptance criteria:

- Shape inference can run without the texture model loaded. **Met.**
- Texture inference can run from a saved shape artifact. **Met** — mesh token (job + `white-mesh.glb` + SHA-256) rehydrates texture work.
- Peak VRAM is measured independently. **Met** — per-operation allocator peaks.
- The worker does not rely only on `torch.cuda.empty_cache()` for memory release. **Met** — model references released in `finally` blocks per stage.

## Phase 5: Optimize texture generation

**Status:** Verified — all variants measured on the target GPU; defaults set to CUDA DINO + FlashVDM on, compile off.

Benchmark:

- DINO on CPU versus CUDA
- FlashVDM enabled versus disabled
- Torch compile enabled versus disabled
- Six views versus fewer views
- Texture resolution presets
- Render resolution presets
- Texture inference step counts

Use the fastest configuration that meets the quality target for each preset.

Acceptance criteria:

- Benchmark results are recorded. **Met** — `data/benchmarks/` per-variant artifacts.
- Each preset has documented expected quality and approximate duration. **Met for performance** — quality remains provisional pending Unity/visual review.
- DINO placement is configurable. **Met** — `HUNYUAN_DINO_DEVICE`.
- FlashVDM and compile options are explicit runtime settings. **Met** — `HUNYUAN_FLASHVDM`, `HUNYUAN_COMPILE` (env-overridable in Compose).

## Phase 6: Reduce transfer and serialization overhead

**Status:** Verified — texture requests carry a job-scoped mesh token (job ID + artifact name + SHA-256) and the worker reads the persisted `white-mesh.glb`; no base64 GLB round trip.

Replace the current full-GLB base64 round trip for texturing.

Preferred design:

1. Save the white mesh to the job directory.
2. Send a secure job-scoped file token or path to the texture worker.
3. Let the texture worker read the shared artifact.
4. Return the textured artifact path or stream.

Fallback design:

- Use multipart upload instead of embedding the GLB in JSON.

Acceptance criteria:

- No unnecessary base64 encoding of large GLB meshes. **Met.**
- Large mesh generation does not cause excessive API memory spikes. **Met** — worker streams artifacts to the job directory; API serves file paths.
- The texture worker can consume a persisted shape checkpoint directly. **Met** — verified by resume/texture-retry tests and token validation.

## Phase 7: Optimize Unity preparation

**Status:** Verified — LOD/collision are selectable, validated when requested, and timed; `white-mesh.glb` is served before texture/export completes; packaging runs last. Unity import verification caught and fixed a real defect: LOD1 duplicated source bytes when quadric decimation dropped UVs; decimated meshes now receive nearest-vertex UV+material transfer (`lod1-uv-transferred-nearest-source`), verified reduced in-editor (11933 vs 26557 verts).

Make CPU post-processing configurable and measurable.

Changes:

- Generate LODs from a reduced working mesh.
- Make convex-hull collision generation optional.
- Add a fast collision mode.
- Avoid re-importing/re-exporting unchanged geometry.
- Run packaging after the main result is available.
- Move nonessential export work to a background stage.

Acceptance criteria:

- Shape and texture artifacts become available before expensive export work finishes. **Met** — `artifact_readiness` + partial preview/download.
- Collision and LOD timings are recorded. **Met.**
- Users can select fast or full Unity preparation. **Met** — `generate_lods` / `generate_collision` toggles.
- A failed collision/export step does not invalidate the generated mesh. **Met** — requested outputs are validated; failure marks that stage, preserves prior artifacts.

## Phase 8: Redesign generation GUI

**Status:** Verified — preset selector, shape-only toggle, advanced disclosure, and all listed controls are wired to the request payload; upload validation and measured duration/VRAM hints included.

Update the Generate panel with:

- Quality preset selector
- Shape-only / Shape + textures selector
- Advanced settings disclosure
- Texture resolution
- Render resolution
- Inference steps
- Octree resolution
- Guidance scale
- Background-removal toggle
- LOD/collision options
- Estimated VRAM and duration

The existing Steps field must become a controlled input and must be connected to the request payload.

Acceptance criteria:

- All visible settings affect the actual job. **Met.**
- Invalid combinations are blocked before submission. **Met.**
- Omni automatically disables incompatible texture options. **Met** — texture options disabled for shape-only.
- Low-VRAM warnings change based on the selected preset. **Met** — preset-driven measured VRAM figures.

## Phase 9: Redesign job activity and partial results

**Status:** Verified — stage-aware status, per-stage/total elapsed, current operation + operation progress, peak VRAM, white-mesh preview/download during texturing, texture-only retry, export-only retry, and tri-state connection reporting (`live`/`degraded`/`down`) with durable degraded warning.

Replace the single linear progress interpretation with stage-aware status:

```text
Shape generation       Running / Complete / Failed
Texture generation     Waiting / Running / Complete / Failed
Unity preparation      Waiting / Running / Complete / Failed
Validation             Waiting / Running / Complete / Failed
```

Add:

- Per-stage elapsed time
- Total elapsed time
- Current operation description
- Peak VRAM when available
- Estimated remaining time
- Shape preview as soon as shape generation completes
- White-mesh download before texture completion
- Texture-only retry
- Export-only retry

Acceptance criteria:

- Users can preview and download the shape before PBR finishes. **Met** — verified in browser.
- Texture failure preserves the usable shape artifact. **Met** — texture failure is a real stage failure; white mesh retained.
- Retry does not rerun completed stages. **Met** — earliest-incomplete-stage resume with checkpoint validation.
- Progress remains correct after SSE reconnects or polling fallback. **Met** — stale-event rejection, 10 s poll timeout, degraded indicator persists while polling healthy.

## Phase 10: Update job and artifact contracts

**Status:** Verified — `shape_checkpoint`/`texture_checkpoint`/`unity_checkpoint`/`validation_checkpoint`, `artifact_readiness`, `available_actions`, `failed_stage`/`resume_stage`, and immutable job history all implemented.

Add explicit checkpoint metadata:

```text
shape_checkpoint
texture_checkpoint
unity_checkpoint
validation_checkpoint
```

Track artifact readiness separately:

- White mesh
- Textured mesh
- LOD0
- LOD1
- Collision mesh
- Manifest
- Validation report
- Unity package

Update retry behavior so it selects the earliest incomplete stage rather than restarting the entire pipeline.

Acceptance criteria:

- Restarted services preserve completed checkpoints. **Met** — verified by an unplanned Docker engine restart during a live job; the job failed cleanly with `resume_stage: shape`, all 8 completed jobs retained artifacts, and a restart child job completed.
- Texture-only failures resume from the white mesh. **Met** — mesh-token checkpoint validation.
- Unity preparation failures resume without model inference. **Met** — `prepare_resume` copies validated checkpoints; tests cover it.
- The UI can determine which actions are available from job state. **Met** — `available_actions`.

## Phase 11: Improve cancellation and error handling

**Status:** Verified — `cancellation_requested`/`cancelling`/`cancelled` states, CUDA OOM → 503, validation → 422, generic → 500; errors name the failed stage, preserved artifacts, and retry guidance.

Add explicit cancellation states:

```text
cancellation_requested
cancelling
cancelled
```

Display actionable errors:

- What failed
- Which stage failed
- Whether an artifact was preserved
- What can be retried
- Whether retry will reuse existing work

Acceptance criteria:

- The UI does not claim immediate cancellation before the worker confirms it. **Met.**
- CUDA/model errors are mapped to understandable messages. **Met.**
- Partial artifacts remain accessible after recoverable failures. **Met.**

## Phase 12: Docker and runtime improvements

**Status:** Verified for the interim single-worker architecture — `HUNYUAN_DINO_DEVICE`/`HUNYUAN_FLASHVDM`/`HUNYUAN_COMPILE`/`HUNYUAN_CPU_THREADS` env-configurable, pinned source (`82920d64`) and model (`0b946776`) revisions, persistent `hunyforge-data` volume, health endpoint reports `runtime_config` + `pipeline_revision` + readiness, distinct API/worker telemetry. Separate OS workers remain a follow-up, not a plan blocker.

Update Docker and Compose to support:

- Separate shape and texture workers
- Health checks for each worker
- Explicit FlashVDM/compile configuration
- Configurable CPU thread counts
- Configurable DINO device
- Pinned Hunyuan source revision
- Persistent model/cache directories
- Separate runtime logs for API, shape, and texture workers
- GPU memory telemetry

Acceptance criteria:

- Container startup reports readiness for every worker. **Met** — `/health` + worker `/health` (503 until ready).
- A worker failure does not silently appear as generic API inactivity. **Met** — readiness gating + error codes.
- Rebuilds use a pinned source revision. **Met.**
- Runtime configuration is visible in the health endpoint. **Met.**

## Phase 13: Validation

**Status:** Verified — automated, browser, GPU, recovery, and headless Unity Editor import evidence recorded. Remaining caveat: formal visual-quality equivalence across presets is not claimed (single-source-image evidence); LOD1 UVs are nearest-vertex approximations.

### Automated tests

- Preset resolution
- Request parameter forwarding
- Stage transitions
- Checkpoint reuse
- Texture-only retry
- Export-only retry
- Cancellation states
- Artifact availability
- Invalid setting combinations

### Browser tests

- Select each quality preset
- Submit shape-only generation
- Submit textured generation
- Observe partial shape result
- Retry texture generation
- Cancel during each stage
- Recover from SSE disconnect
- Verify mobile layout and keyboard access

### GPU tests

- Draft, Standard, and Final generation — **done** (181.9 s / 176.6 s / 250.8 s).
- CPU versus CUDA DINO — **done** (151.6 s → 0.55 s feature extraction).
- FlashVDM and compile variants — **done** (FlashVDM: mesh extraction 9.6 s → 1.4 s, default on; compile: 323 s vs 182 s, default off).
- Shape-only and full generation — **done** (shape-only job `cc1b8ef8` completed in 112 s, texture stage skipped, texture model never loaded).
- Texture failure recovery — **done** (unit/integration tests; texture failure is a real stage failure preserving the white mesh).
- Service restart recovery — **done** (unplanned Docker engine restart during a live job: clean failure state, intact history, successful restart child job).
- Peak VRAM and elapsed-time comparison — **done** (see evidence table).

## Final success criteria

The enhancement is complete when:

- Full generation time is measured and improved against the baseline. **Met** — 3279 s → 182–251 s depending on preset (≈87–92 % less elapsed time; quality differs per preset, so not like-for-like).
- Shape and texture stages are independently managed. **Met.**
- The GUI exposes meaningful quality/performance choices. **Met.**
- Every visible generation setting reaches the actual inference pipeline. **Met** — contract-verified.
- Shape results are available before texture/export completion. **Met.**
- Failed texture or export stages can be retried independently. **Met.**
- The UI accurately reports progress, cancellation, partial results, and errors. **Met.**
- Docker supports reproducible, observable, low-VRAM operation. **Met.**
- Performance, GPU, browser, and recovery evidence is recorded. **Met** — with the standing caveat that formal visual-quality equivalence across presets is not claimed; evidence is single-source-image runs, and Unity Editor import remains the user-owned gate.

## Execution evidence

- Original-runtime public API baseline job: `e0a72b1a-39a2-478b-95b0-d8910311ce46` — status: complete; warm-start original runtime; job timestamps span 3279.388917 seconds (created 14:28:01.304564Z, updated 15:22:40.693481Z); observed poll showed the shape stage at 487 seconds; fine-grain CPU timings were missing, so this is an end-to-end span, not a single PBR-only duration
- Source job ID: `6b3910fa-2e78-437c-bded-7031d9e2b984`
- Source revision: `82920d643c0dc2f7bfd7255f45f62d386edfe60c` / model revision `0b94677654c57bb9a6b6845cd7b704ccf551d327`
- Benchmark script: `scripts/benchmark_hunyforge.py`; summarizer: `scripts/summarize_benchmarks.py`
- Benchmark output: `data/benchmarks/pre-enhancement-public-api.json`

### Optimized-runtime benchmark evidence (RTX 4080, `dino=cuda, flashvdm=1, compile=0` unless noted)

| Run | Artifact | Total | Shape inf | Texture inf | DINO | Mesh extract | Views | Peak alloc VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Draft, CPU DINO, forced 6 views | `enhanced-draft-cpu-dino.json` | 429.8 s | 6.0 s | 201.6 s | 151.1 s | 10.3 s | 6 | 11048 MB |
| Draft, CPU DINO, 4 views | `enhanced-draft-four-views-cpu-dino.json` | 479.7 s | 8.1 s | 172.0 s | 151.6 s | 9.8 s | 4 | 8952 MB |
| Draft, CUDA DINO, 4 views | `enhanced-draft-cuda-dino.json` | 198.7 s | 6.4 s | 19.9 s | 0.55 s | 9.6 s | 4 | 8948 MB |
| Draft, CUDA DINO + FlashVDM | `enhanced-draft-flashvdm.json` | 181.9 s | 6.2 s | 19.7 s | 0.68 s | 1.4 s | 4 | 8945 MB |
| Draft, CUDA DINO + FlashVDM + compile | `enhanced-draft-compile.json` | 323.1 s | 38.5 s | 125.2 s | 0.75 s | 8.0 s | 4 | 8948 MB |
| Standard | `enhanced-standard-optimal.json` | 176.6 s | 15.4 s | 26.6 s | 0.84 s | 2.6 s | 4 | 8940 MB |
| Standard, post-LOD-UV fix | `enhanced-standard-lodfix.json` / job `79d76ae1` | 222.8 s | — | — | — | — | 4 | 8933 MB |
| Final | `enhanced-final-optimal.json` | 250.8 s | 23.3 s | 65.2 s | 0.59 s | 2.3 s | 6 | 11149 MB |
| Draft shape-only (post-restart recovery) | job `cc1b8ef8-446c-42b8-a8f9-d3f086a2d130` | 112.1 s | — | skipped | — | — | — | 7858 MB |

Notes:

- View count verified from production telemetry: 8 `super_model_*` calls = 4 views; 12 = 6 views. The upstream `range(6)` bug was patched in `docker/configure_runtime.py`.
- Allocator peaks are PyTorch `max_memory_allocated`; whole-device sampled peaks are higher (e.g. Final ≈ 16015 MiB) because they include desktop/other GPU usage. Both are recorded; claims cite allocator peaks.
- Totals include cold model-load variance (shape model loading ranged 68–101 s across runs); per-stage inference deltas are the reliable comparison.
- Compile loses because lazy per-stage model release never amortizes ~140 s of warmup.
- Service-restart recovery verified organically: Docker engine restarted mid-shape on job `0041fefe`; job recorded `PIPELINE_ERROR` with `resume_stage=shape`, no partial artifacts lost, history intact, restart child `cc1b8ef8` completed. The `enhanced-draft-shape-only.json` artifact is the observer snapshot of that crashed job (status `failed`); the successful shape-only result is job `cc1b8ef8` recorded via the API.
- The successful shape-only run produced `white-mesh.glb`, `unity-lod0.glb`, manifest, validation report, and `unity-package.zip`; texture stage recorded as `skipped` and `texture.texture_model_loading` is absent — the texture model was never loaded.
- Unity import verified headless (Unity 6000.5.8f1, glTFast 6.20.0, `unity-smoke-test/`): 6/6 checks pass — all GLBs import, textured mesh has UVs/materials, LOD1 genuinely reduced (11933 vs 26557 verts, UVs retained via nearest-vertex transfer), collision imports. The first pass caught the LOD1-duplicates-LOD0 defect; fixed via `_transfer_source_visuals` in `backend/pipeline.py` (note `lod1-uv-transferred-nearest-source` in job `79d76ae1` manifest).

### Test evidence

- Backend: 97 tests pass, 4 skipped (host); upstream contract suite: 8/8 pass inside the container against patched `/opt/hunyuan`.
- Frontend: 36 tests pass; `vite build` succeeds (large `ThreePreview` chunk warning only).
- Browser: main workflows verified via Playwright; mobile a11y defects fixed and re-verified (`inert` sidebar, focus movement, 44 px targets, Escape).
