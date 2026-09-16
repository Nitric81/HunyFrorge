# HunyForge Status

**Updated:** 2026-09-14  
**Overall:** Performance & GUI plan verified; text-to-3D, text-guided retexture, and vehicle wheel-rigging all verified on target hardware  
**Current milestone:** Vehicle rigging complete — rigged GLB + WheelCollider auto-setup verified end-to-end in headless Unity  

## Current truth

- Implemented: the full performance and GUI scope — explicit Draft/Standard/Final presets resolved into real inference parameters, lazy per-stage model loading with release between shape and texture, stage-aware progress and operation telemetry, checkpointed retry/resume/restart, cancellation states, partial-artifact availability (white mesh preview/download during texturing), configurable LOD/collision with mandatory validation and packaging for completed jobs, mesh-token texture requests (no base64 GLB round trip), SSE with polling fallback and tri-state connection reporting (`live`/`degraded`/`down`), mobile drawer focus management, and Docker runtime env configuration.
- **Text features (FLUX.2-klein-4B):** `input_mode` text/image switch, server-side prompt scaffolding, `POST /api/reference-preview` (approve-before-burn), text-guided retexture as immutable child jobs, Edit tab with viewport capture → FLUX edit → child job, Before/After parent compare, T2I health block, lineage badges. Deps bumped: diffusers 0.38.0, transformers 4.51.3; deepspeed removed; `trust_remote_code` for vendored pipeline.
- **Vehicle rigging:** `asset_type` selector (Generic / Character placeholder / Vehicle); new `rig` pipeline stage (skipped for non-vehicle); `POST /wheel-suggest` (ground-strip + quadrant clustering + dual-axis cylinder fit with axial-normal scoring) and `POST /rig` → CPU-side child job partitioning wheels onto axle pivots; rigged GLB (`Chassis` + `Wheel_*`), manifest `vehicle` block, `HunyForgeVehicleSetup.cs` in the Unity package; viewer wheel markers with click-to-place + numeric editing; `RIG` badge, `Rigged` preview source.
- Runtime defaults (RTX 4080 verified): `HUNYUAN_DINO_DEVICE=cuda`, `HUNYUAN_FLASHVDM=1`, `HUNYUAN_COMPILE=0`, `HUNYUAN_CPU_THREADS=8`. Pinned source `82920d643c0dc2f7bfd7255f45f62d386edfe60c`, model `0b94677654c57bb9a6b6845cd7b704ccf551d327`.
- Verified on real GPU: Draft 181.9 s, Standard 176.6 s, Final 250.8 s (vs original-runtime baseline 3279 s). Text-to-3D buggy `d9ba5f4c` end-to-end; rig child `8928f6fc` complete with validation passed.
- Verified recovery: unplanned Docker restart mid-job → clean failure + restart child completes; pre-fix rigged child correctly re-fails on stale checkpoints (immutability preserved).
- Verified tests: backend 136 pass (11 vehicle-rig, 19 T2I), frontend 45 pass, `tsc`/Vite build clean.
- Verified Unity Editor import: headless 6000.5.8f1 + glTFast smoke test — **0 failures** including rigged-vehicle checks (node hierarchy, pivots, spin-not-orbit, 4/4 WheelColliders + Rigidbody via the packaged script).
- Unverified: formal visual-quality equivalence across presets; LOD1 UVs are nearest-vertex approximations; wheel-partition cap seams are visible up close (documented).
- Planned TODO: separate OS-level shape/texture workers; character rigging (selector placeholder present); multi-image quality comparisons.
- Blocked: none.

## Milestones

| ID | Name | Status | Exit evidence |
|---|---|---|---|
| E01 | Foundation | complete | API/UI vertical slice, health, persistence |
| E02 | Shape | complete | Real GPU shape generation, GLB viewer, checkpointed artifacts |
| E03 | PBR | complete | Real GPU texture generation with separated model lifecycle and telemetry |
| E04 | Unity | complete | LOD/collision artifacts, manifest, validation report, ZIP; headless Editor import verified via `unity-smoke-test/` |
| E05 | Omni/hardening | complete | Controls, cancellation, SSE recovery, restart recovery, release evidence |
| E06 | Text bridge | complete | FLUX.2-klein text-to-3D and retexture verified on real GPU |
| E07 | Vehicle rig | complete | Wheel marking → rigged GLB → WheelCollider setup verified in headless Unity |

## Resume prompt

Continue from `STATUS.md` and `VERIFICATION.md`. Benchmark evidence lives in `data/benchmarks/`. Preserve unrelated work. Do not claim Unity Editor or visual-quality verification without evidence.

## Milestones

| ID | Name | Status | Exit evidence |
|---|---|---|---|
| E01 | Foundation | complete | API/UI vertical slice, health, persistence |
| E02 | Shape | complete | Real GPU shape generation, GLB viewer, checkpointed artifacts |
| E03 | PBR | complete | Real GPU texture generation with separated model lifecycle and telemetry |
| E04 | Unity | complete | LOD/collision artifacts, manifest, validation report, ZIP; headless Editor import verified via `unity-smoke-test/` |
| E05 | Omni/hardening | complete | Controls, cancellation, SSE recovery, restart recovery, release evidence |

## Resume prompt

Continue from `STATUS.md` and `VERIFICATION.md`. Benchmark evidence lives in `data/benchmarks/`. Preserve unrelated work. Do not claim Unity Editor or visual-quality verification without evidence.
