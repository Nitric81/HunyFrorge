# HunyForge Status

**Updated:** 2026-09-17  
**Overall:** Performance & GUI plan verified; text-to-3D, text-guided retexture, vehicle wheel-rigging, and opt-in Unreal Engine export on target hardware  
**Current milestone:** Unreal export shipped — per-job `unreal_export` produces `unreal-package.zip` (GLB + Editor Python setup); UE 5.8.2 capability probe verified end-to-end  

## Current truth

- Implemented: the full performance and GUI scope — explicit Draft/Standard/Final presets resolved into real inference parameters, lazy per-stage model loading with release between shape and texture, stage-aware progress and operation telemetry, checkpointed retry/resume/restart, cancellation states, partial-artifact availability (white mesh preview/download during texturing), configurable LOD/collision with mandatory validation and packaging for completed jobs, mesh-token texture requests (no base64 GLB round trip), SSE with polling fallback and tri-state connection reporting (`live`/`degraded`/`down`), mobile drawer focus management, and Docker runtime env configuration.
- **Text features (FLUX.2-klein-4B):** `input_mode` text/image switch, server-side prompt scaffolding, `POST /api/reference-preview` (approve-before-burn), text-guided retexture as immutable child jobs, Edit tab with viewport capture → FLUX edit → child job, Before/After parent compare, T2I health block, lineage badges. Deps bumped: diffusers 0.38.0, transformers 4.51.3; deepspeed removed; `trust_remote_code` for vendored pipeline.
- **Vehicle rigging:** `asset_type` selector (Generic / Character placeholder / Vehicle); new `rig` pipeline stage (skipped for non-vehicle); `POST /wheel-suggest` (ground-strip + quadrant clustering + dual-axis cylinder fit with axial-normal scoring) and `POST /rig` → CPU-side child job partitioning wheels onto axle pivots; rigged GLB (`Chassis` + `Wheel_*`), manifest `vehicle` block, `HunyForgeVehicleSetup.cs` in the Unity package; viewer wheel markers with click-to-place + numeric editing; `RIG` badge, `Rigged` preview source.
- Runtime defaults (RTX 4080 verified): `HUNYUAN_DINO_DEVICE=cuda`, `HUNYUAN_FLASHVDM=1`, `HUNYUAN_COMPILE=0`, `HUNYUAN_CPU_THREADS=8`. Pinned source `82920d643c0dc2f7bfd7255f45f62d386edfe60c`, model `0b94677654c57bb9a6b6845cd7b704ccf551d327`.
- Verified on real GPU: Draft 181.9 s, Standard 176.6 s, Final 250.8 s (vs original-runtime baseline 3279 s). Text-to-3D buggy `d9ba5f4c` end-to-end; rig child `8928f6fc` complete with validation passed.
- Verified recovery: unplanned Docker restart mid-job → clean failure + restart child completes; pre-fix rigged child correctly re-fails on stale checkpoints (immutability preserved).
- **Unreal export (opt-in per job):** `unreal_export` flag on `POST /jobs` (inherited by retexture/rig/resume children); export stage emits `unreal-lod0/1.glb`, `unreal-collision.glb`, `hunyforge-unreal-manifest.json` (cm/Z-up notes, `collision_mode`, Chaos hints for rigs); validation gains `checks.unreal` + `unreal_ready` with `unreal-export-*`/`unreal-manifest-*` blocking rules; `unreal-package.zip` (`HunyForge/` layout: `SM_*.glb`, `Collision_*.glb`, manifest, reports, `HunyForgeUnrealSetup.py`, README). `scripts/setup-unreal.ps1` auto-detects the engine (registry/UE_* dirs/`-EnginePath`), enables plugins in the `.uproject` (with backup), and runs the setup headless.
- **Vegetation asset type:** `asset_type: vegetation` (job + UI selector) adds COLOR_0 wind weights (R sway / G height) to all export LODs via GLB surgery, `collision_mode: trunk` fits a capsule to the base quartile instead of wrapping the crown, a FLUX leaf-spray atlas (`T_Leaf_*.png`, magenta chroma-keyed) is generated in the unity stage when T2I is enabled, and the unreal manifest gains a `foliage` block (`material_mode`, `wind_channels`, `trunk_capsule`). `HunyForgeUnrealSetup.py` builds a dedicated `M_<mesh>_Foliage` material (base-color texture from the mesh's own folder, opaque blend — PBR alpha is not leaf alpha — two-sided foliage shading) with a manual sway graph on WPO (`sin(Time·1.3 + WorldPos·(0.03,0.017,0)) × VertexColor.R × 6 × direction`; UE 5.8 Python has no SimpleGrassWind expression and no AggGeom write access, so trunk-only collision is a fitted capsule + manual-resize dims in the report). Validation blocks on missing vertex colors in vegetation exports. `ironhound-assets/queue-ironhound-vegetation.ps1` queues cypress/snag/willow.
- **Verified vegetation end-to-end (UE 5.8.2):** all three queue assets (cypress `SM_e050801a` 81k/32k LOD verts, dead-snag `SM_fc9de27e` 53k/20k, swamp-willow `SM_f93bc920` 112k/42k) generated on the real GPU path and imported with LOD1, `T_Leaf_*` atlas, `M_*_Foliage` material (wind chain compiled live via MCP, `wind-wpo-vertex-color` action), and fitted capsule collision. Live viewport captures confirm textured leaf-card canopies and bark — Hunyuan3D produces real card-structured crowns for dense-canopy species. Caveats: prompts may need species tuning (cypress reads palm-ish), GLB pivots sit below geometry (actors float ~1 m — snap on placement), and headless setup runs still end in a cosmetic Slate shutdown crash after `result: OK`.
- Verified tests: backend 144 pass (5 Unreal-export incl. package layout + retry propagation), frontend 49 pass (4 Unreal UI), `tsc`/Vite build clean.
- Verified Unity Editor import: headless 6000.5.8f1 + glTFast smoke test — **0 failures** including rigged-vehicle checks (node hierarchy, pivots, spin-not-orbit, 4/4 WheelColliders + Rigidbody via the packaged script).
- Verified Unreal bootstrap: `setup-unreal.ps1` probe against UE 5.8.2 headless — glTF import via Interchange works; `import_lod`, `set_lod_reduction_settings`, `add_simple_collisions`, `set_convex_decomposition_collisions` all present. Real `unreal-package.zip` → StaticMesh import still to be verified on a live project.
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
| E08 | Unreal export | complete | `unreal_export` per-job flag → unreal artifacts + manifest + `unreal-package.zip`; bootstrap probe verified on UE 5.8.2; live-project import pending |

## Resume prompt

Continue from `STATUS.md` and `VERIFICATION.md`. Benchmark evidence lives in `data/benchmarks/`. Preserve unrelated work. Do not claim Unity Editor or visual-quality verification without evidence.
