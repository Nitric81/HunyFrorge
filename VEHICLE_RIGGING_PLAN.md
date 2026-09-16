# HunyForge Vehicle Rigging Plan

## Objective

Hunyuan3D-2.1 emits a **single fused rigid mesh** — no parts, pivots, or hierarchy. Unity `WheelCollider` vehicles need the opposite: a chassis transform plus separate wheel transforms with pivots centered on each axle. This plan adds a **vehicle rigging flow** that converts a generated vehicle mesh into a rigged GLB node hierarchy plus Unity-side auto-setup metadata.

Two user-visible additions:

1. **Asset-type selector** on the Generate panel: `Generic | Character | Vehicle`. `Vehicle` opts the job into rigging. `Character` is a disabled placeholder — character features (skeleton/skinning) are a later plan.
2. **Wheel-marking → rig** flow on completed vehicle jobs: user marks wheel positions on the mesh in the 3D viewer (with server-side auto-suggestion), the backend partitions the mesh into chassis + wheel nodes with correct pivots, and the Unity package gains a `WheelCollider` auto-setup editor script.

## Hard constraints and honest caveats

- **No true articulation exists in the source mesh.** Wheels are fused to the body. We *partition faces* — triangles inside a wheel's cutting cylinder become the wheel node, the rest stays chassis. This is not a boolean cut; the boundary is jagged at triangle granularity.
- **Holes appear** where wheels detach from the body. We cap open boundary loops on both parts. Caps on the chassis look like covered wheel wells (usually hidden behind the wheel); caps on wheels are inside the fender. Cap faces get fallback UVs/material — visible up close on textured models. Documented, not hidden.
- **Blobby generations defeat partition.** When a generated "car" melts wheels into the undercarriage, partition may produce a wheel containing fender geometry. The flow must fail honestly ("region not separable") or let the user shrink/re-position the marker region — never silently emit garbage.
- **Pivots must be axle-centered.** A wheel node whose pivot is off-axis *orbits* instead of spins. The pivot is the cylinder-axis center, not the marked surface point. The Unity smoke test must verify spin-not-orbit.
- **Axis conventions:** meshes are Y-up; forward is assumed −Z per glTF, which Unity/glTFast converts to +Z. The manifest records the assumed forward/axle axes so the setup script is deterministic. Mismatched orientation is corrected by the user marking + a `flip` flag, not by guessing.
- **Interactive input cannot be a pipeline stage.** Jobs run autonomously once queued; a stage that blocks waiting for wheel clicks would break cancellation/timeout semantics. Rigging is therefore a **post-completion child-job action**, identical in spirit to the retexture flow.

## Architecture decisions

- **`asset_type` job field:** `generic | character | vehicle` on `JobCreate`, stored on the job record, surfaced in health-agnostic metadata. Drives UI affordances only — the pipeline itself is unchanged by asset type.
- **Rig = child job with lineage** (`parent_job_id`, `input_mode` extended or a dedicated `derivation: "vehicle-rig"` marker — decide in Phase 1; keep `input_mode` literal narrow if possible). Parent stays immutable. Markers are persisted on the child (`rig_spec`) so the rig is auditable and re-runnable with different markers.
- **New pipeline stage `rig`** between `texture` and `unity`: `PIPELINE_STAGES = ("shape", "texture", "rig", "unity", "validation")`. Non-vehicle jobs mark it `skipped` (existing `stage_status.setdefault` + skip pattern). The rig child resumes at `rig`, so its `shape`/`texture` stages inherit `complete` via the existing resume machinery and `unity`/`validation` re-run against the rigged artifacts — LODs, collision, manifest, and `unity-package.zip` are all regenerated for the rigged model.
- **Rig runs on CPU in the API process** (trimesh, like `prepare_unity_geometry` via `asyncio.to_thread`). No GPU worker involvement, no worker lock, no new model weights.
- **Artifact set for a rigged vehicle:** `vehicle-rigged.glb` (hierarchical: `Chassis` + `Wheel_*` nodes), plus the standard LOD/collision/manifest set produced by the unity stage — where LODs apply to the **chassis** and wheel nodes ride along unchanged. `hunyforge-manifest.json` gains a `vehicle` block: per-wheel node name, local pivot, radius, axle axis, steer flag, suggested suspension travel/mass.
- **Unity auto-setup:** the package gains `Assets/HunyForge/Editor/HunyForgeVehicleSetup.cs` — an editor menu action that reads the manifest, locates the imported prefab's wheel nodes, and creates `WheelCollider`s (position, radius, suspension spring) plus a `Rigidbody` with a sane default mass. Ships inside `unity-package.zip` only for rigged vehicles.
- **Wheel suggestion endpoint:** `POST /api/jobs/{id}/wheel-suggest` → trimesh analysis of the job's mesh: ground plane = min-Y; cluster bottom-region geometry by lateral/longitudinal quadrants; per-cluster PCA fits a cylinder (axis, center, radius). Returns suggested wheel descriptors. Always a *suggestion* — the user confirms or re-marks. This runs in-process like other geometry checks.
- **Viewer marking:** `ThreePreview` gains a marking mode — pointer raycast against the loaded mesh places/moves marker spheres (one per wheel), re-click moves the nearest marker, up to 4 by default. Marker positions are model-space (identical across white/textured artifact variants).

## Phase 0: Geometry feasibility spike — ✅ PASSED (2026-09-14)

Spiked with `scripts/spike_vehicle_rig.py` on a real generated vehicle — a buggy produced by a **text-to-3D job** (`d9ba5f4c`, Draft preset, `input_mode=text`, all 8 artifacts, real GPU — this also closed the text plan's real-GPU text-to-3D verification).

**Verified working:**

- Face partition by marked cylinder → `trimesh.Scene` with named nodes (`Chassis`, `Wheel_FL/FR/RL/RR`) → GLB export → reload preserves node names and pivot transforms.
- Wheel suggestion heuristic (bottom-band quadrant clustering + two-pass cylinder fit): 4 plausible wheels on a real generation — symmetric centers, radius ~0.15, ~5% face capture each.
- Pivot ≈ suggestion center is adequate: centroid drift under 30° axle rotation ~0.015 (≈9% of wheel radius) — small wobble from asymmetric capture, not pivot error. Bbox-center pivots tested worse; keep cluster-fit centers.

**Findings that changed the design:**

- **Generated meshes can keep the studio floor.** The buggy's `white-mesh.glb` was 694k faces — 231k (33%) were a flat ground plane at min-Y. Without stripping, wheel suggestion over-captured by ~2.5× (radius 0.42 vs true 0.16, half-width spanning the whole flank). The rig stage must run `strip_ground_plane` first (faces fully within a thin min-Y layer with near-vertical normals, two passes). Also a general artifact-quality flag — floor geometry is dead weight in every exported model, worth a `remove_ground_plane` generation setting or default-on cleanup.
- **`fill_holes` is not sufficient** for partition caps — boundary loops are ~400–460 jagged edges each and post-fill parts remain non-watertight. Implementation needs explicit boundary-loop fan triangulation (chassis gets 4 wheel holes + floor-removal edges: ~3.7k boundary edges total on this model).
- Axle orientation was confirmed empirical (vehicle faced ±Z → axles along X), but suggestion must **not hard-assume** — fit both candidate axes and prefer the better cylinder fit (parts should be thin on exactly one axis).

Acceptance was met: rigged GLB reloads with correct hierarchy, pivots are axle-plausible, seams are bounded. The shared-geometry fallback was not needed.

## Phase 1: Backend — ✅ DONE (2026-09-14)

Shipped as designed, with these additions/corrections found during implementation and the real-mesh run:

- `models.py`: `asset_type` on `JobCreate`/`JobStatus`; `VehicleWheelSpec`/`VehicleRigSpec` models; `rig` in `PIPELINE_STAGES`/`resume_from`/`JobStage.RIGGING`; `input_mode="vehicle-rig"` (kept the literal narrow — no separate `derivation` field); `rig_checkpoint` on the job record.
- `vehicle_rig.py` (new, ~300 lines): `strip_ground_plane` (multi-pass, UV-preserving), `suggest_wheel_regions` (quadrant clustering + two-pass cylinder fit, **both axle axes tried and scored**), face partition, boundary-loop fan caps with **cap-vertex UVs** (mean of loop UVs), pivot recentering, `build_rigged_glb`, `rigged_node_summary`, `rig_quality_checks`.
- `pipeline.py`: `_rig_stage` (skipped without `rig_spec`); `_prepare_rigged_unity_geometry` — decimates the Chassis node only for LOD1, wheels ride along with pivots; manifest `vehicle` block; package embeds `HunyForgeVehicleSetup.cs`; validation runs `rig_quality_checks` (node presence, pivot match, overlap blocking).
- `storage.py`: rig checkpoint requirements + `prepare_resume` copies rig artifacts for rigged jobs resuming at `unity`/`validation` (previously skipped unconditionally — a resume would have silently produced un-rigged exports).
- `main.py`: `POST /api/jobs/{id}/wheel-suggest` and `POST /api/jobs/{id}/rig` (bounds + plausibility validation → 422s; `prepare_resume` failures → clean 409s). `_child_job` now propagates `asset_type`/`rig_spec`/lineage fields so resumed rig children don't lose their rig.

**Real-run bugs caught and fixed (Phase-3 evidence):**
- Boundary-loop walker could cycle forever on junction vertices — each step now consumes a fresh boundary edge.
- Ground-strip rebuilt the mesh without visuals → textured rigged GLBs lost UVs/material; strip now uses the UV-preserving submesh helper.
- Cap center vertices lacked UVs (`len(uv) < len(verts)`) → export wrote degenerate all-0.5 UVs; cap centers now inherit the loop's mean UV.
- Axis scoring originally picked a thin *slab* through the wheel instead of the true disk — added an axial-normal-fraction penalty (wheel surfaces point radially, ⊥ the real axle; measured 0.22 vs 0.58 on the real mesh) which selects the correct X axle on the buggy.

Acceptance met: 11 new tests in `backend/test_vehicle_rig.py` (synthetic box+cylinder vehicle) covering strip, suggestion naming/steer, partition→export→reload pivots, spin recentering, empty-region honest failure, overlap blocking, rig-stage skip on generic jobs, full API lineage (suggest → rig child → manifest/package/validation), out-of-bounds rejection, and resume-at-unity copying the rig checkpoint. Suite: **136 pass**.

## Phase 2: Frontend — ✅ DONE (2026-09-14)

- **Generate tab:** asset-type segmented (`Generic | Character | Vehicle`) above the input-mode switch; Character disabled with "lands in a later phase" hint; Vehicle submits `asset_type` and shows the rigging hint.
- **Rig tab** (4th inspector tab, dimmed for non-vehicle jobs): auto-suggest → editable wheel cards (name, steer toggle, X/Y/Z center, radius, width, remove) → **Build rigged child job**. "Place" enters click-to-mark mode (`ThreePreview` raycast; numeric fields remain the keyboard-accessible path).
- **Viewer:** wireframe cylinder + hub markers per wheel (teal = steering, amber = driven); `Rigged` added to the White/Textured preview-source toggle; `RIG` badge + lineage in history; `vehicle-rigged.glb` preferred in auto preview.
- Stage list gains a `Vehicle rig` row (legacy jobs render `skipped`).

Acceptance met: 3 new frontend tests (asset-type payload, full suggest→build flow, RIG badge); **45 pass**, `tsc` clean, `vite build` clean.

## Phase 3: Unity package + smoke test — ✅ DONE (2026-09-14)

- `backend/unity/HunyForgeVehicleSetup.cs` ships in `unity-package.zip` under `Assets/HunyForge/Editor/` for rigged vehicles. Menu `HunyForge → Setup Vehicle Colliders` reads the manifest, finds wheel nodes, creates configured `WheelCollider`s (radius, suspension spring scaled to chassis mass) + `Rigidbody`.
- `unity-smoke-test` extended: imports `vehicle-rigged.glb`, asserts node hierarchy + pivots vs manifest (axis-flip tolerant), wheel bounds centered on pivot (spin-not-orbit), then runs the setup script and asserts collider count + radii.
- Headless Unity 6000.5.8f1 run on the real rigged buggy (`8928f6fc`): **10 checks, 0 failures** — lod0 20,035 verts/5 meshes with UVs+materials, lod1 7,783 verts (61% chassis reduction, hierarchy preserved), `wheelColliders=4/4`, `rigidbody mass=1200`, collider radii within tolerance.

## Phase 4: Verification — ✅ DONE (2026-09-14)

- Backend suite **136 pass** (11 new vehicle tests). Frontend suite **45 pass** (3 new); `tsc`/`vite build` clean.
- **Real end-to-end:** text-to-3D buggy `d9ba5f4c` → `POST /wheel-suggest` (4 wheels, correct X axle, FL/FR steer, ground strip 364 faces on the textured LOD) → `POST /rig` → child `8928f6fc` complete: `vehicle-rigged.glb` (Chassis + 4 wheel nodes, pivots verified on reload), `vehicle-rig-report.json`, regenerated LOD/collision/manifest (with `vehicle` block), validation `passed` with rig checks, `unity-package.zip` containing the setup script.
- **Recovery verified:** a pre-fix failed rig child (`efe65245`, UV regression) resumes correctly at `validation` with preserved checkpoints — it re-fails validation on its stale rigged artifact, as the immutable-checkpoint design requires; the fixed path is proven by `8928f6fc`.
- Live at `http://127.0.0.1:8081` — backend + frontend synced into the running container.

## Non-goals

- **Character rigging** (skeletons, skinning, Humanoid avatar) — explicitly deferred; the selector placeholder reserves the UX.
- More than 4 wheels / tracked vehicles / trailers — the `rig_spec` schema is a list so N wheels are structurally possible, but only 4-wheel layouts are tested.
- Physics tuning presets (suspension curves, torque curves) — the manifest emits sane defaults only.
- Automatic wheel detection without user confirmation — suggestion is assistive, never authoritative.
- In-place rigging of the original job — lineage/immutability rules apply as with retexture.

## Open decisions — resolved

- `input_mode` vs dedicated `derivation` marker — resolved: `input_mode="vehicle-rig"`; the literal stays narrow.
- Marker count flexibility — resolved: schema supports N wheels (1–8); the UI defaults to the suggested 4 with an "Add wheel marker" escape hatch.
- `rig` on non-vehicle jobs — resolved: the endpoint stays permissive (any job with a mesh artifact can be rigged); the UI only surfaces the Rig tab for `asset_type === 'vehicle'` or `vehicle-rig` lineage.

## Known limitations

- Cut seams are jagged at triangle granularity; fan caps are visible up close on textured models (cap verts inherit mean-loop UVs — plausible but not a re-texture).
- Auto-suggestion can over-capture fender geometry on blobby generations — the markers are editable precisely because suggestion is assistive, never authoritative.
- Ground-plane stripping targets flat min-Y sheets only; non-planar studio junk survives.
- A pre-fix failed rig child keeps its stale artifacts under immutable checkpoints — re-rig from the parent rather than resuming pre-fix children.
