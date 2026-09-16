# HunyForge — Product and Technical Design

**Status:** Proposed | **Version:** 0.1 | **Updated:** 2026-09-11

## Product

HunyForge is a local-first desktop web application that turns reference images into Unity-prepared 3D game assets using Hunyuan3D. It keeps inference and project data local, exposes intermediate results, and does not mark an asset complete until Unity export validation passes.

**Audience:** solo Unity developers and small teams with NVIDIA GPUs.  
**Positioning:** Local AI-powered 3D asset generation for Unity developers.

## Constraints and non-goals

- Baseline: RTX 4080 16 GB VRAM, 32 GB RAM, Windows-first.
- No cloud inference, telemetry, or collaboration in v1.
- Hunyuan3D shape and texture stages must be memory-separated. Tencent reports about 10 GB for shape, 21 GB for texture, and 29 GB together.
- Unity export preparation is mandatory MVP functionality.
- One active GPU inference job at a time in v1.
- Preserve seed, parameters, model versions, inputs, outputs, and validation history.
- Review Hunyuan and third-party licenses before commercial distribution or bundling weights.

## User workflow

1. Upload one or more reference images.
2. Choose Hunyuan3D 2.1 or Hunyuan3D-Omni.
3. Configure seed, inference steps, guidance, octree resolution, background removal, and optional control inputs.
4. Generate a white mesh locally.
5. Release/offload the shape model, then generate PBR textures.
6. Inspect geometry, materials, dimensions, topology, and channels in a 3D viewer.
7. Prepare and validate a Unity package.
8. Export the package with manifest, validation report, and known limitations.

## Reference capabilities

Hunyuan3D 2.1 provides image-to-shape, PBR texture synthesis, Python APIs, a local Gradio app, GLB/OBJ export, seed and generation controls, background removal, multi-view support, mesh reduction, and model viewing. The official setup is tested with Python 3.10 and PyTorch 2.5.1 + CUDA 12.4.

Hunyuan3D-Omni adds point-cloud, voxel, skeleton/pose, and bounding-box control. Use it as an advanced shape backend after the 2.1 path is stable.

## Architecture

- **UI:** React + TypeScript + Vite; Three.js or React Three Fiber viewer.
- **API:** FastAPI, Pydantic, Uvicorn, localhost-only by default.
- **Persistence:** SQLite for metadata; filesystem for binary artifacts.
- **Orchestration:** serialized job worker that owns model loading, VRAM transitions, cancellation, retries, and recovery.
- **Inference:** official Hunyuan3D Shape/Paint repositories behind adapter interfaces.
- **Processing:** mesh cleanup, reduction, normalization, materials, LODs, collisions, and validation behind replaceable adapters. Blender headless may be used for robust Unity processing.
- **Events:** WebSocket preferred, polling fallback.

## Required UI

### Generate

Image upload, optional front/back/left/right views, optional prompt, backend selection, Omni control type/file, background removal, seed/randomization, inference steps, guidance scale, octree resolution, chunk size, texture toggle, and saved presets.

### Inspect

Orbit/pan/zoom, solid/wireframe, normals, material preview, PBR channels, dimensions, scale, face/vertex counts, white/textured comparison, intermediate downloads, and job logs.

### Unity preparation

This panel is required after every generation. It configures units, scale, axes, pivot, triangle budget, texture size/packing, material mode, LOD policy, collision policy, format, and output location. Blocking failures and warnings must be distinct.

## UI/UX design system and requirements

### Design principles

- **Workbench, not wizard:** keep the preview visible while controls change.
- **Progressive disclosure:** show the common path first; place expert inference and mesh controls behind expandable sections.
- **State is always visible:** the user can always see current asset state, active stage, queue position, GPU usage, elapsed time, and the next available action.
- **Evidence over decoration:** prioritize geometry, material, validation, and provenance information over marketing visuals.
- **Local and calm:** use a dark neutral editor-style surface with one restrained accent color for active actions and progress. Do not use color alone to communicate errors.

### Application shell

Use a desktop-first three-column shell:

- **Left rail, 240–280 px:** project switcher, asset library, new generation, presets, settings, and model status.
- **Center workspace, fluid:** viewer and inspection tabs; this is the visual focus.
- **Right inspector, 320–380 px:** context-sensitive Generate, Inspect, Unity Prepare, and Validation controls.
- **Bottom activity strip:** current job, stage progress, VRAM, cancel/retry, and expandable logs.

The shell must remain usable at 1280×720 and scale to wide desktop monitors. At narrower widths, collapse the left rail and turn the inspector into a tabbed drawer; never hide an active job or blocking validation failure.

### Primary screens

1. **Project Home:** recent assets, status filters, search, sort, new asset action, and a compact GPU/model readiness panel.
2. **Generation Workspace:** source image drop zone, generation controls, live viewer, and job timeline.
3. **Asset Inspection:** viewer controls, mesh/material statistics, artifact list, provenance, and comparison mode.
4. **Unity Preparation:** export preset, normalization controls, LOD/collision settings, material/texture settings, validation preview, and export action.
5. **Validation Report:** blocking errors, warnings, passed checks, affected artifact, suggested fix, and rerun action.
6. **Settings:** model paths, cache location, project root, GPU diagnostics, renderer/tool availability, and privacy/license information.

### Component requirements

- Use consistent 8 px spacing increments, clear section headings, compact labels, and predictable control ordering.
- Every advanced numeric input needs a unit, range, default, reset action, and inline explanation.
- Destructive actions such as delete, replace, or cleanup require confirmation and identify the exact artifact affected.
- Buttons must have explicit labels; icon-only buttons require tooltips and accessible names.
- Use `queued`, `active`, `passed`, `warning`, `failed`, and `cancelled` states consistently across jobs and validation.
- Toasts are for non-blocking confirmation only. Important errors remain visible in the relevant panel and job history.
- Long-running actions must support cancel, retry, and opening the latest intermediate result.

### Generation interaction requirements

- Disable only controls that cannot safely change during the active stage; explain why a control is disabled.
- Show the pipeline timeline as Shape, Mesh Processing, VRAM Transition, Texture, Unity Preparation, and Validation.
- Display the selected seed and provide “rerun with same settings” and “duplicate and edit” actions.
- Show a clear low-VRAM notice before texture generation on a 16 GB target.
- If a stage fails, preserve completed artifacts and show the exact failed stage, error category, log access, and retry scope.

### Viewer requirements

- Provide mouse, keyboard, and touchpad orbit/pan/zoom controls.
- Include fit-to-view, reset camera, grid, axes, lighting, wireframe, normals, and material toggles.
- Support side-by-side or slider comparison between source, white mesh, and textured mesh.
- Keep statistics readable without obscuring the model.
- Never load untrusted remote content; viewer assets must come from the local artifact service.

### Accessibility and quality bar

- Meet WCAG 2.2 AA practices where applicable: keyboard navigation, visible focus, semantic labels, sufficient contrast, reduced-motion support, and screen-reader announcements for job-state changes.
- Do not communicate status through color alone; pair color with text and an icon or pattern.
- Maintain usable contrast over the 3D viewport and provide a high-contrast UI option.
- Test at 100%, 125%, and 150% Windows display scaling.
- UI tests must cover keyboard-only generation, validation failure recovery, cancellation, retry, and export completion.

### UI acceptance criteria

- A new user can create a project, upload an image, start generation, and find the resulting asset without documentation.
- An expert can reach every model and Unity setting without editing configuration files.
- During a long job, the user can identify stage, progress, VRAM, elapsed time, cancel action, and latest artifact within two seconds.
- A Unity export failure identifies the blocking check and does not offer a misleading “complete” state.
- A completed asset presents an obvious Unity export action and links to its manifest and validation report.
- All screens remain functional with keyboard navigation and at the supported minimum resolution.

## API contract

Required endpoints: `GET /health`, `POST /api/projects`, `GET /api/projects`, `GET /api/projects/{id}`, `POST /api/jobs`, `GET /api/jobs`, `GET /api/jobs?project_id={id}`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`, `POST /api/jobs/{id}/retry`, job event streaming, artifact listing/download, and Unity preparation/validation endpoints.

Required job states: `queued`, `loading_shape_model`, `generating_shape`, `processing_mesh`, `releasing_shape_model`, `loading_texture_model`, `generating_textures`, `preparing_unity_export`, `validating_unity_export`, `complete`, `failed`, `cancelled`.

Every schema is versioned. API paths must be project-ID based; arbitrary filesystem paths are forbidden. Store `peak_vram_mb`, timings, model revisions, seed, parameters, artifact hashes, and error codes.

## Unity export contract

Every completed asset must include:

- Unity-compatible GLB or FBX.
- Normalized units, axes, and pivot metadata.
- Materials plus base color, normal, metallic, roughness, and applicable PBR textures.
- Triangle-budget result.
- LOD output or explicit LOD failure.
- Collision output or explicit collision failure.
- `hunyforge-manifest.json` with provenance and settings.
- `validation-report.json` with checks, severities, and tool versions.
- Human-readable Unity import instructions and known limitations.

Blocking validation checks include readable files, finite vertices, valid indices, non-empty mesh, UVs when textured, resolvable materials, known scale, recorded axis conversion, texture dimensions/formats, LOD consistency, and collision policy.

## Storage and privacy

Use `<workspace>/hunyforge.db` and `<workspace>/projects/<project-id>/assets/<asset-id>/` with separate `inputs`, `generations`, `unity-export`, `previews`, and `logs` folders. Never overwrite a generation. Use content hashes. Bind to localhost, sanitize names, prevent path traversal, constrain external tools with timeouts, and redact secrets from logs.

## Delivery phases

- **E01 Foundation:** repository, pinned environment, config, SQLite schema, artifact paths, health check, logging, CI.
- **E02 Shape:** upload, 2.1 shape worker, async lifecycle, progress, reproducible seed, white-mesh GLB, viewer.
- **E03 PBR:** staged texture worker, unload/reload, low-VRAM behavior, retry/resume, measured VRAM, texture preview.
- **E04 Unity:** scale/axis/pivot, materials, texture packing, triangle budgets, LODs, collisions, manifest, validator, package export.
- **E05 Omni and hardening:** point/voxel/pose/bounding-box inputs, cancellation, recovery, performance, accessibility, release packaging.

## Development TODO — generation history and stage resume

**Priority:** High. **Status:** Planned.

Add durable generation history and true stage-level recovery so a failed or interrupted local run does not require repeating completed GPU work.

Required behavior:

- Persist every job, input, parameter set, model revision, stage transition, timing, error, and artifact hash.
- Write and validate checkpoints after shape generation, mesh processing, texture generation, Unity preparation, and validation.
- Mark jobs interrupted after an API/container restart when they were active.
- Add `POST /api/jobs/{id}/resume`; resume only from the latest valid checkpoint.
- Keep `POST /api/jobs/{id}/retry` as an explicit restart-from-beginning action.
- Reject resume when a required checkpoint is missing, corrupt, or incompatible with the current pipeline/model revision; offer the earliest safe restart stage.
- Add a GUI History view with thumbnails, timestamps, duration, backend, seed, stage, failure details, artifacts, Resume, and Restart actions.
- Preserve the existing mandatory Unity preparation and validation stages when resuming; never report a job complete without a valid Unity export package and validation report.
- Keep the API usable by non-GUI workflows; expose job status, checkpoints, resume/restart actions, event streaming, and artifact downloads through documented endpoints.

Acceptance criteria:

1. A failure during texture generation resumes from the saved shape/mesh checkpoint without rerunning shape inference.
2. A failure during Unity preparation resumes without rerunning shape or texture inference.
3. Restarting a failed job creates a new generation record and does not overwrite the original.
4. Restarting the container converts active jobs to `interrupted` and makes them recoverable.
5. The GUI and API show the same authoritative stage, checkpoint, error, and artifact state.
6. Tests cover checkpoint integrity, incompatible checkpoints, resume transitions, interruption recovery, duplicate resume requests, and artifact immutability.

## Verification

Use unit tests for schemas, path safety, transitions, normalization, and manifests; backend tests with mocked workers; Playwright UI tests; GPU smoke tests on the target RTX 4080; golden-file tests for manifests/reports; and a Unity-project import smoke test. Record commit, environment, model revision, command, result, and artifact path for release evidence.

## Definition of done

An item is complete only when implementation, tests, documentation, and evidence exist. GPU-dependent behavior must be labeled `verified`, `unverified`, or `blocked`. Generated geometry may still need retopology, UV correction, texture cleanup, and human review; the UI must state this clearly.

## References

- [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [Hunyuan3D 2.1 API documentation](https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan3D-2.1/main/API_DOCUMENTATION.md)
- [Hunyuan3D 2.1 Gradio app](https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan3D-2.1/main/gradio_app.py)
- [Hunyuan3D-Omni](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni)
