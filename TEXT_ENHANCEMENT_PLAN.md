# HunyForge Text-to-3D & Text-Guided Retexture Plan

## Objective

Add two prompt-driven flows on top of the existing Hunyuan3D-2.1 pipeline:

1. **Text-to-3D** — user prompt → generated reference image → existing image→3D job.
2. **Text-guided retexture** — user prompt + render of an existing model's mesh → styled reference image matching the geometry → texture-only stage on the job's `white-mesh.glb` checkpoint.

Hunyuan3D-2.1 has no text channel (verified: shape conditioner is `image=` only; hy3dpaint textures from the reference image). Text therefore enters exclusively through a **text-to-image front stage** producing the same reference image the upload field produces today. Every downstream contract — image-hash binding, background removal, mesh tokens, validation, packaging — stays unchanged.

## Selected model: FLUX.2 [klein] 4B

`black-forest-labs/FLUX.2-klein-4B` (step+guidance distilled variant — **not** `klein-base-4B`, which needs ~50 steps).

- **License:** Apache 2.0 — commercial-safe. The 9B sibling is non-commercial; pin the 4B exactly.
- **Unified T2I + editing:** `Flux2KleinPipeline` accepts `image=` reference input (single and multi-reference editing). Retexture uses the mesh render as the edit reference — no ControlNet stack required.
- **Speed:** 4 inference steps, guidance 1.0, sub-second-class generation.
- **Size:** ~8 GB VRAM per BFL; fits the RTX 4080 easily since model stages serialize and unload.

## Constraint and risk inventory

- The container pins `diffusers==0.30.0` and `transformers==4.46.0`. `Flux2KleinPipeline` requires a newer diffusers (post-0.30 merge) and a transformers new enough for `Qwen3ForCausalLM`. **This is the plan's biggest risk:** hy3dshape/hy3dpaint consume diffusers schedulers/transformers — a bump can silently regress generation. Phase 0 exists solely to gate this.
- The GPU worker is a single serialized process. Reference previews must queue behind the same lock — a preview can never run concurrently with a job stage.
- Weight delivery follows the existing model-mount pattern (`HUNYUAN_MODEL_PATH` host mount, read-only) rather than baking ~8 GB into the image.
- Retexture jobs carry a **different image than their parent** — today's worker hard-rejects that (`worker_runtime.py` image-equality check). The plan adds a lineage-aware exception, deliberately narrow.
- Prompt scaffolding (wrapping the raw prompt with "single object, centered, plain background, front view") is the largest quality lever for image→3D inputs and is applied server-side so API and UI share it.

## Phase 0: Dependency feasibility gate — ✅ PASSED (2026-09)

Verified in-container before baking; now baked into `Dockerfile` + `docker/configure_runtime.py`.

- **Upgrade set:** `diffusers==0.38.0` (klein pipeline present since 0.37), `transformers==4.51.3` (first Qwen3-capable 4.x — minimal delta from 4.46), `huggingface-hub==0.36.2` (latest 0.x; transformers requires <1.0, diffusers requires >=0.34), `safetensors==0.8.0`, `tokenizers==0.21.4`, `accelerate==1.10.1`. Layer placed after CUDA extension builds to preserve compile cache.
- **Defect found & fixed — deepspeed:** upstream pins `deepspeed`, whose `tp_collectives` custom op uses `list[int]` schema annotations unsupported by torch 2.5.1 — `import deepspeed` was *already broken* in the production image, but nothing on the inference path imported it (latent landmine). transformers 4.51 eagerly imports deepspeed when present → hard failure. Fix: `pip uninstall deepspeed` in the image (nothing in `/opt/hunyuan` or `backend/` imports it).
- **Defect found & fixed — trust_remote_code:** diffusers 0.38 requires `trust_remote_code=True` for local `custom_pipeline` dirs; `hy3dpaint/utils/multiview_utils.py` loads the vendored `hunyuanpaintpbr` this way (its `modules.UNet2p5DConditionModel` is also custom code). Patched via `configure_runtime.py` — the code is our own pinned vendor source. Verified end-to-end by a real texture-stage resume.
- **FLUX verified:** `Flux2KleinPipeline.from_pretrained` + `enable_model_cpu_offload()` loads in 1.4s; T2I gen 1024² in ~107–119s cold-settling (per-step warmup curve descends to ~3s/it — steady state ~15–30s); `image=`-conditioned edit works (28s); peak VRAM ~8.0–9.5GB — fits the 4080.
- **Regression:** 106/106 backend tests pass on the new deps; real Standard job failed at texture (the trust_remote_code issue) then its resume child completed **all stages** post-fix.

**Architecture refinements discovered during gate:**

- **Worker contract unchanged.** The image-equality check validates `request.image` against the *child job's own* persisted `job.image`, and `mesh_token` resolves inside the child's own job dir. A retexture child (`resume_from="texture"` + `prepare_resume` checkpoint import + styled image as its `job.image`) passes all existing validation — no narrow exception needed. `parameters["image"]` must be updated to the styled image on the child.
- **T2I config stays out of `runtime_config`.** That dict feeds `checkpoint_signature` — adding keys would invalidate every existing job's checkpoints (breaking resume + retexture lineage on history). T2I provenance (`t2i_model`, `t2i_seed`) is recorded as job input metadata instead, and surfaced in health via a separate field.
- **Edit reference = client viewport capture.** The three.js viewer already renders the exact mesh — canvas capture gives a geometry-matched edit reference with zero new backend render dependencies. Server-side mesh render remains a possible later addition.

## Phase 1: Runtime plumbing

- Add env config: `HUNYFORGE_T2I_MODEL_PATH` (default `/models/FLUX.2-klein-4B`), `HUNYFORGE_T2I_ENABLED`, `HUNYFORGE_T2I_MAX_PROMPT_CHARS`.
- Extend `worker_runtime.py` with a lazily-loaded T2I capability following the established lifecycle: load on demand → generate → release in `finally`. `enable_model_cpu_offload()` so idle footprint is minimal.
- New telemetry operations: `reference.t2i_model_loading`, `reference.t2i_inference`, `reference.mesh_render`; stage name `reference` added to stage ordering and the public stage contract.
- `runtime_config` reports `t2i_model` + `t2i_revision` for provenance.
- Download and pin `black-forest-labs/FLUX.2-klein-4B` at a fixed revision; mount read-only like the Hunyuan models.

Acceptance: a worker-level T2I call runs inside the container, releases memory, and emits telemetry.

## Phase 2: API contracts

- `POST /api/reference-preview` — `{prompt, seed?, scaffold?}` → `{image_b64, seed}`. Routed through the worker's serialized lock; returns 409 "worker busy" when a job stage holds the GPU (preview is never allowed to contend). Prompt validated for length/non-empty; response is ephemeral (not a job).
- Job creation accepts `input_mode: "image" | "text"`. Text mode requires the client to have obtained a reference via the preview endpoint first — the job payload carries the generated image bytes plus `prompt`, `t2i_seed`, `t2i_model` recorded on the job for audit. `job.image` = generated image (hash binding unchanged).
- `POST /api/jobs/{job_id}/retexture` — `{prompt, t2i_seed?, conditioning?}` → child job: `parent_job_id` set, `input_mode: "retexture"`, texture stage only. Requires the parent's `white-mesh.glb` checkpoint (hash-verified via existing mesh-token rules, extended to accept cross-job lineage).
- Worker validation change: image-equality check gains a narrow exception — `input_mode == "retexture"` jobs may carry an image differing from the parent, provided parent linkage and the checkpoint token validate. All other checks unchanged.
- `JobStatus`/job records expose `input_mode`, `prompt`, `t2i_seed`, `t2i_model`, `parent_job_id` (existing), and `resume_from` reuse for the retexture child.

Acceptance: API tests cover preview, text-mode job creation, retexture lineage, validation rejections (empty prompt, missing checkpoint, image mismatch on non-retexture jobs).

## Phase 3: Worker stages

- **Reference stage (text-to-3D):** prompt (+ scaffold) → `Flux2KleinPipeline` T2I → PNG written to `inputs/reference-image.png` in the job dir → proceeds into the normal shape stage with that image.
- **Retexture reference stage:** render the checkpointed `white-mesh.glb` headless to a view image (reuse the paint pipeline's renderer — `custom_rasterizer`/`DifferentiableRenderer` are already compiled in the image — or an Open3D offscreen render as fallback) → `Flux2KleinPipeline(prompt=style_prompt, image=mesh_render)` → styled reference matching geometry → texture stage on the mesh token.
- Conditioning parameter (edit strength / guidance) is a named, validated setting with a conservative default.
- Cancellation between reference and shape/texture stages must leave a clean failed/partial state per existing stage semantics.

Acceptance: unit tests with a stubbed pipeline (no weights on host) verify request routing, mesh-render invocation, lineage validation, and telemetry; container contract test verifies the real pipeline class is used.

## Phase 4: Frontend

Per the design canvas (Generate-tab input-mode switch, Edit tab, before/after viewer):

- **Generate tab:** `Image upload | Text prompt` segmented control. Text mode: prompt textarea with counter, "Generate reference" action, preview thumbnail, regenerate-seed control, scaffold toggle (default on), T2I model display in Advanced. Submit disabled until a reference image exists (uploaded or generated).
- **Edit tab:** new inspector tab enabled when the selected job has a `white-mesh.glb` checkpoint; style prompt, conditioning control, generated-reference preview, "Run retexture" → child job.
- **Viewer:** Before/After toggle for retexture children (parent textured vs child textured), reusing the White/Textured toggle pattern.
- **Job list:** input-type badge (image/text/retexture) and lineage chip extending the "resumed from" pattern.
- **Activity strip:** `reference` stage renders through the existing stage-aware display; T2I previews show a lightweight busy state, never a fake job.
- Existing accessibility rules apply: 44px targets, inert drawer, focus movement, aria-live activity, tri-state connection indicator.

Acceptance: frontend tests cover mode switching, preview lifecycle (success/failure/busy), retexture enablement rules, before/after toggle, lineage display, and prompt validation. `npm run build` clean.

## Phase 5: Verification

- Backend suite green (host) + contract suite green (container) + regression Standard run post-bump. ✅ 136 backend tests pass (19 new T2I/API tests), 42 frontend tests pass, `tsc` + Vite build clean.
- Real GPU: one text-to-3D job end-to-end — ✅ `d9ba5f4c` (Draft preset, `input_mode=text`, prompt "a red off-road buggy with four large wheels", all 8 artifacts, `complete`). Reference preview on real GPU verified twice (~95–110s incl. cold warmup; steady state faster). Retexture child job on real GPU pending.
- Unity smoke test re-run on a text-to-3D artifact (`unity-smoke-test/` headless report must stay green). Pending.
- Evidence added to `VERIFICATION.md`; `STATUS.md` updated; this plan's phases marked with results. Pending — container currently runs layer-patched code; Docker rebuild still owed for durability.

## Non-goals

- No text-guided **geometry** editing ("add a tail") — Hunyuan3D-2.1 cannot do it; would require a different model family.
- No LoRA fine-tuning of klein, no multi-reference character-consistent editing (klein supports it; out of scope until the basic flows land).
- No support for `FLUX.2-klein-9B` (non-commercial license) or base variants (50-step latency).

## Open decisions

- Preview-under-load policy: strict 409 vs. short queue. Spec above chooses 409 to keep the GPU contract absolute; revisit if it feels hostile in use.
- Edit-strength default for retexture: start conservative (preserve silhouette), tune against real outputs.
- Whether the mesh render for retexture uses the textured GLB (better appearance cue) or the white mesh (cleaner geometry cue) — decide empirically in Phase 3; both are cheap to try once the render path exists.
