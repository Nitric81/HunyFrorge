# HunyForge Worker OOM Kills — Analysis

Status: **Fix implemented, awaiting on-machine validation** · Documented 2026-09-18
Context: Ironhound asset-generation batch — first sustained multi-job use of the stack.

> **Update 2026-09-18 (fix landed):** the doc's "models never evicted"
> assumption was wrong — `_release_models()` already runs after every stage;
> the ~14 GiB residual is *private anonymous heap* the kernel never reclaims
> from a live process (glibc arenas, CUDA host caches, native libs). Fix is
> therefore process-boundary isolation: `hunyuan_worker` is now a supervisor
> that runs each stage via `python -m backend.stage_runner` in a fresh
> subprocess (`HUNYFORGE_STAGE_ISOLATION=1` default, `0` = in-process
> fallback). A memory admission gate (`HUNYFORGE_MIN_AVAILABLE_MB`,
> default 12288) rejects work with `insufficient_memory` instead of dying
> mid-stage. Child SIGKILL → `WORKER_OOM_KILLED` in the job record, the
> supervisor survives, and the container no longer restarts. `/health` now
> reports `memory` (RSS + VM available) and `active_stage`. `malloc_trim`
> added to `_release_models`. Tests: `backend/test_stage_isolation.py` +
> `_fail` code coverage in `test_orchestration.py`. Ops: rebuild the image;
> keep the restart-per-job workaround until validated, then simplify
> `queue-ironhound-assets.ps1`.

## TL;DR

The `hunyforge` container's Hunyuan worker subprocess is being SIGKILLed by the
Linux OOM killer after roughly **one heavy job per fresh container**. There is no
container memory limit; the ceiling is the WSL2 VM memory cap (24 GB of a 32 GB
host). Models stay resident after a job, host RSS ratchets upward across
consecutive jobs, and the kill typically lands mid-pipeline on the second heavy
job — surfaced to clients only as `RemoteDisconnected` / `PIPELINE_ERROR`.

**Proven workaround:** `docker restart hunyforge` before each heavy job and wait
for `runtime_ready`. Went 5/5 clean after adopting it.

---

## Environment

| Item | Value |
|---|---|
| Host RAM | 32 GB (31.8 GB physical) |
| GPU | RTX 4080, 16 GB VRAM — **not the bottleneck** |
| WSL2 cap | `memory=24GB`, `swap=12GB` (`%USERPROFILE%\.wslconfig`) |
| Docker-visible memory | 23.47 GiB (`docker system info`) |
| Container `mem_limit` | **none** (`MemLimit=0`) |
| Restart policy | `unless-stopped` (`docker-compose.yml:48`) |
| Container idle with models loaded | **~14.8 GiB** (63% of Docker memory) |
| Worker peak RSS (2nd consecutive job) | **~21 GB** observed before kill |

## Architecture context

The `hunyforge` container co-hosts, in one memory space:

- FastAPI app (`:8081`) — job orchestration, artifacts, health
- Hunyuan3D-2.1 worker — spawned by `docker/entrypoint.sh` as a uvicorn
  subprocess (`backend.hunyuan_worker:app`, `:8082`), kept as a long-lived
  process across jobs
- FLUX.2-klein-4B text-to-image model (`/api/reference-preview`)
- rembg/u2net background removal, trimesh/numpy geometry + texture baking
  (`HUNYUAN_RENDER_SIZE=1024`, `HUNYUAN_TEXTURE_SIZE=2048`)

`hunyforge-hunyforge-multiview-1` is a **separate** container and was up 2 days
throughout — never OOM'd. It is only loaded for `multi-image` input mode, which
the batch did not use. This supports the "per-container model co-residency" root
cause rather than a stack-wide problem.

Models load lazily on first request ("models stay unloaded until first
request") and are **never evicted**. After one job, ~15 GiB stays resident;
the next heavy job allocates shape-generation + texture-baking buffers on top
of that baseline.

## Observed failures — timeline

Container log timestamps are UTC; local is UTC-5. All evidence from
`docker logs hunyforge -t`.

| Local time | Event |
|---|---|
| Sep 17 ~11:02 | Container start (1st "Starting HunyForge API" line) |
| Sep 18 ~07:0x | Truckstop job `fccbf721` — **success** (first job on fresh container) |
| ~08:1x | First batch attempt: 6 reference-previews submitted against a container still holding truckstop's models; FLUX load pushed RSS over the cap |
| **08:20:23** | **Kill #1** — `14 Killed ( cd /opt/hunyuan && exec python3.10 -m uvicorn backend.hunyuan_worker:app ... )` |
| 08:20:28 | Container auto-restarts; all 6 queued requests fail 500/503 |
| ~08:3x | Hauler job `aa6ac746` — **success** (first job on fresh container) |
| ~08:42 | Kiosk attempt 1, shape stage → **Kill #2** at 08:42:50 — job ends `PIPELINE_ERROR` 25%, `Remote end closed connection without response` |
| 08:45:51 | Scripted restart → kiosk retry — **success** |
| 08:57:09 | Restart → hub-mast — **success** |
| 09:09:28 | Restart → wreck — **success** |
| 09:20:37 | Restart → dog — **success** |
| 09:51:43 | Restart → door-slab — **success** |

Eight "Starting HunyForge API" lines in 48h = 1 initial + 2 crash restarts +
5 scripted restarts.

## Signature — how to recognize it

Worker-side (authoritative):

```
/app/docker/entrypoint.sh: line 40:    14 Killed    ( cd /opt/hunyuan && exec python3.10 -m uvicorn backend.hunyuan_worker:app --host 127.0.0.1 --port 8082 > "$worker_fifo" 2>&1 )
```

API-side (what the job record shows — symptom, not cause):

```
http.client.RemoteDisconnected: Remote end closed connection without response
```

`docker inspect` caveat: `State.OOMKilled` reads **false** because the kernel
killed the worker *subprocess* (PID 14) inside the container, not the container
cgroup itself. The container still restarts because the entrypoint exits when
its child dies, and `restart: unless-stopped` brings it back — which is why
"Starting HunyForge API" reappears in the log 5–10 s after each kill.

Post-kill state: `runtime_ready: false` (models unload on restart) until the
next request lazily reloads them — reload takes ~1–2 min.

## Root cause

1. **Unbounded memory growth**: no `mem_limit`/`deploy.resources` on the
   service — processes grow until the WSL2 VM-wide OOM killer picks the fattest
   process (the worker).
2. **Resident-model accumulation**: Hunyuan + FLUX + rembg + torch context all
   stay loaded between jobs (~15 GiB steady state). Python GC + the CUDA
   caching allocator do not return host RSS; each additional job peaks higher.
3. **Co-residency**: FLUX lives in the same container as the worker — the first
   kill happened when FLUX loaded on top of an already-warm worker.
4. **No admission control**: the worker accepts a new heavy stage without
   checking available memory, so the failure lands mid-pipeline instead of
   being rejected up front.

## False-failure modes (operational gotchas)

- **HTTP client timeout ≠ job failure.** During GPU stages the API can take
  >15 s to respond; a client timeout while the job is still `running` is
  transient. Poll with long timeouts and treat request errors as retryable.
- **`RemoteDisconnected` / `PIPELINE_ERROR` is usually this OOM.** Check
  `docker logs hunyforge | grep Killed` before assuming a network blip.
- **500/503 right after a kill** = container restarting / models reloading.
  Wait for `runtime_ready`, then retry — do not resubmit into the outage window.
- **Job IDs survive the crash** — a job killed mid-stage stays in the store as
  `PIPELINE_ERROR`; it does not resume on restart. Resubmit a new job.

## Workaround (validated)

`ironhound-assets\queue-ironhound-assets.ps1` implements:

1. `docker restart hunyforge` before **each** asset
2. Poll `/health` until `runtime_ready == true`
3. One reference-preview → one job → poll (long timeouts, transient-error
   tolerance) → download → import → next asset

Result: 5/5 consecutive assets clean (~10–13 min each end-to-end).

## Proposed fixes

### Short term (ops)

- Keep restart-per-job for any batch orchestration.
- Expose worker RSS in `/health` so clients can see headroom before submitting.

### Medium term (engineering)

- **Model eviction between jobs** — biggest single win. Unload the texture/
  shape pipelines at job completion (`del` refs + `gc.collect()` +
  `torch.cuda.empty_cache()` + free host buffers); reload lazily on next
  request. Returns per-job footprint to fresh-container levels without a
  restart.
- **Memory admission check** — `psutil.virtual_memory()` gate at stage entry;
  fail fast with a clear `insufficient_memory` error instead of dying
  mid-stage. Surface the reason in the job record.
- **Split FLUX into its own service** — the multiview container already proves
  the pattern; removes the largest co-resident model from the worker's memory
  space.
- **Container `mem_limit` + `oom_score_adj`** — a defined threshold converts an
  unpredictable VM-wide kill into a deterministic, attributable failure.
  (Secondary to eviction; a limit alone still loses the in-flight job.)

### Long term

- Surface OOM diagnostics in the job record — today `PIPELINE_ERROR:
  RemoteDisconnected` is indistinguishable from a network fault without
  reading docker logs.
- Consider per-job worker subprocesses (spawn → run → exit) so the OS reclaims
  all RSS between jobs; trades ~seconds of spawn time for isolation.

## Diagnostics cheat sheet

```powershell
# Confirm a kill happened
docker logs hunyforge -t | Select-String "Killed|Starting HunyForge"

# Live memory vs the 23.5 GiB ceiling
docker stats hunyforge --no-stream

# Container-level OOM flag (will NOT catch worker-subprocess kills)
docker inspect hunyforge --format "{{.State.OOMKilled}} {{.RestartCount}}"

# Runtime readiness after a restart
Invoke-RestMethod http://127.0.0.1:8081/health | Select-Object runtime_ready

# WSL2 ceiling
docker system info --format "{{.MemTotal}}"
Get-Content "$env:USERPROFILE\.wslconfig"
```

## Related defects found during the same batch

- `backend/unreal/HunyForgeUnrealSetup.py` — `pick()` matched `"sm_"` inside
  `Collision_SM_*.glb` (sorted first), importing the collision box as LOD0.
  Fixed locally; packages still ship the buggy copy, which `setup-unreal.ps1`
  prefers over the repo copy.
- Windows path `\t` in `ironhound-assets\truckstop-pkg` was interpreted as a
  literal tab by `-ExecutePythonScript` — script silently never ran and the
  wrapper reported a stale `ok:true` report. Avoid `t`-prefixed dir segments;
  use unique report paths per run.
- `UnrealEditor-Cmd` can hang ~20 min after `LogExit` on headless shutdown —
  import reports are already on disk; kill the process rather than waiting.
