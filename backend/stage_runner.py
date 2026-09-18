"""Single-stage subprocess entry point.

Runs one worker stage in a fresh interpreter so process exit returns all model
RSS (anonymous heap, CUDA host caches, native arenas) to the kernel. The
long-lived uvicorn supervisor in backend.hunyuan_worker spawns one of these
per request instead of running stages in-process; see OOM-ISSUES.md.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def run_stage(spec: dict, engine, slot: Path) -> dict:
    from .hunyuan_worker import PreviewRequest, WorkerRequest
    kind = spec.get("kind")
    if kind == "generate":
        artifact = engine.generate(WorkerRequest.model_validate(spec["payload"]))
        return {"status": "ok", "artifact": str(artifact)}
    if kind == "preview":
        png, records = engine.generate_preview(PreviewRequest.model_validate(spec["payload"]))
        artifact = slot / "preview.png"
        artifact.write_bytes(png)
        timings = {name: round(record.get("elapsed_seconds") or 0, 2) for name, record in records.items()}
        return {"status": "ok", "artifact": str(artifact), "timings": timings}
    raise ValueError(f"unknown stage kind: {kind!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    request_path = Path(args.request)
    result_path = Path(args.result)
    try:
        spec = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"stage_runner: unreadable request {request_path}: {e}", file=sys.stderr)
        return 2
    from .worker_runtime import RuntimeEngine
    root = Path(os.environ.get("HUNYFORGE_DATA_ROOT", Path.cwd() / "data"))
    engine = RuntimeEngine(root)
    try:
        result = run_stage(spec, engine, request_path.parent)
    except Exception as e:
        traceback.print_exc()
        result = {"status": "error", "error_type": type(e).__name__, "error": str(e)}
    result_path.write_text(json.dumps(result, default=str), encoding="utf-8")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
