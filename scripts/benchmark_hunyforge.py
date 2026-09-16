import argparse
import base64
import hashlib
import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request('http://127.0.0.1:8081' + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-job', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--preset', choices=['draft', 'standard', 'final'])
    parser.add_argument('--shape-only', action='store_true')
    parser.add_argument('--parameters', type=json.loads, default={})
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite benchmark evidence')
    health = request('/health')
    if not health.get('runtime_ready'):
        raise SystemExit('Runtime is not ready')
    if any(job['stage'] not in {'complete', 'failed', 'cancelled'} for job in request('/api/jobs')):
        raise SystemExit('An existing job is active; refusing concurrent GPU work')
    source = request('/api/jobs/' + args.source_job)
    if not source.get('image'):
        source.update(request('/api/jobs/' + args.source_job + '/input'))
    payload = {key: source.get(key) for key in ('backend', 'seed', 'image', 'control_type')}
    payload['texture'] = not args.shape_only
    payload['parameters'] = {} if args.preset else {key: value for key, value in source.get('parameters', {}).items() if key not in {'image', 'control_data', 'seed', 'control_type'}}
    payload['parameters'].update(args.parameters)
    if args.preset:
        payload['preset'] = args.preset
    raw_image = base64.b64decode((source.get('image') or '').split(',')[-1])
    evidence = {'label': args.label, 'started_at': datetime.now(timezone.utc).isoformat(), 'source_job_id': args.source_job, 'input_sha256': hashlib.sha256(raw_image).hexdigest(), 'request': {key: value for key, value in payload.items() if key != 'image'}, 'health_before': health, 'sampling_interval_seconds': 5, 'memory_scope': 'nvidia-smi whole-device used MiB; sampled, includes desktop and other processes; not allocator peak', 'stage_time_scope': 'API-observed transition times; up to one polling interval uncertainty; worker telemetry is authoritative when available', 'samples': [], 'transitions': [], 'status': 'starting'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    job = request('/api/jobs', payload)
    evidence['job_id'] = job['id']
    print('Benchmark job: ' + job['id'], flush=True)
    last_stage = None
    while True:
        elapsed = time.monotonic() - start
        try:
            job = request('/api/jobs/' + evidence['job_id'])
            if job['stage'] != last_stage:
                last_stage = job['stage']
                evidence['transitions'].append({'stage': last_stage, 'observed_elapsed_seconds': elapsed})
                print(f'{elapsed:.1f}s {last_stage}', flush=True)
            sample = {'elapsed_seconds': elapsed, 'stage': last_stage}
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
                memory, utilization = result.stdout.strip().splitlines()[0].split(',')
                sample.update(device_used_mib=int(memory.strip()), utilization_percent=int(utilization.strip()))
            except (OSError, ValueError, subprocess.SubprocessError):
                sample['memory_unavailable'] = True
            evidence['samples'].append(sample)
            evidence['status'] = job['stage']
            evidence['job'] = {key: value for key, value in job.items() if key not in {'image', 'parameters'}}
            evidence['resolved_parameters'] = {key: value for key, value in job.get('parameters', {}).items() if key not in {'image', 'control_data'}}
        except (OSError, ValueError) as error:
            evidence.setdefault('poll_errors', []).append({'elapsed_seconds': elapsed, 'type': type(error).__name__})
        evidence['observed_elapsed_seconds'] = time.monotonic() - start
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        temporary.replace(args.output)
        if evidence['status'] in {'complete', 'failed', 'cancelled'}:
            print('Evidence: ' + str(args.output), flush=True)
            return 0 if evidence['status'] == 'complete' else 1
        if elapsed > 10800:
            raise SystemExit('Observation timeout; backend work was NOT cancelled. Inspect the recorded job before starting another run.')
        time.sleep(5)


if __name__ == '__main__':
    raise SystemExit(main())
