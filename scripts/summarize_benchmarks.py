import argparse
import json
from datetime import datetime
from pathlib import Path


def summarize(path):
    evidence = json.loads(path.read_text(encoding='utf-8'))
    job = evidence.get('job', {})
    timings = job.get('timings', {})
    elapsed = timings.get('elapsed_seconds_total')
    scope = 'monotonic pipeline execution, includes model loading and export'
    if elapsed is None and job.get('created_at') and job.get('updated_at'):
        elapsed = (datetime.fromisoformat(job['updated_at'].replace('Z', '+00:00')) - datetime.fromisoformat(job['created_at'].replace('Z', '+00:00'))).total_seconds()
        scope = 'legacy job created-to-final-update wall-clock span; models already resident before submission'
    device_samples = [sample['device_used_mib'] for sample in evidence.get('samples', []) if 'device_used_mib' in sample]
    stages = {name: record for name, record in timings.items() if isinstance(record, dict) and 'elapsed_seconds' in record}
    return {'evidence_file': str(path), 'label': evidence['label'], 'job_id': evidence['job_id'], 'status': evidence['status'], 'input_sha256': evidence['input_sha256'], 'elapsed_seconds': elapsed, 'elapsed_scope': scope, 'sampled_whole_device_peak_mib': max(device_samples) if device_samples else None, 'whole_device_memory_scope': evidence.get('memory_scope'), 'preset': job.get('preset'), 'texture': job.get('texture'), 'parameters': evidence.get('resolved_parameters'), 'runtime_config': job.get('runtime_config'), 'stages': stages, 'error': job.get('error_message'), 'artifact_readiness': job.get('artifact_readiness', {}), 'limitations': ['Single reference-image run, not a population estimate or quality score.', 'Original worker ignored requested seed and shape parameters; legacy persisted parameters are not actual inference settings.', 'Original warm resident models and enhanced lazy loading have different startup conditions.', 'Device sampling includes non-worker usage; allocator and whole-device peaks are distinct.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path, nargs='+')
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    baseline = summarize(args.baseline) if args.baseline else None
    summaries = []
    for path in args.evidence:
        result = summarize(path)
        if baseline and baseline['status'] == 'complete' and result['status'] == 'complete' and baseline['input_sha256'] == result['input_sha256'] and baseline['elapsed_seconds'] and result['elapsed_seconds']:
            result['reference_run_time_reduction_percent'] = 100 * (1 - result['elapsed_seconds'] / baseline['elapsed_seconds'])
            result['reference_run_speed_ratio'] = baseline['elapsed_seconds'] / result['elapsed_seconds']
            result['baseline_job_id'] = baseline['job_id']
        summaries.append(result)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
