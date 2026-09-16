// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  API_URL,
  ConnectionState,
  GenerationSettings,
  JobUpdate,
  PresetEntry,
  apiError,
  generatedArtifactName,
  inputModeBadge,
  isTerminalStage,
  presetEstimate,
  readyArtifactNames,
  safeAssetUrl,
  selectPreviewName,
  settingsError,
  submissionError,
  warningText,
  watchJob,
} from './job-utils';

const BASE_SETTINGS: GenerationSettings = {
  num_inference_steps: 30,
  guidance_scale: 5,
  octree_resolution: 384,
  num_chunks: 8000,
  remove_background: true,
  render_size: 768,
  texture_size: 1024,
  max_num_view: 4,
  texture_inference_steps: 15,
  texture_guidance_scale: 3,
  multiview_resolution: 512,
  face_count: 40000,
  generate_lods: true,
  generate_collision: true,
  collision_mode: 'box',
  unity_mode: 'fast',
};

function jobFixture(overrides: Partial<JobUpdate> = {}): JobUpdate {
  return {
    id: 'job-1',
    stage: 'queued',
    progress: 0,
    backend: 'demo',
    seed: 48291,
    texture: true,
    preset: 'standard',
    parameters: { ...BASE_SETTINGS },
    artifacts: [],
    artifact_readiness: {},
    available_actions: [],
    resume_stage: null,
    stage_status: {},
    timings: {},
    current_operation: null,
    operation_progress: null,
    started_at: null,
    finished_at: null,
    created_at: '2026-09-13T10:00:00Z',
    updated_at: '2026-09-13T10:00:00Z',
    parent_job_id: null,
    resume_from: null,
    failed_stage: null,
    peak_vram_mb: null,
    error_code: null,
    error_message: null,
    ...overrides,
  };
}

class MockEventSource {
  static instances: MockEventSource[] = [];
  listeners: Record<string, ((event: MessageEvent) => void)[]> = {};
  onerror: (() => void) | null = null;
  closed = false;
  constructor(public url: string) {
    MockEventSource.instances.push(this);
  }
  addEventListener(type: string, fn: (event: MessageEvent) => void) {
    (this.listeners[type] ??= []).push(fn);
  }
  close() {
    this.closed = true;
  }
  emit(job: object) {
    for (const fn of this.listeners['job'] ?? []) fn({ data: JSON.stringify(job) } as MessageEvent);
  }
  fail() {
    this.onerror?.();
  }
}

describe('job utilities', () => {
  it('selects the artifact matching the texture setting', () => {
    expect(generatedArtifactName(true)).toBe('textured-mesh.glb');
    expect(generatedArtifactName(false)).toBe('white-mesh.glb');
  });

  it('recognizes terminal job stages', () => {
    expect(isTerminalStage('complete')).toBe(true);
    expect(isTerminalStage('failed')).toBe(true);
    expect(isTerminalStage('cancelled')).toBe(true);
    expect(isTerminalStage('generating_shape')).toBe(false);
    expect(isTerminalStage('cancelling')).toBe(false);
    expect(isTerminalStage('cancellation_requested')).toBe(false);
  });

  it('merges readiness map and legacy artifact list', () => {
    const job = jobFixture({ artifact_readiness: { 'white-mesh.glb': true, 'textured-mesh.glb': false }, artifacts: ['artifacts/unity-lod0.glb'] });
    expect(readyArtifactNames(job).sort()).toEqual(['unity-lod0.glb', 'white-mesh.glb']);
    expect(readyArtifactNames(null)).toEqual([]);
  });

  it('rejects invalid setting combinations', () => {
    expect(settingsError(BASE_SETTINGS)).toBeNull();
    expect(settingsError({ ...BASE_SETTINGS, texture_size: 1024, render_size: 768 })).toBeNull();
    expect(settingsError({ ...BASE_SETTINGS, unity_mode: 'fast', generate_collision: true, collision_mode: 'convex_hull' })).toContain('convex hull');
    expect(settingsError({ ...BASE_SETTINGS, unity_mode: 'full', generate_collision: true, collision_mode: 'convex_hull' })).toBeNull();
  });

  it('guards real Hunyuan inputs without blocking demo mode', () => {
    expect(submissionError('demo', null, null, null, BASE_SETTINGS)).toBeNull();
    expect(submissionError('hunyuan3d-2.1', null, null, null, BASE_SETTINGS)).toContain('reference image');
    expect(submissionError('hunyuan3d-omni', 'image', 'bbox', null, BASE_SETTINGS)).toContain('Omni control');
  });

  it('reports null evidence as unbenchmarked with conservative warnings', () => {
    const entry: PresetEntry = {
      label: 'Final',
      description: '',
      expected_duration_seconds: null,
      expected_peak_vram_mb: null,
      settings: { ...BASE_SETTINGS, texture_size: 2048, max_num_view: 6 },
    };
    const estimate = presetEstimate(entry);
    expect(estimate.duration).toBe('Not benchmarked');
    expect(estimate.vram).toBe('Not benchmarked');
    expect(estimate.warnings.join(' ')).toContain('16 GB');
    expect(presetEstimate(null).duration).toBe('Not benchmarked');
  });

  it('tracks current settings and suppresses texture warnings for shape-only runs', () => {
    const entry: PresetEntry = {
      label: 'Standard',
      description: '',
      expected_duration_seconds: 120,
      expected_peak_vram_mb: 8192,
      settings: BASE_SETTINGS,
    };
    const exact = presetEstimate(entry, { ...BASE_SETTINGS }, true);
    expect(exact.duration).toBe('~120 s');
    expect(exact.vram).toContain('8.0 GB');
    const edited = presetEstimate(entry, { ...BASE_SETTINGS, num_inference_steps: 99 }, true);
    expect(edited.duration).toBe('Not benchmarked');
    expect(edited.vram).toBe('Not benchmarked');
    const shapeOnly = presetEstimate(entry, { ...BASE_SETTINGS, texture_size: 2048, max_num_view: 6 }, false);
    expect(shapeOnly.warnings.join(' ')).not.toContain('16 GB');
  });

  it('selects the preview artifact by readiness and explicit choice', () => {
    const running = jobFixture({
      stage: 'generating_textures',
      texture: true,
      artifact_readiness: { 'white-mesh.glb': true },
    });
    expect(selectPreviewName(running, 'auto')).toBe('white-mesh.glb');
    expect(selectPreviewName(running, 'white')).toBe('white-mesh.glb');
    expect(selectPreviewName(running, 'textured')).toBe('white-mesh.glb');
    const textured = jobFixture({
      stage: 'complete',
      texture: true,
      artifact_readiness: { 'white-mesh.glb': true, 'textured-mesh.glb': true },
    });
    expect(selectPreviewName(textured, 'auto')).toBe('textured-mesh.glb');
    expect(selectPreviewName(textured, 'white')).toBe('white-mesh.glb');
    expect(selectPreviewName(textured, 'textured')).toBe('textured-mesh.glb');
    const shapeOnly = jobFixture({ texture: false, artifact_readiness: { 'white-mesh.glb': true } });
    expect(selectPreviewName(shapeOnly, 'auto')).toBe('white-mesh.glb');
    expect(selectPreviewName(jobFixture(), 'auto')).toBeNull();
  });

  it('restricts viewer asset URLs to same-origin job artifacts and safe embedded data', () => {
    const source = `${API_URL}/api/jobs/job-9/artifacts/white-mesh.glb`;
    expect(safeAssetUrl(`${API_URL}/api/jobs/job-9/artifacts/detail.bin`, source)).toBe(`${API_URL}/api/jobs/job-9/artifacts/detail.bin`);
    expect(safeAssetUrl('detail.bin', source)).toBeNull();
    expect(safeAssetUrl('data:image/png;base64,AAAA', source)).toBe('data:image/png;base64,AAAA');
    expect(safeAssetUrl('data:image/jpeg;base64,AAAA', source)).toBe('data:image/jpeg;base64,AAAA');
    expect(safeAssetUrl('data:application/octet-stream;base64,AAAA', source)).toBe('data:application/octet-stream;base64,AAAA');
    expect(safeAssetUrl('blob:http://127.0.0.1:5173/abc', source)).toBe('blob:http://127.0.0.1:5173/abc');
    expect(safeAssetUrl('data:image/svg+xml;base64,AAAA', source)).toBeNull();
    expect(safeAssetUrl('data:text/html;base64,AAAA', source)).toBeNull();
    expect(safeAssetUrl('//evil.example/x.glb', source)).toBeNull();
    expect(safeAssetUrl('http://evil.example/x.glb', source)).toBeNull();
    expect(safeAssetUrl('https://evil.example/x.glb', source)).toBeNull();
    expect(safeAssetUrl(`http://user@127.0.0.1:8081/api/jobs/job-9/artifacts/x.glb`, source)).toBeNull();
    expect(safeAssetUrl(`${API_URL}/api/jobs/other-job/artifacts/x.glb`, source)).toBeNull();
    expect(safeAssetUrl(`${API_URL}/health`, source)).toBeNull();
    expect(safeAssetUrl('javascript:alert(1)', source)).toBeNull();
  });

  it('rejects nested asset URLs when the source document itself is remote', () => {
    const remote = 'http://evil.example/api/jobs/job-9/artifacts/white-mesh.glb';
    expect(safeAssetUrl('http://evil.example/api/jobs/job-9/artifacts/detail.bin', remote)).toBeNull();
    expect(safeAssetUrl(`${API_URL}/api/jobs/job-9/artifacts/detail.bin`, remote)).toBeNull();
    expect(safeAssetUrl('data:image/png;base64,AAAA', remote)).toBeNull();
  });

  it('parses FastAPI detail strings and validation arrays', async () => {
    const text = await apiError(new Response(JSON.stringify({ detail: 'Worker busy' }), { status: 409 }));
    expect(text.message).toBe('Worker busy');
    const list = await apiError(new Response(JSON.stringify({ detail: [{ loc: ['body', 'seed'], msg: 'Field required' }] }), { status: 422 }));
    expect(list.message).toContain('Field required');
    const plain = await apiError(new Response('oops', { status: 500 }));
    expect(plain.message).toContain('500');
  });

  it('normalizes warning payloads', () => {
    expect(warningText('plain')).toBe('plain');
    expect(warningText({ severity: 'warning', message: 'object' })).toBe('object');
  });

  it('labels text and retexture input modes for history badges', () => {
    expect(inputModeBadge('text')?.label).toBe('TXT');
    expect(inputModeBadge('retexture')?.label).toBe('RETEX');
    expect(inputModeBadge('image')).toBeNull();
    expect(inputModeBadge(undefined)).toBeNull();
  });
});

describe('watchJob', () => {
  beforeEach(() => {
    MockEventSource.instances = [];
    vi.stubGlobal('EventSource', MockEventSource);
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(jobFixture({ stage: 'generating_shape', updated_at: '2026-09-13T10:00:05Z' })), { status: 200 })));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('reports a durable degraded state after SSE error while polling continues', async () => {
    vi.useFakeTimers();
    const states: ConnectionState[] = [];
    const received: JobUpdate[] = [];
    const stop = watchJob('job-1', job => received.push(job), state => states.push(state));
    await vi.advanceTimersByTimeAsync(0);
    expect(states).toEqual(['live']);
    expect(received).toHaveLength(1);
    MockEventSource.instances[0].fail();
    expect(states.at(-1)).toBe('degraded');
    expect(MockEventSource.instances[0].closed).toBe(true);
    await vi.advanceTimersByTimeAsync(5100);
    expect(states.at(-1)).toBe('degraded');
    stop();
  });

  it('ignores stale updates and stops after teardown', async () => {
    const received: JobUpdate[] = [];
    const stop = watchJob('job-1', job => received.push(job), () => undefined);
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(received).toHaveLength(1);
    MockEventSource.instances[0].emit(jobFixture({ stage: 'generating_textures', updated_at: '2026-09-13T10:00:01Z' }));
    expect(received).toHaveLength(1);
    MockEventSource.instances[0].emit(jobFixture({ stage: 'generating_textures', updated_at: '2026-09-13T10:00:10Z' }));
    expect(received).toHaveLength(2);
    stop();
    MockEventSource.instances[0].emit(jobFixture({ stage: 'complete', updated_at: '2026-09-13T10:00:20Z' }));
    expect(received).toHaveLength(2);
  });

  it('stops polling entirely after teardown', async () => {
    vi.useFakeTimers();
    const received: JobUpdate[] = [];
    const fetchMock = vi.mocked(fetch);
    const stop = watchJob('job-1', job => received.push(job), () => undefined);
    await vi.advanceTimersByTimeAsync(0);
    expect(received).toHaveLength(1);
    stop();
    const calls = fetchMock.mock.calls.length;
    await vi.advanceTimersByTimeAsync(25000);
    expect(fetchMock.mock.calls.length).toBe(calls);
    expect(received).toHaveLength(1);
  });

  it('times out a hung poll and recovers on the next cycle', async () => {
    vi.useFakeTimers();
    let hang = false;
    vi.stubGlobal('fetch', vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      if (hang) {
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        });
      }
      return new Response(JSON.stringify(jobFixture({ stage: 'generating_shape', updated_at: '2026-09-13T10:00:05Z' })), { status: 200 });
    }));
    const states: ConnectionState[] = [];
    const received: JobUpdate[] = [];
    const stop = watchJob('job-1', job => received.push(job), state => states.push(state));
    await vi.advanceTimersByTimeAsync(0);
    expect(received).toHaveLength(1);
    hang = true;
    await vi.advanceTimersByTimeAsync(5000);
    await vi.advanceTimersByTimeAsync(11000);
    expect(states.at(-1)).toBe('down');
    hang = false;
    await vi.advanceTimersByTimeAsync(21000);
    expect(states.at(-1)).toBe('live');
    expect(received.length).toBeGreaterThan(1);
    stop();
  });

  it('keeps polling when EventSource is unavailable or fails to construct', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('EventSource', undefined);
    const received: JobUpdate[] = [];
    const stop = watchJob('job-1', job => received.push(job), () => undefined);
    await vi.advanceTimersByTimeAsync(0);
    expect(received).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(5000);
    expect(received.length).toBeGreaterThan(1);
    stop();
  });

  it('ignores events emitted on a closed event source', async () => {
    const received: JobUpdate[] = [];
    const states: ConnectionState[] = [];
    const stop = watchJob('job-1', job => received.push(job), state => states.push(state));
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(received).toHaveLength(1);
    MockEventSource.instances[0].fail();
    MockEventSource.instances[0].emit(jobFixture({ stage: 'complete', updated_at: '2026-09-13T10:00:09Z' }));
    expect(received).toHaveLength(1);
    stop();
  });

  it('stops watching when a terminal job arrives', async () => {
    const received: JobUpdate[] = [];
    const stop = watchJob('job-1', job => received.push(job), () => undefined);
    await new Promise(resolve => setTimeout(resolve, 0));
    MockEventSource.instances[0].emit(jobFixture({ stage: 'complete', updated_at: '2026-09-13T10:00:30Z' }));
    expect(received.at(-1)?.stage).toBe('complete');
    expect(MockEventSource.instances[0].closed).toBe(true);
    stop();
  });
});
