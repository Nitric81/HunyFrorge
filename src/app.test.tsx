// @vitest-environment jsdom
import { act, forwardRef, useImperativeHandle } from 'react';
import { createRoot, Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { GenerationSettings, JobUpdate, PresetContract, StageState, StageStateValue } from './job-utils';

(globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('./ThreePreview', () => ({
  default: forwardRef(function MockPreview({ url }: { url: string | null }, ref) {
    useImperativeHandle(ref, () => ({ capture: () => 'data:image/png;base64,bW9ja2NhcHR1cmU=' }), []);
    return <div data-testid="three-preview">{url ?? 'empty'}</div>;
  }),
}));

const DRAFT_SETTINGS: GenerationSettings = {
  num_inference_steps: 10, guidance_scale: 5, octree_resolution: 256, num_chunks: 8000, remove_background: true,
  render_size: 512, texture_size: 1024, max_num_view: 4, texture_inference_steps: 10, texture_guidance_scale: 3,
  multiview_resolution: 512, face_count: 20000, generate_lods: false, generate_collision: false,
  collision_mode: 'box', unity_mode: 'fast',
};
const STANDARD_SETTINGS: GenerationSettings = {
  num_inference_steps: 30, guidance_scale: 5, octree_resolution: 384, num_chunks: 8000, remove_background: true,
  render_size: 768, texture_size: 1024, max_num_view: 4, texture_inference_steps: 15, texture_guidance_scale: 3,
  multiview_resolution: 512, face_count: 40000, generate_lods: true, generate_collision: true,
  collision_mode: 'box', unity_mode: 'fast',
};
const FINAL_SETTINGS: GenerationSettings = {
  ...STANDARD_SETTINGS, num_inference_steps: 50, render_size: 1024, texture_size: 2048, max_num_view: 6,
  collision_mode: 'convex_hull', unity_mode: 'full',
};

const CONTRACT: PresetContract = {
  version: 1,
  default: 'standard',
  presets: {
    draft: { label: 'Draft', description: '', expected_duration_seconds: null, expected_peak_vram_mb: null, settings: DRAFT_SETTINGS },
    standard: { label: 'Standard', description: '', expected_duration_seconds: null, expected_peak_vram_mb: null, settings: STANDARD_SETTINGS },
    final: { label: 'Final', description: '', expected_duration_seconds: null, expected_peak_vram_mb: null, settings: FINAL_SETTINGS },
  },
  settings_schema: {
    properties: {
      num_inference_steps: { type: 'integer', minimum: 1, maximum: 100 },
      guidance_scale: { type: 'number', minimum: 0.1, maximum: 20 },
      octree_resolution: { type: 'integer', enum: [256, 384, 512] },
      num_chunks: { type: 'integer', minimum: 1000, maximum: 20000 },
      remove_background: { type: 'boolean' },
      render_size: { type: 'integer', enum: [512, 768, 1024] },
      texture_size: { type: 'integer', enum: [1024, 2048] },
      max_num_view: { type: 'integer', enum: [4, 6] },
      texture_inference_steps: { type: 'integer', minimum: 1, maximum: 50 },
      texture_guidance_scale: { type: 'number', minimum: 0.1, maximum: 20 },
      multiview_resolution: { type: 'integer', enum: [512, 768] },
      face_count: { type: 'integer', minimum: 1000, maximum: 100000 },
      generate_lods: { type: 'boolean' },
      generate_collision: { type: 'boolean' },
      collision_mode: { type: 'string', enum: ['box', 'convex_hull'] },
      unity_mode: { type: 'string', enum: ['fast', 'full'] },
    },
  },
};

function stageRecord(state: StageStateValue, elapsed = 0): StageState {
  return { state, started_at: null, finished_at: null, elapsed_seconds: elapsed, error: null };
}

function jobFixture(overrides: Partial<JobUpdate> = {}): JobUpdate {
  return {
    id: 'job-00000000-0000-0000-0000-000000000001',
    stage: 'queued',
    progress: 0,
    backend: 'demo',
    seed: 48291,
    texture: true,
    preset: 'standard',
    parameters: { ...STANDARD_SETTINGS },
    artifacts: [],
    artifact_readiness: {},
    available_actions: [],
    resume_stage: null,
    stage_status: {
      shape: stageRecord('waiting'),
      texture: stageRecord('waiting'),
      unity: stageRecord('waiting'),
      validation: stageRecord('waiting'),
    },
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

type Handler = (body: unknown, url: string) => { status?: number; body: unknown };
const calls: { method: string; url: string; body: unknown }[] = [];
let routes: { method: string; match: RegExp; handler: Handler }[] = [];
let jobsById: Record<string, JobUpdate> = {};
let listedJobs: JobUpdate[] = [];

function route(method: string, match: RegExp, handler: Handler) {
  routes.push({ method, match, handler });
}

function defaults() {
  calls.length = 0;
  routes = [];
  jobsById = {};
  listedJobs = [];
  route('GET', /\/api\/presets$/, () => ({ body: CONTRACT }));
  route('GET', /\/health$/, () => ({ body: { inference_mode: 'demo', runtime_ready: true, hunyuan_service_ready: true } }));
  route('GET', /\/api\/projects$/, () => ({ body: [{ id: 'proj-1', name: 'Asset Lab' }] }));
  route('GET', /\/api\/jobs$/, () => ({ body: listedJobs }));
  route('GET', /\/api\/jobs\/([^/]+)$/, (_body, url) => {
    const id = /\/api\/jobs\/([^/]+)$/.exec(url)?.[1] ?? '';
    return jobsById[id] ? { body: jobsById[id] } : { status: 404, body: { detail: 'Job not found' } };
  });
}

beforeEach(() => {
  defaults();
  window.localStorage.clear();
  MockEventSource.instances = [];
  vi.stubGlobal('EventSource', MockEventSource);
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      const body = init?.body ? JSON.parse(String(init.body)) : undefined;
      calls.push({ method, url, body });
      for (const entry of [...routes].reverse()) {
        if (entry.method === method && entry.match.test(url)) {
          const result = entry.handler(body, url);
          return new Response(JSON.stringify(result.body), { status: result.status ?? 200, headers: { 'Content-Type': 'application/json' } });
        }
      }
      return new Response('{}', { status: 404 });
    }),
  );
});

let container: HTMLDivElement;
let root: Root;

async function settle() {
  if (vi.isFakeTimers()) await vi.advanceTimersByTimeAsync(0);
  else await new Promise(resolve => setTimeout(resolve, 0));
}

async function mount() {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root.render(<App />);
    await settle();
  });
}

async function flush() {
  await act(async () => {
    await settle();
  });
}

afterEach(async () => {
  if (root) {
    await act(async () => root.unmount());
  }
  container?.remove();
  vi.unstubAllGlobals();
});

function field(label: string): HTMLElement {
  for (const el of Array.from(container.querySelectorAll('label'))) {
    if (el.textContent?.startsWith(label)) {
      const input = el.querySelector('input,select,textarea');
      if (input) return input as HTMLElement;
    }
  }
  throw new Error(`field not found: ${label}`);
}

function buttonWith(text: string): HTMLButtonElement {
  for (const el of Array.from(container.querySelectorAll('button'))) {
    if (el.textContent?.includes(text)) return el;
  }
  throw new Error(`button not found: ${text}`);
}

async function setText(input: HTMLElement, value: string) {
  const proto = input instanceof HTMLTextAreaElement ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, 'value')!.set!;
  await act(async () => {
    setter.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

async function setSelect(select: HTMLElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value')!.set!;
  await act(async () => {
    setter.call(select, value);
    select.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

async function submitForm() {
  const form = container.querySelector('form') as HTMLFormElement;
  await act(async () => {
    form.requestSubmit();
    await settle();
  });
}

async function pickFile(ariaLabel: string, file: File, expectAccepted = true) {
  const input = container.querySelector(`input[aria-label="${ariaLabel}"]`) as HTMLInputElement;
  expect(input).toBeTruthy();
  await act(async () => {
    Object.defineProperty(input, 'files', { value: [file], configurable: true });
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
  if (!expectAccepted) {
    await flush();
    return;
  }
  for (let attempt = 0; attempt < 20; attempt += 1) {
    await flush();
    if (container.textContent?.includes(file.name)) return;
  }
  throw new Error(`file ${file.name} was not accepted`);
}

const postJobs = () => calls.filter(call => call.method === 'POST' && /\/api\/jobs$/.test(call.url));
const jobCalls = (suffix: string) => calls.filter(call => call.method === 'POST' && call.url.endsWith(suffix));
const previewCalls = () => calls.filter(call => call.method === 'POST' && /\/api\/reference-preview$/.test(call.url));

function enableT2i() {
  route('GET', /\/health$/, () => ({ body: { inference_mode: 'demo', runtime_ready: true, hunyuan_service_ready: true, t2i: { enabled: true, model: 'flux2-klein-4b', model_path: '/models/FLUX.2-klein-4B', max_prompt_chars: 2000 } } }));
}

describe('App', () => {
  it('renders preset contract controls and submits every visible setting with seed zero', async () => {
    route('POST', /\/api\/jobs$/, body => {
      const created = jobFixture({ id: 'job-new', parameters: (body as Record<string, unknown>).parameters as Record<string, unknown> });
      jobsById['job-new'] = created;
      return { status: 202, body: created };
    });
    await mount();
    await setText(field('Seed'), '0');
    await submitForm();
    expect(postJobs()).toHaveLength(1);
    const body = postJobs()[0].body as Record<string, unknown>;
    expect(body.seed).toBe(0);
    expect(body.backend).toBe('demo');
    expect(body.preset).toBe('standard');
    expect(body.texture).toBe(true);
    const parameters = body.parameters as Record<string, unknown>;
    for (const key of Object.keys(STANDARD_SETTINGS)) expect(parameters).toHaveProperty(key, STANDARD_SETTINGS[key as keyof GenerationSettings]);
  });

  it('blocks submission when a field violates schema limits', async () => {
    await mount();
    await setText(field('Shape steps'), '500');
    await submitForm();
    expect(postJobs()).toHaveLength(0);
  });

  it('disables texture output for Omni and forces shape-only payload', async () => {
    route('POST', /\/api\/jobs$/, body => {
      const created = jobFixture({ id: 'job-omni', backend: 'hunyuan3d-omni', texture: false });
      jobsById['job-omni'] = created;
      return { status: 202, body: created };
    });
    await mount();
    await setSelect(field('Runtime'), 'hunyuan');
    await flush();
    await setSelect(field('Shape backend'), 'hunyuan3d-omni');
    await flush();
    expect((field('Output') as HTMLSelectElement).disabled).toBe(true);
    expect(container.querySelector('input[aria-label="Omni control file"]')).toBeTruthy();
    await pickFile('Reference image', new File(['img'], 'ref.png', { type: 'image/png' }));
    await pickFile('Omni control file', new File(['{}'], 'ctrl.json', { type: 'application/json' }));
    await submitForm();
    expect(postJobs()).toHaveLength(1);
    const body = postJobs()[0].body as Record<string, unknown>;
    expect(body.texture).toBe(false);
    expect(body.backend).toBe('hunyuan3d-omni');
    expect(String(body.image)).toContain('data:image/png');
    expect(String(body.control_data)).toContain('data:application/json');
  });

  it('renders five stage rows from backend stage_status', async () => {
    const job = jobFixture({
      stage: 'generating_textures',
      stage_status: {
        shape: stageRecord('complete', 12.3),
        texture: stageRecord('running'),
        unity: stageRecord('waiting'),
        validation: stageRecord('waiting'),
      },
    });
    jobsById[job.id] = job;
    listedJobs = [job];
    window.localStorage.setItem('hunyforge.selectedJob', job.id);
    await mount();
    const rows = Array.from(container.querySelectorAll('.stage-entry'));
    expect(rows).toHaveLength(5);
    expect(rows.map(row => row.querySelector('.stage-state')?.textContent)).toEqual(['complete', 'running', 'skipped', 'waiting', 'waiting']);
  });

  it('shows ready artifact links while texture is still running', async () => {
    const job = jobFixture({
      stage: 'generating_textures',
      artifact_readiness: { 'white-mesh.glb': true },
      artifacts: ['artifacts/white-mesh.glb'],
      stage_status: { shape: stageRecord('complete'), texture: stageRecord('running'), unity: stageRecord('waiting'), validation: stageRecord('waiting') },
    });
    listedJobs = [job];
    jobsById[job.id] = job;
    window.localStorage.setItem('hunyforge.selectedJob', job.id);
    await mount();
    const links = Array.from(container.querySelectorAll('.artifact-link')).map(link => link.textContent);
    expect(links).toContain('white-mesh.glb');
    expect(links).not.toContain('textured-mesh.glb');
    expect(links).not.toContain('unity-package.zip');
  });

  it('shows cancelling status from the server without optimistic terminal state', async () => {
    const job = jobFixture({ stage: 'generating_textures', available_actions: ['cancel'], stage_status: { shape: stageRecord('complete'), texture: stageRecord('running'), unity: stageRecord('waiting'), validation: stageRecord('waiting') } });
    listedJobs = [job];
    jobsById[job.id] = job;
    window.localStorage.setItem('hunyforge.selectedJob', job.id);
    route('POST', /\/api\/jobs\/[^/]+\/cancel$/, () => ({ body: { ...job, stage: 'cancelling', available_actions: [] } }));
    await mount();
    await act(async () => {
      buttonWith('Cancel').click();
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(jobCalls('/cancel')).toHaveLength(1);
    expect(container.textContent).toContain('Cancelling');
    expect(container.textContent).not.toContain('Cancelled');
  });

  it('resumes from the server-advertised stage via /resume', async () => {
    const job = jobFixture({ stage: 'failed', failed_stage: 'texture', available_actions: ['restart', 'resume', 'resume_texture'], resume_stage: 'texture', error_message: 'texture exploded' });
    listedJobs = [job];
    jobsById[job.id] = job;
    window.localStorage.setItem('hunyforge.selectedJob', job.id);
    route('POST', /\/api\/jobs\/[^/]+\/resume$/, () => ({ status: 202, body: jobFixture({ id: 'job-child', parent_job_id: job.id, resume_from: 'texture' }) }));
    route('POST', /\/api\/jobs\/[^/]+\/restart$/, () => ({ status: 202, body: jobFixture({ id: 'job-restarted', parent_job_id: job.id }) }));
    await mount();
    await act(async () => {
      buttonWith('Resume from texture').click();
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(jobCalls('/resume')).toHaveLength(1);
    expect(jobCalls('/retry')).toHaveLength(0);
    await act(async () => {
      for (const el of Array.from(container.querySelectorAll('.history-item'))) {
        if (el.textContent?.includes('job-0000')) (el as HTMLButtonElement).click();
      }
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    await act(async () => {
      buttonWith('Restart from shape').click();
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(jobCalls('/restart')).toHaveLength(1);
  });

  it('toggles the mobile sidebar from the topbar menu button and Escape', async () => {
    await mount();
    const toggle = container.querySelector<HTMLButtonElement>('.topbar-menu')!;
    const sidebar = container.querySelector<HTMLElement>('#app-sidebar')!;
    expect(sidebar.classList.contains('open')).toBe(false);
    expect(toggle.getAttribute('aria-controls')).toBe('app-sidebar');
    await act(async () => {
      toggle.click();
    });
    expect(sidebar.classList.contains('open')).toBe(true);
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    expect(sidebar.classList.contains('open')).toBe(false);
  });

  it('keeps the closed mobile sidebar inert and moves focus between close and toggle', async () => {
    vi.stubGlobal('matchMedia', (query: string) => ({
      matches: true,
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
    }));
    await mount();
    const sidebar = container.querySelector<HTMLElement>('#app-sidebar')!;
    const main = container.querySelector<HTMLElement>('#main-content')!;
    const toggle = container.querySelector<HTMLButtonElement>('.topbar-menu')!;
    const close = container.querySelector<HTMLButtonElement>('.sidebar-close')!;
    expect(sidebar.hasAttribute('inert')).toBe(true);
    expect(main.hasAttribute('inert')).toBe(false);
    await act(async () => {
      toggle.click();
    });
    expect(sidebar.hasAttribute('inert')).toBe(false);
    expect(main.hasAttribute('inert')).toBe(true);
    expect(document.activeElement).toBe(close);
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    expect(sidebar.hasAttribute('inert')).toBe(true);
    expect(main.hasAttribute('inert')).toBe(false);
    expect(document.activeElement).toBe(toggle);
  });

  it('shows worker step progress beside the current operation', async () => {
    const job = jobFixture({
      stage: 'generating_textures',
      current_operation: 'texture_diffusion',
      operation_progress: { step: 3, total: 10 },
      stage_status: {
        shape: stageRecord('complete'),
        texture: stageRecord('running'),
        unity: stageRecord('waiting'),
        validation: stageRecord('waiting'),
      },
    });
    jobsById[job.id] = job;
    listedJobs = [job];
    window.localStorage.setItem('hunyforge.selectedJob', job.id);
    await mount();
    expect(container.textContent).toContain('texture diffusion');
    expect(container.textContent).toContain('step 3/10');
  });

  it('previews the white mesh while texture runs and switches to textured when ready', async () => {
    const job = jobFixture({
      stage: 'generating_textures',
      texture: true,
      artifact_readiness: { 'white-mesh.glb': true },
      stage_status: {
        shape: stageRecord('complete'),
        texture: stageRecord('running'),
        unity: stageRecord('waiting'),
        validation: stageRecord('waiting'),
      },
    });
    jobsById[job.id] = job;
    listedJobs = [job];
    localStorage.setItem('hunyforge.selectedJob', job.id);
    await mount();
    const preview = () => container.querySelector('[data-testid="three-preview"]')!.textContent ?? '';
    expect(preview()).toContain('white-mesh.glb');
    expect(preview()).not.toContain('textured-mesh.glb');
    const textured = { ...job, artifact_readiness: { 'white-mesh.glb': true, 'textured-mesh.glb': true }, updated_at: '2026-09-13T10:00:10Z' };
    await act(async () => {
      MockEventSource.instances[0].emit(textured);
    });
    expect(preview()).toContain('textured-mesh.glb');
    await act(async () => {
      buttonWith('White').click();
    });
    expect(preview()).toContain('white-mesh.glb');
  });

  it('restores the persisted selection and falls back to the latest job on 404', async () => {
    const saved = jobFixture({ id: 'job-saved' });
    jobsById['job-saved'] = saved;
    listedJobs = [saved];
    window.localStorage.setItem('hunyforge.selectedJob', 'job-saved');
    await mount();
    expect(container.textContent).toContain('job-sav');
    await act(async () => root.unmount());
    container.remove();
    window.localStorage.setItem('hunyforge.selectedJob', 'job-missing');
    const latest = jobFixture({ id: 'job-latest' });
    jobsById['job-latest'] = latest;
    listedJobs = [latest];
    await mount();
    expect(container.textContent).toContain('job-late');
  });

  it('keeps polling authoritative after SSE disconnect with a durable degraded indicator', async () => {
    vi.useFakeTimers();
    try {
      const job = jobFixture({ stage: 'generating_shape' });
      listedJobs = [job];
      jobsById[job.id] = job;
      window.localStorage.setItem('hunyforge.selectedJob', job.id);
      await mount();
      expect(MockEventSource.instances).toHaveLength(1);
      await act(async () => {
        MockEventSource.instances[0].fail();
      });
      expect(container.textContent).toContain('Live updates degraded');
      expect(MockEventSource.instances[0].closed).toBe(true);
      await act(async () => {
        MockEventSource.instances[0].emit({ ...job, updated_at: '2026-09-13T10:00:10Z' });
      });
      expect(container.textContent).toContain('Live updates degraded');
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5100);
      });
      expect(container.textContent).toContain('Live updates degraded');
      expect(container.textContent).toContain('polling job status');
    } finally {
      vi.useRealTimers();
    }
  });

  it('ignores stale updates from a torn-down watcher', async () => {
    const first = jobFixture({ id: 'job-first', stage: 'generating_shape' });
    const second = jobFixture({ id: 'job-second', stage: 'complete', progress: 100 });
    listedJobs = [first, second];
    jobsById['job-first'] = first;
    jobsById['job-second'] = second;
    window.localStorage.setItem('hunyforge.selectedJob', 'job-first');
    await mount();
    expect(MockEventSource.instances).toHaveLength(1);
    await act(async () => {
      for (const el of Array.from(container.querySelectorAll('.history-item'))) {
        if (el.textContent?.includes('job-seco')) (el as HTMLButtonElement).click();
      }
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(MockEventSource.instances).toHaveLength(2);
    await act(async () => {
      MockEventSource.instances[0].emit({ ...first, stage: 'failed', error_message: 'stale' });
    });
    expect(container.textContent).toContain('job-seco');
    expect(container.textContent).not.toContain('stale');
  });

  it('keeps the reference image when a control file is chosen', async () => {
    await mount();
    await pickFile('Reference image', new File(['img'], 'ref.png', { type: 'image/png' }));
    expect(container.textContent).toContain('ref.png');
    await setSelect(field('Runtime'), 'hunyuan');
    await flush();
    await setSelect(field('Shape backend'), 'hunyuan3d-omni');
    await flush();
    await pickFile('Omni control file', new File(['{}'], 'ctrl.json', { type: 'application/json' }));
    expect(container.textContent).toContain('ref.png');
    expect(container.textContent).toContain('ctrl.json');
  });

  it('clears a stored image when an invalid file is picked', async () => {
    await mount();
    await pickFile('Reference image', new File(['img'], 'ref.png', { type: 'image/png' }));
    expect(container.textContent).toContain('ref.png');
    await pickFile('Reference image', new File(['x'], 'evil.txt', { type: 'text/plain' }), false);
    expect(container.textContent).toContain('PNG or JPEG');
    expect(container.textContent).not.toContain('ref.png');
  });

  it('clears the pending upload state when an invalid file supersedes an in-flight read', async () => {
    class DeferredReader {
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      result: string | null = null;
      error = null;
      readAsDataURL() {}
    }
    vi.stubGlobal('FileReader', DeferredReader);
    await mount();
    const input = container.querySelector('input[aria-label="Reference image"]') as HTMLInputElement;
    await act(async () => {
      Object.defineProperty(input, 'files', { value: [new File(['img'], 'ref.png', { type: 'image/png' })], configurable: true });
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(container.textContent).toContain('Reading upload');
    await pickFile('Reference image', new File(['x'], 'evil.txt', { type: 'text/plain' }), false);
    expect(container.textContent).toContain('PNG or JPEG');
    expect(container.textContent).not.toContain('Reading upload');
  });

  it('generates a text-mode reference preview and submits it as the job image', async () => {
    enableT2i();
    route('POST', /\/api\/reference-preview$/, () => ({ body: { image: 'data:image/png;base64,Z2VucmF0ZWQ=', timings: { inference: 1.2 }, prompt_effective: 'single object, centered, plain background: worn boot', seed: 48291 } }));
    route('POST', /\/api\/jobs$/, body => {
      const created = jobFixture({ id: 'job-text', input_mode: 'text' });
      jobsById['job-text'] = created;
      return { status: 202, body: created };
    });
    await mount();
    await act(async () => { buttonWith('Text').click(); });
    await setText(field('Text prompt'), 'a worn leather boot');
    await act(async () => { buttonWith('Generate reference').click(); await settle(); });
    expect(previewCalls()).toHaveLength(1);
    const previewBody = previewCalls()[0].body as Record<string, unknown>;
    expect(previewBody.prompt).toBe('a worn leather boot');
    expect(previewBody.scaffold).toBe(true);
    expect(container.querySelector('.ref-preview img')).toBeTruthy();
    await submitForm();
    expect(postJobs()).toHaveLength(1);
    const body = postJobs()[0].body as Record<string, unknown>;
    expect(body.input_mode).toBe('text');
    expect(body.prompt).toBe('a worn leather boot');
    expect(body.t2i_seed).toBe(48291);
    expect(String(body.image)).toBe('data:image/png;base64,Z2VucmF0ZWQ=');
  });

  it('blocks text-mode submission until a reference is generated', async () => {
    enableT2i();
    await mount();
    await act(async () => { buttonWith('Text').click(); });
    await setText(field('Text prompt'), 'a brass lantern');
    await submitForm();
    expect(postJobs()).toHaveLength(0);
    expect(container.textContent).toContain('Generate a reference image and review it before submitting');
  });

  it('keeps text mode unavailable when the T2I worker is disabled', async () => {
    await mount();
    const textButton = Array.from(container.querySelectorAll('.segmented button')).find(el => el.textContent?.includes('Text')) as HTMLButtonElement;
    expect(textButton.disabled).toBe(true);
    expect(container.querySelector('.prompt-area')).toBeNull();
  });

  it('runs the edit flow: viewport capture, preview, retexture child job, parent compare', async () => {
    enableT2i();
    const parent = jobFixture({
      id: 'job-parent',
      stage: 'complete',
      progress: 100,
      artifact_readiness: { 'white-mesh.glb': true, 'textured-mesh.glb': true },
      artifacts: ['artifacts/white-mesh.glb', 'artifacts/textured-mesh.glb'],
      stage_status: { shape: stageRecord('complete'), texture: stageRecord('complete'), unity: stageRecord('complete'), validation: stageRecord('complete') },
    });
    jobsById['job-parent'] = parent;
    listedJobs = [parent];
    window.localStorage.setItem('hunyforge.selectedJob', parent.id);
    route('POST', /\/api\/reference-preview$/, () => ({ body: { image: 'data:image/png;base64,ZWRpdGVk', timings: { inference: 0.8 }, prompt_effective: 'edit: rusty', seed: 7 } }));
    route('POST', /\/api\/jobs\/[^/]+\/retexture$/, () => {
      const child = jobFixture({ id: 'job-child', parent_job_id: 'job-parent', input_mode: 'retexture', prompt: 'make it rusty metal' });
      jobsById['job-child'] = child;
      return { status: 202, body: child };
    });
    await mount();
    await act(async () => { buttonWith('Edit').click(); });
    await setText(field('Edit instruction'), 'make it rusty metal');
    await act(async () => { buttonWith('Generate edit preview').click(); await settle(); });
    expect(previewCalls()).toHaveLength(1);
    const previewBody = previewCalls()[0].body as Record<string, unknown>;
    expect(previewBody.image).toBe('data:image/png;base64,bW9ja2NhcHR1cmU=');
    expect(previewBody.parent_job_id).toBe('job-parent');
    await act(async () => { buttonWith('Apply retexture as child job').click(); await settle(); });
    expect(jobCalls('/retexture')).toHaveLength(1);
    const retextureBody = jobCalls('/retexture')[0].body as Record<string, unknown>;
    expect(retextureBody.image).toBe('data:image/png;base64,ZWRpdGVk');
    expect(retextureBody.prompt).toBe('make it rusty metal');
    expect(container.textContent).toContain('job-chil');
    await act(async () => { buttonWith('Parent').click(); });
    const preview = container.querySelector('[data-testid="three-preview"]')!.textContent ?? '';
    expect(preview).toContain('job-parent');
  });

  it('shows input-mode badges and lineage in job history', async () => {
    const textJob = jobFixture({ id: 'job-textmode', input_mode: 'text', stage: 'complete', progress: 100 });
    const retexJob = jobFixture({ id: 'job-retex', input_mode: 'retexture', parent_job_id: 'job-textmode', stage: 'complete', progress: 100 });
    listedJobs = [retexJob, textJob];
    jobsById['job-textmode'] = textJob;
    jobsById['job-retex'] = retexJob;
    await mount();
    const badges = Array.from(container.querySelectorAll('.input-badge')).map(el => el.textContent);
    expect(badges).toContain('TXT');
    expect(badges).toContain('RETEX');
  });

  it('submits the selected asset type with the job payload', async () => {
    listedJobs = [];
    route('POST', /\/api\/jobs$/, body => {
      const created = jobFixture({ id: 'job-vehicle', stage: 'queued', ...(body as object) });
      jobsById['job-vehicle'] = created;
      return { status: 202, body: created };
    });
    await mount();
    await act(async () => { buttonWith('Vehicle').click(); });
    await act(async () => { buttonWith('Character').click(); }); // disabled — should not select
    await submitForm();
    const posts = calls.filter(call => call.method === 'POST' && /\/api\/jobs$/.test(call.url));
    expect(posts).toHaveLength(1);
    expect((posts[0].body as Record<string, unknown>).asset_type).toBe('vehicle');
  });

  it('runs the vehicle rig flow: suggest wheels, build rigged child job', async () => {
    const parent = jobFixture({
      id: 'job-vehicle',
      stage: 'complete',
      progress: 100,
      asset_type: 'vehicle',
      artifacts: ['artifacts/white-mesh.glb', 'artifacts/textured-mesh.glb'],
      artifact_readiness: { 'white-mesh.glb': true, 'textured-mesh.glb': true },
      stage_status: { shape: stageRecord('complete'), texture: stageRecord('complete'), rig: stageRecord('skipped'), unity: stageRecord('complete'), validation: stageRecord('complete') },
    });
    jobsById['job-vehicle'] = parent;
    listedJobs = [parent];
    window.localStorage.setItem('hunyforge.selectedJob', parent.id);
    const wheels = [
      { name: 'Wheel_FL', center: [-0.45, -0.3, 0.55], axis: [1, 0, 0], radius: 0.18, half_width: 0.08, steer: true },
      { name: 'Wheel_FR', center: [0.45, -0.3, 0.55], axis: [1, 0, 0], radius: 0.18, half_width: 0.08, steer: true },
      { name: 'Wheel_RL', center: [-0.45, -0.3, -0.55], axis: [1, 0, 0], radius: 0.18, half_width: 0.08, steer: false },
      { name: 'Wheel_RR', center: [0.45, -0.3, -0.55], axis: [1, 0, 0], radius: 0.18, half_width: 0.08, steer: false },
    ];
    route('POST', /\/api\/jobs\/[^/]+\/wheel-suggest$/, () => ({ body: { wheels, ground_faces_removed: 120, source_mesh: 'textured-mesh.glb' } }));
    route('POST', /\/api\/jobs\/[^/]+\/rig$/, () => {
      const child = jobFixture({ id: 'job-rigged', parent_job_id: 'job-vehicle', input_mode: 'vehicle-rig', asset_type: 'vehicle', stage: 'queued' });
      jobsById['job-rigged'] = child;
      return { status: 202, body: child };
    });
    await mount();
    await act(async () => { buttonWith('Rig').click(); });
    await act(async () => { buttonWith('Auto-suggest wheels').click(); await settle(); });
    expect(jobCalls('/wheel-suggest')).toHaveLength(1);
    expect(container.querySelectorAll('.wheel-card')).toHaveLength(4);
    await act(async () => { buttonWith('Build rigged child job').click(); await settle(); });
    const rigCalls = jobCalls('/rig');
    expect(rigCalls).toHaveLength(1);
    const rigBody = (rigCalls[0].body as Record<string, unknown>).rig_spec as { wheels: { name: string }[] };
    expect(rigBody.wheels).toHaveLength(4);
    expect(rigBody.wheels[0].name).toBe('Wheel_FL');
    expect(container.textContent).toContain('job-rigg');
  });

  it('shows a RIG badge for vehicle rig children in history', async () => {
    const rigged = jobFixture({ id: 'job-rigged', input_mode: 'vehicle-rig', asset_type: 'vehicle', parent_job_id: 'job-parent', stage: 'complete', progress: 100 });
    listedJobs = [rigged];
    jobsById['job-rigged'] = rigged;
    await mount();
    const badges = Array.from(container.querySelectorAll('.input-badge')).map(el => el.textContent);
    expect(badges).toContain('RIG');
  });
});
