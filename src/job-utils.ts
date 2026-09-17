interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}

export const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8081';

export type BackendId = 'demo' | 'hunyuan3d-2.1' | 'hunyuan3d-omni';
export type PresetName = 'draft' | 'standard' | 'final';
export type InputMode = 'image' | 'multi-image' | 'text' | 'retexture' | 'vehicle-rig';
export type AssetType = 'generic' | 'character' | 'vehicle';
export type ReferenceView = 'front' | 'rear' | 'left' | 'right' | 'front_left' | 'rear_right' | 'top' | 'bottom';

export interface ReferenceImage {
  view: ReferenceView;
  image?: string;
  filename: string;
  content_type: 'image/png' | 'image/jpeg';
  byte_size: number;
  sha256?: string | null;
}
export type StageName = 'shape' | 'texture' | 'rig' | 'unity' | 'validation';
export type StageStateValue = 'waiting' | 'running' | 'complete' | 'failed' | 'skipped' | 'cancelled';
export type JobStage =
  | 'idle'
  | 'queued'
  | 'loading_shape_model'
  | 'generating_shape'
  | 'processing_mesh'
  | 'releasing_shape_model'
  | 'loading_texture_model'
  | 'generating_textures'
  | 'rigging'
  | 'preparing_unity_export'
  | 'validating_unity_export'
  | 'cancellation_requested'
  | 'cancelling'
  | 'complete'
  | 'failed'
  | 'cancelled';

export interface GenerationSettings {
  num_inference_steps: number;
  guidance_scale: number;
  octree_resolution: 256 | 384 | 512;
  num_chunks: number;
  remove_background: boolean;
  render_size: 512 | 768 | 1024;
  texture_size: 1024 | 2048;
  max_num_view: 4 | 6;
  texture_inference_steps: number;
  texture_guidance_scale: number;
  multiview_resolution: 512 | 768;
  face_count: number;
  generate_lods: boolean;
  generate_collision: boolean;
  collision_mode: 'box' | 'convex_hull';
  unity_mode: 'fast' | 'full';
}

export interface SchemaProperty {
  type?: string;
  title?: string;
  minimum?: number;
  maximum?: number;
  enum?: (number | string)[];
  default?: unknown;
}

export interface PresetEntry {
  label: string;
  description: string;
  quality_status?: string;
  expected_duration_seconds: number | null;
  expected_peak_vram_mb: number | null;
  settings: GenerationSettings;
}

export interface PresetContract {
  version: number;
  default: PresetName;
  presets: Record<string, PresetEntry>;
  settings_schema: { properties?: Record<string, SchemaProperty>; required?: string[] };
}

export interface StageState {
  state: StageStateValue;
  started_at: string | null;
  finished_at: string | null;
  elapsed_seconds: number;
  error: string | null;
}

export interface VehicleWheel {
  name: string;
  center: [number, number, number];
  axis: [number, number, number];
  radius: number;
  half_width: number;
  steer: boolean;
}

export interface VehicleRigSpec {
  wheels: VehicleWheel[];
  strip_ground?: boolean;
}

export interface WheelSuggestResult {
  wheels: VehicleWheel[];
  ground_faces_removed: number;
  source_mesh: string;
}

export interface JobUpdate {
  id: string;
  stage: JobStage;
  progress: number;
  backend: BackendId;
  seed: number;
  texture: boolean;
  preset: PresetName;
  parameters: Partial<GenerationSettings> & Record<string, unknown>;
  artifacts: string[];
  artifact_readiness: Record<string, boolean>;
  available_actions: string[];
  resume_stage: StageName | null;
  stage_status: Partial<Record<StageName, StageState>>;
  timings: Record<string, unknown>;
  current_operation: string | null;
  operation_progress: Record<string, unknown> | null;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
  updated_at: string;
  parent_job_id: string | null;
  resume_from: StageName | null;
  input_mode?: InputMode;
  reference_images?: ReferenceImage[];
  prompt?: string | null;
  t2i_seed?: number | null;
  t2i_model?: string | null;
  asset_type?: AssetType;
  rig_spec?: VehicleRigSpec | null;
  unreal_export?: boolean;
  failed_stage: string | null;
  peak_vram_mb: number | null;
  error_code: string | null;
  error_message: string | null;
}

export interface ValidationReport {
  status: string;
  blocking_failures?: { check: string; severity?: string }[];
  warnings?: (string | { severity?: string; message: string })[];
  demo?: boolean;
  attempt_job_id?: string;
  source_manifest_job_id?: string | null;
  geometry_processing?: string[];
  unreal_ready?: boolean | null;
  checks?: Record<string, string[]>;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
}

export interface HealthInfo {
  inference_mode: string;
  gpu_available?: boolean;
  runtime_ready: boolean;
  hunyuan_service_ready: boolean;
  workers?: Record<string, { ready?: boolean; busy?: boolean; last_error?: string | null; state?: string }>;
  t2i?: { enabled: boolean; model: string; model_path: string; max_prompt_chars: number };
  multi_view?: { enabled: boolean; adapter_configured: boolean; reason?: string | null };
}

export const STAGE_ROWS: { key: StageName; label: string }[] = [
  { key: 'shape', label: 'Shape' },
  { key: 'texture', label: 'Texture' },
  { key: 'rig', label: 'Vehicle rig' },
  { key: 'unity', label: 'Engine export prep' },
  { key: 'validation', label: 'Validation & package' },
];

export const STAGE_LABELS: Record<JobStage, string> = {
  idle: 'Ready',
  queued: 'Queued',
  loading_shape_model: 'Loading shape model',
  generating_shape: 'Generating shape',
  processing_mesh: 'Processing mesh',
  releasing_shape_model: 'Releasing shape model',
  loading_texture_model: 'Loading texture model',
  generating_textures: 'Generating PBR textures',
  rigging: 'Partitioning vehicle wheels',
  preparing_unity_export: 'Preparing engine export',
  validating_unity_export: 'Validating & packaging',
  cancellation_requested: 'Cancellation requested',
  cancelling: 'Cancelling',
  complete: 'Complete',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

export function generatedArtifactName(texture: boolean): 'textured-mesh.glb' | 'white-mesh.glb' {
  return texture ? 'textured-mesh.glb' : 'white-mesh.glb';
}

export function isTerminalStage(stage: string): boolean {
  return ['complete', 'failed', 'cancelled'].includes(stage);
}

export function readyArtifactNames(job: JobUpdate | null): string[] {
  if (!job) return [];
  const names = new Set(Object.keys(job.artifact_readiness ?? {}).filter(name => job.artifact_readiness[name]));
  for (const path of job.artifacts ?? []) names.add(path.split('/').pop() as string);
  return [...names];
}

export type PreviewChoice = 'auto' | 'white' | 'textured' | 'rigged';

export function selectPreviewName(job: JobUpdate | null, choice: PreviewChoice): string | null {
  if (!job) return null;
  const ready = new Set(readyArtifactNames(job));
  const whiteReady = ready.has('white-mesh.glb');
  const texturedReady = ready.has('textured-mesh.glb');
  const riggedReady = ready.has('vehicle-rigged.glb');
  if (choice === 'rigged') return riggedReady ? 'vehicle-rigged.glb' : null;
  if (choice === 'white') return whiteReady ? 'white-mesh.glb' : null;
  if (choice === 'textured') return texturedReady ? 'textured-mesh.glb' : whiteReady ? 'white-mesh.glb' : null;
  if (riggedReady) return 'vehicle-rigged.glb';
  if (job.texture && texturedReady) return 'textured-mesh.glb';
  return whiteReady ? 'white-mesh.glb' : null;
}

export function settingsError(settings: GenerationSettings): string | null {
  if (settings.texture_size < settings.render_size) return 'Texture size must be at least the render size';
  if (settings.unity_mode === 'fast' && settings.generate_collision && settings.collision_mode === 'convex_hull')
    return 'Fast Unity mode does not support convex hull collision — use box collision or Full mode';
  return null;
}

export function submissionError(backend: BackendId, image: string | null, controlType: string | null, controlData: string | null, settings: GenerationSettings): string | null {
  if (backend !== 'demo' && !image) return 'Upload a reference image before starting a real Hunyuan run';
  if (backend === 'hunyuan3d-omni' && (!controlType || !controlData)) return 'Upload an Omni control file before starting an Omni run';
  return settingsError(settings);
}

export interface PresetEstimate {
  duration: string;
  vram: string;
  warnings: string[];
}

function sameSettings(a: GenerationSettings, b: GenerationSettings): boolean {
  return (Object.keys(a) as (keyof GenerationSettings)[]).every(key => a[key] === b[key]);
}

export function presetEstimate(entry: PresetEntry | null | undefined, settings?: GenerationSettings, textureEnabled = true): PresetEstimate {
  const warnings: string[] = [];
  const s = settings ?? entry?.settings;
  const exact = !!entry && !!settings && sameSettings(entry.settings, settings);
  const duration = exact && entry.expected_duration_seconds != null ? `~${Math.round(entry.expected_duration_seconds)} s` : 'Not benchmarked';
  const vram = exact && entry.expected_peak_vram_mb != null ? `~${(entry.expected_peak_vram_mb / 1024).toFixed(1)} GB peak` : 'Not benchmarked';
  if (s && textureEnabled) {
    if (s.texture_size >= 2048 || s.max_num_view >= 6) warnings.push('All textured profiles may exceed 16 GB VRAM; unverified until measured on your GPU.');
    else if (s.render_size <= 512 || s.texture_size <= 1024) warnings.push('Lower-resolution profile; fidelity unverified until measured.');
    else warnings.push('Timing and quality are provisional until measured on your GPU.');
  }
  warnings.push('Estimated remaining time is unavailable until this preset is benchmarked.');
  return { duration, vram, warnings };
}

export function formatElapsed(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—';
  if (seconds < 60) return `${Math.max(0, seconds).toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

export function warningText(warning: string | { severity?: string; message: string }): string {
  return typeof warning === 'string' ? warning : warning.message;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) throw await apiError(response);
  return response.json() as Promise<T>;
}

export async function apiError(response: Response): Promise<Error> {
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === 'string') return new Error(detail);
    if (Array.isArray(detail)) return new Error(detail.map(entry => (typeof entry === 'object' && entry?.msg ? `${(entry.loc || []).join('.')}: ${entry.msg}` : String(entry))).join('; '));
    if (detail) return new Error(String(detail));
    return new Error(`Request failed (${response.status})`);
  } catch {
    return new Error(`Request failed (${response.status})`);
  }
}

export function describeError(error: unknown): string {
  return error instanceof Error ? error.message : 'Unable to reach the local HunyForge API';
}

export function fetchPresets(): Promise<PresetContract> {
  return request<PresetContract>('/api/presets');
}

export function fetchHealth(): Promise<HealthInfo> {
  return request<HealthInfo>('/health');
}

export function fetchProjects(): Promise<Project[]> {
  return request<Project[]>('/api/projects');
}

export function createProject(name: string): Promise<Project> {
  return request<Project>('/api/projects', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
}

export function fetchJobs(projectId?: string): Promise<JobUpdate[]> {
  return request<JobUpdate[]>(projectId ? `/api/jobs?project_id=${projectId}` : '/api/jobs');
}

export function fetchJob(id: string): Promise<JobUpdate> {
  return request<JobUpdate>(`/api/jobs/${id}`);
}

export function fetchJobInput(id: string): Promise<{ image: string | null; control_data: string | null }> {
  return request(`/api/jobs/${id}/input`);
}

export function fetchValidation(id: string): Promise<ValidationReport> {
  return request<ValidationReport>(`/api/jobs/${id}/validation`);
}

export interface JobPayload {
  project_id: string | null;
  backend: BackendId;
  preset: PresetName;
  seed: number;
  texture: boolean;
  image: string | null;
  reference_images?: ReferenceImage[];
  control_type: string | null;
  control_data: string | null;
  input_mode?: InputMode;
  prompt?: string | null;
  t2i_seed?: number | null;
  asset_type?: AssetType;
  unreal_export?: boolean;
  parameters: GenerationSettings;
}

export function submitJob(payload: JobPayload): Promise<JobUpdate> {
  return request<JobUpdate>('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

export interface ReferencePreview {
  image: string;
  timings: Record<string, number>;
  prompt_effective: string;
  seed: number;
}

export interface ReferencePreviewPayload {
  prompt: string;
  seed: number;
  image?: string | null;
  scaffold?: boolean;
  size?: 512 | 768 | 1024;
  parent_job_id?: string | null;
}

export function previewReference(payload: ReferencePreviewPayload): Promise<ReferencePreview> {
  return request<ReferencePreview>('/api/reference-preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

export function retextureJob(id: string, payload: { image: string; prompt?: string | null; seed?: number | null; t2i_seed?: number | null; parameters?: Partial<GenerationSettings> }): Promise<JobUpdate> {
  return request<JobUpdate>(`/api/jobs/${id}/retexture`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

export function suggestWheels(id: string): Promise<WheelSuggestResult> {
  return request<WheelSuggestResult>(`/api/jobs/${id}/wheel-suggest`, { method: 'POST' });
}

export function rigJob(id: string, spec: VehicleRigSpec): Promise<JobUpdate> {
  return request<JobUpdate>(`/api/jobs/${id}/rig`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rig_spec: spec }) });
}

export function jobAction(id: string, action: 'retry' | 'resume' | 'restart' | 'cancel'): Promise<JobUpdate> {
  return request<JobUpdate>(`/api/jobs/${id}/${action}`, { method: 'POST' });
}

export function inputModeBadge(mode: InputMode | undefined): { label: string; title: string } | null {
  if (mode === 'text') return { label: 'TXT', title: 'Text-generated reference' };
  if (mode === 'multi-image') return { label: 'MULTI', title: 'Multi-view reference set' };
  if (mode === 'retexture') return { label: 'RETEX', title: 'Text-guided retexture' };
  if (mode === 'vehicle-rig') return { label: 'RIG', title: 'Vehicle wheel rig' };
  return null;
}

export function artifactUrl(jobId: string, name: string): string {
  return `${API_URL}/api/jobs/${jobId}/artifacts/${encodeURIComponent(name)}`;
}

export function safeAssetUrl(requested: string, sourceUrl: string): string | null {
  let source: URL;
  let api: URL;
  try {
    api = new URL(API_URL);
    source = new URL(sourceUrl, API_URL);
  } catch {
    return null;
  }
  if (source.origin !== api.origin) return null;
  const jobMatch = /^\/api\/jobs\/([^/]+)\/artifacts\//.exec(source.pathname);
  if (!jobMatch) return null;
  if (requested.startsWith('blob:')) return requested;
  if (/^data:/i.test(requested)) {
    return /^data:(image\/png|image\/jpe?g|application\/octet-stream|application\/gltf-buffer)[;,]/i.test(requested) ? requested : null;
  }
  let parsed: URL;
  try {
    parsed = new URL(requested, API_URL);
  } catch {
    return null;
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
  if (parsed.origin !== source.origin) return null;
  if (parsed.username || parsed.password) return null;
  if (!parsed.pathname.startsWith(`/api/jobs/${jobMatch[1]}/artifacts/`)) return null;
  return parsed.href;
}

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

export function imageFileError(file: File): string | null {
  if (file.type !== 'image/png' && file.type !== 'image/jpeg') return 'Reference image must be a PNG or JPEG file';
  if (file.size > MAX_UPLOAD_BYTES) return 'Reference image exceeds the 10 MB upload limit';
  return null;
}

export function controlFileError(file: File): string | null {
  if (file.size > MAX_UPLOAD_BYTES) return 'Omni control file exceeds the 10 MB upload limit';
  return null;
}

export function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error('Unable to read file'));
    reader.readAsDataURL(file);
  });
}

export type ConnectionState = 'live' | 'degraded' | 'down';

export function watchJob(id: string, onJob: (job: JobUpdate) => void, onConnection: (state: ConnectionState) => void): () => void {
  let stopped = false;
  let lastUpdated = Number.NaN;
  let events: EventSource | null = null;
  let pollTimer: number | undefined;
  let polling = false;
  let inFlight: AbortController | null = null;
  let sseDead = false;

  const stop = () => {
    stopped = true;
    if (pollTimer !== undefined) window.clearInterval(pollTimer);
    events?.close();
    events = null;
    inFlight?.abort();
    inFlight = null;
  };

  const receive = (job: JobUpdate) => {
    if (stopped) return;
    const stamp = Date.parse(job.updated_at ?? '');
    if (Number.isFinite(lastUpdated) && Number.isFinite(stamp) && stamp < lastUpdated) return;
    if (Number.isFinite(stamp)) lastUpdated = stamp;
    onConnection(sseDead ? 'degraded' : 'live');
    onJob(job);
    if (isTerminalStage(job.stage)) stop();
  };

  const poll = async () => {
    if (stopped || polling) return;
    polling = true;
    const controller = new AbortController();
    inFlight = controller;
    const timeout = window.setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch(`${API_URL}/api/jobs/${id}`, { signal: controller.signal });
      if (!response.ok) throw new Error(`Job status returned ${response.status}`);
      const job = (await response.json()) as JobUpdate;
      receive(job);
    } catch {
      if (!stopped) onConnection('down');
    } finally {
      window.clearTimeout(timeout);
      if (inFlight === controller) inFlight = null;
      polling = false;
    }
  };

  try {
    if (typeof EventSource !== 'undefined') events = new EventSource(`${API_URL}/api/jobs/${id}/events`);
  } catch {
    events = null;
  }
  if (events === null) sseDead = true;
  const source = events;
  source?.addEventListener('job', event => {
    if (stopped || events !== source) return;
    try {
      receive(JSON.parse((event as MessageEvent).data) as JobUpdate);
    } catch {
      onConnection('degraded');
    }
  });
  if (source) {
    source.onerror = () => {
      source.close();
      if (events === source) events = null;
      sseDead = true;
      if (!stopped) onConnection('degraded');
      // SSE is an optimization. The polling loop remains authoritative when
      // a browser, proxy, or long-running request times out the event stream.
    };
  }
  void poll();
  pollTimer = window.setInterval(() => void poll(), 5000);
  return stop;
}
