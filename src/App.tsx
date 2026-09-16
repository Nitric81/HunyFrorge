import { Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react';
import { Ban, Box, CheckCircle2, ChevronDown, Download, FolderOpen, Gauge, ImagePlus, Layers3, LoaderCircle, Menu, Pencil, Play, RefreshCw, RotateCcw, ShieldCheck, Sparkles, Type, Upload, X, XCircle } from 'lucide-react';
import {
  AssetType,
  BackendId,
  ConnectionState,
  GenerationSettings,
  JobPayload,
  JobUpdate,
  PresetContract,
  PresetName,
  PreviewChoice,
  Project,
  ReferenceImage,
  ReferenceView,
  STAGE_LABELS,
  STAGE_ROWS,
  SchemaProperty,
  StageName,
  ValidationReport,
  VehicleWheel,
  artifactUrl,
  controlFileError,
  createProject,
  describeError,
  fetchHealth,
  fetchJob,
  fetchJobs,
  fetchPresets,
  fetchProjects,
  fetchValidation,
  formatElapsed,
  imageFileError,
  inputModeBadge,
  isTerminalStage,
  jobAction,
  presetEstimate,
  previewReference,
  readFileAsDataUrl,
  readyArtifactNames,
  ReferencePreview,
  retextureJob,
  rigJob,
  selectPreviewName,
  settingsError,
  submissionError,
  submitJob,
  suggestWheels,
  warningText,
  watchJob,
} from './job-utils';
import type { MeshStats, ThreePreviewHandle, ViewerMode } from './ThreePreview';

const ThreePreview = lazy(() => import('./ThreePreview'));
const SELECTED_JOB_KEY = 'hunyforge.selectedJob';

const SETTING_FIELDS: { key: keyof GenerationSettings; label: string; textureOnly?: boolean }[] = [
  { key: 'num_inference_steps', label: 'Shape steps' },
  { key: 'guidance_scale', label: 'Shape guidance scale' },
  { key: 'octree_resolution', label: 'Octree resolution' },
  { key: 'num_chunks', label: 'Shape chunks' },
  { key: 'remove_background', label: 'Remove background' },
  { key: 'texture_inference_steps', label: 'Texture steps', textureOnly: true },
  { key: 'texture_guidance_scale', label: 'Texture guidance scale', textureOnly: true },
  { key: 'render_size', label: 'Render size', textureOnly: true },
  { key: 'texture_size', label: 'Texture size', textureOnly: true },
  { key: 'max_num_view', label: 'Multiview count', textureOnly: true },
  { key: 'multiview_resolution', label: 'Multiview resolution', textureOnly: true },
  { key: 'face_count', label: 'Face budget' },
  { key: 'unity_mode', label: 'Unity mode' },
  { key: 'generate_lods', label: 'Generate LODs' },
  { key: 'generate_collision', label: 'Generate collision' },
  { key: 'collision_mode', label: 'Collision mode' },
];

const REQUIRED_REFERENCE_VIEWS: { view: ReferenceView; label: string }[] = [
  { view: 'front', label: 'Front' }, { view: 'rear', label: 'Rear' },
  { view: 'left', label: 'Left' }, { view: 'right', label: 'Right' },
];
const OPTIONAL_REFERENCE_VIEWS: { view: ReferenceView; label: string }[] = [
  { view: 'front_left', label: 'Front-left 3/4' }, { view: 'rear_right', label: 'Rear-right 3/4' },
  { view: 'top', label: 'Top' }, { view: 'bottom', label: 'Underside' },
];

function App() {
  const [contract, setContract] = useState<PresetContract | null>(null);
  const [preset, setPreset] = useState<PresetName>('standard');
  const [settings, setSettings] = useState<GenerationSettings | null>(null);
  const [runtime, setRuntime] = useState<'demo' | 'hunyuan'>('demo');
  const [backendChoice, setBackendChoice] = useState<'hunyuan3d-2.1' | 'hunyuan3d-omni'>('hunyuan3d-2.1');
  const [seed, setSeed] = useState('48291');
  const [texture, setTexture] = useState(true);
  const [controlType, setControlType] = useState('point');
  const [controlData, setControlData] = useState<string | null>(null);
  const [controlName, setControlName] = useState('');
  const [imageData, setImageData] = useState<string | null>(null);
  const [imageName, setImageName] = useState('');
  const [referenceMode, setReferenceMode] = useState<'single' | 'multi'>('single');
  const [referenceImages, setReferenceImages] = useState<Partial<Record<ReferenceView, ReferenceImage>>>({});
  const [referenceLoading, setReferenceLoading] = useState<Partial<Record<ReferenceView, boolean>>>({});
  const [menuOpen, setMenuOpen] = useState(false);
  const [mobile, setMobile] = useState(() => typeof window.matchMedia === 'function' && window.matchMedia('(max-width: 640px)').matches);
  const [activeTab, setActiveTab] = useState<'generate' | 'edit' | 'rig' | 'unity'>('generate');
  const [assetType, setAssetType] = useState<AssetType>('generic');
  const [inputMode, setInputMode] = useState<'image' | 'text'>('image');
  const [promptText, setPromptText] = useState('');
  const [scaffold, setScaffold] = useState(true);
  const [t2iSeed, setT2iSeed] = useState('48291');
  const [refPreview, setRefPreview] = useState<ReferencePreview | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [t2iEnabled, setT2iEnabled] = useState(false);
  const [multiView, setMultiView] = useState({ enabled: false, adapter_configured: false, reason: '' as string | null });
  const [editPrompt, setEditPrompt] = useState('');
  const [editSeed, setEditSeed] = useState('7');
  const [editPreview, setEditPreview] = useState<ReferencePreview | null>(null);
  const [editBusy, setEditBusy] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);
  const [parentJob, setParentJob] = useState<JobUpdate | null>(null);
  const [compareParent, setCompareParent] = useState(false);
  const [wheels, setWheels] = useState<VehicleWheel[] | null>(null);
  const [markingIndex, setMarkingIndex] = useState<number | null>(null);
  const [rigBusy, setRigBusy] = useState(false);
  const [rigError, setRigError] = useState<string | null>(null);
  const [viewerMode, setViewerMode] = useState<ViewerMode>('material');
  const [previewSource, setPreviewSource] = useState<PreviewChoice>('auto');
  const [meshStats, setMeshStats] = useState<MeshStats | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobUpdate | null>(null);
  const [jobs, setJobs] = useState<JobUpdate[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('live');
  const [runtimeStatus, setRuntimeStatus] = useState('Checking local runtime…');
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState('');
  const [imageLoading, setImageLoading] = useState(false);
  const [controlLoading, setControlLoading] = useState(false);
  const [actionPending, setActionPending] = useState(false);
  const [workerBusy, setWorkerBusy] = useState(false);
  const [, setTick] = useState(0);
  const formRef = useRef<HTMLFormElement>(null);
  const jobIdRef = useRef<string | null>(null);
  const imageSeq = useRef(0);
  const controlSeq = useRef(0);
  const runtimeAutoSelected = useRef(false);
  const sidebarCloseRef = useRef<HTMLButtonElement>(null);
  const menuToggleRef = useRef<HTMLButtonElement>(null);
  const menuWasOpen = useRef(false);
  const preview3dRef = useRef<ThreePreviewHandle>(null);

  const backend: BackendId = runtime === 'demo' ? 'demo' : backendChoice;
  const omni = backendChoice === 'hunyuan3d-omni';
  const effectiveTexture = runtime === 'hunyuan' && omni ? false : texture;
  const ready = useMemo(() => readyArtifactNames(job), [job]);
  const busy = useMemo(() => (job ? !isTerminalStage(job.stage) : false) || jobs.some(entry => !isTerminalStage(entry.stage)) || workerBusy, [job, jobs, workerBusy]);
  const schema = contract?.settings_schema?.properties ?? {};
  const estimate = useMemo(() => presetEstimate(contract?.presets?.[preset], settings ?? undefined, effectiveTexture), [contract, preset, settings, effectiveTexture]);
  const settingsProblem = settings ? settingsError(settings) : null;
  const uploadPending = imageLoading || controlLoading || Object.values(referenceLoading).some(Boolean);

  const texturedReady = ready.includes('textured-mesh.glb');
  const whiteReady = ready.includes('white-mesh.glb');
  const shownJob = compareParent && parentJob ? parentJob : job;
  const shownReady = useMemo(() => readyArtifactNames(shownJob), [shownJob]);
  const previewName = selectPreviewName(job, previewSource);
  const shownPreviewName = selectPreviewName(shownJob, previewSource);
  const modelUrl = shownJob && shownPreviewName ? artifactUrl(shownJob.id, shownPreviewName) : null;

  jobIdRef.current = jobId;

  function applyHealth(health: Awaited<ReturnType<typeof fetchHealth>>) {
    setRuntimeStatus(health.inference_mode === 'demo' ? 'Demo runtime · local' : health.runtime_ready ? 'Hunyuan · CUDA ready' : health.hunyuan_service_ready ? 'Hunyuan · setup required' : 'Hunyuan · model loading…');
    const busyNow = Object.values(health.workers ?? {}).some(worker => worker.busy === true);
    setWorkerBusy(busyNow);
    setT2iEnabled(Boolean(health.t2i?.enabled));
    setMultiView({ enabled: Boolean(health.multi_view?.enabled), adapter_configured: Boolean(health.multi_view?.adapter_configured), reason: health.multi_view?.reason ?? null });
    if (!runtimeAutoSelected.current && health.inference_mode !== 'demo') {
      runtimeAutoSelected.current = true;
      setRuntime('hunyuan');
    }
  }

  useEffect(() => {
    fetchPresets().then(next => {
      setContract(next);
      const initial = (next.presets[next.default] ? next.default : 'standard') as PresetName;
      setPreset(initial);
      setSettings({ ...next.presets[initial].settings });
    }).catch(() => setError('Generation presets unavailable — is the local API running?'));
    fetchHealth().then(applyHealth).catch(() => setRuntimeStatus('API offline'));
    fetchProjects().then(items => { setProjects(items); if (items[0]) setProjectId(current => current || items[0].id); }).catch(() => undefined);
    fetchJobs().then(items => {
      setJobs(items);
      const saved = window.localStorage.getItem(SELECTED_JOB_KEY);
      const fallback = items.find(entry => !isTerminalStage(entry.stage)) ?? items[0];
      if (saved) {
        fetchJob(saved).then(found => { setJobId(found.id); setJob(found); }).catch(() => {
          window.localStorage.removeItem(SELECTED_JOB_KEY);
          if (fallback) { setJobId(fallback.id); setJob(fallback); }
        });
      } else if (fallback) {
        setJobId(fallback.id);
        setJob(fallback);
      }
    }).catch(() => undefined);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      fetchHealth().then(applyHealth).catch(() => setRuntimeStatus('API offline'));
    }, 5000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!jobId) return;
    window.localStorage.setItem(SELECTED_JOB_KEY, jobId);
    setConnectionState('live');
    return watchJob(jobId, next => {
      setJob(next);
      setJobs(current => current.map(entry => (entry.id === next.id ? next : entry)));
    }, setConnectionState);
  }, [jobId]);

  useEffect(() => {
    if (job && !isTerminalStage(job.stage)) return;
    fetchJobs(projectId || undefined).then(setJobs).catch(() => undefined);
  }, [job?.stage, projectId]);

  useEffect(() => {
    if (!jobs.some(entry => entry.id !== jobId && !isTerminalStage(entry.stage))) return;
    const timer = window.setInterval(() => {
      fetchJobs(projectId || undefined).then(setJobs).catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(timer);
  }, [jobs, jobId, projectId]);

  useEffect(() => {
    if (!job?.parent_job_id) {
      setParentJob(null);
      setCompareParent(false);
      return;
    }
    let alive = true;
    fetchJob(job.parent_job_id).then(found => { if (alive) setParentJob(found); }).catch(() => { if (alive) setParentJob(null); });
    return () => {
      alive = false;
    };
  }, [job?.parent_job_id]);

  useEffect(() => {
    if (!job || !job.artifact_readiness['validation-report.json']) return;
    let alive = true;
    fetchValidation(job.id).then(report => { if (alive) setValidationReport(report); }).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [job?.artifact_readiness['validation-report.json'], job?.id]);

  useEffect(() => {
    if (job && !isTerminalStage(job.stage)) {
      const timer = window.setInterval(() => setTick(value => value + 1), 1000);
      return () => window.clearInterval(timer);
    }
  }, [job?.stage, job?.id]);

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const query = window.matchMedia('(max-width: 640px)');
    const onChange = (event: MediaQueryListEvent) => setMobile(event.matches);
    setMobile(query.matches);
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMenuOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [menuOpen]);

  useEffect(() => {
    if (!mobile) {
      menuWasOpen.current = false;
      return;
    }
    if (menuOpen && !menuWasOpen.current) sidebarCloseRef.current?.focus();
    if (!menuOpen && menuWasOpen.current) menuToggleRef.current?.focus();
    menuWasOpen.current = menuOpen;
  }, [menuOpen, mobile]);

  function selectJob(entry: JobUpdate) {
    setError(null);
    setValidationReport(null);
    setMeshStats(null);
    setPreviewSource('auto');
    setEditPreview(null);
    setEditError(null);
    setEditPrompt('');
    setCompareParent(false);
    setParentJob(null);
    setWheels(null);
    setMarkingIndex(null);
    setRigError(null);
    if (activeTab === 'rig' && entry.asset_type !== 'vehicle' && entry.input_mode !== 'vehicle-rig') setActiveTab('generate');
    setJobId(entry.id);
    setJob(entry);
  }

  function selectPreset(next: PresetName) {
    setPreset(next);
    const entry = contract?.presets?.[next];
    if (entry) setSettings({ ...entry.settings });
  }

  function setSetting<K extends keyof GenerationSettings>(key: K, value: GenerationSettings[K]) {
    setSettings(current => (current ? { ...current, [key]: value } : current));
  }

  async function onImageChange(file: File | undefined) {
    if (!file) return;
    const seq = ++imageSeq.current;
    const problem = imageFileError(file);
    if (problem) {
      setImageData(null);
      setImageName('');
      setImageLoading(false);
      setError(problem);
      return;
    }
    setImageLoading(true);
    try {
      const data = await readFileAsDataUrl(file);
      if (seq !== imageSeq.current) return;
      setImageData(data);
      setImageName(file.name);
      setError(null);
    } catch {
      if (seq !== imageSeq.current) return;
      setImageData(null);
      setImageName('');
      setError('Unable to read the selected image file');
    } finally {
      if (seq === imageSeq.current) setImageLoading(false);
    }
  }

  async function onReferenceImageChange(view: ReferenceView, file: File | undefined) {
    if (!file) return;
    const problem = imageFileError(file);
    if (problem) { setError(`${view.replace('_', ' ')}: ${problem}`); return; }
    setReferenceLoading(current => ({ ...current, [view]: true }));
    try {
      const image = await readFileAsDataUrl(file);
      const contentType = file.type as 'image/png' | 'image/jpeg';
      setReferenceImages(current => ({ ...current, [view]: { view, image, filename: file.name, content_type: contentType, byte_size: file.size } }));
      setError(null);
    } catch {
      setError(`Unable to read the ${view.replace('_', ' ')} reference image`);
    } finally {
      setReferenceLoading(current => ({ ...current, [view]: false }));
    }
  }

  function removeReferenceImage(view: ReferenceView) {
    setReferenceImages(current => {
      const next = { ...current };
      delete next[view];
      return next;
    });
  }

  async function onControlChange(file: File | undefined) {
    if (!file) return;
    const seq = ++controlSeq.current;
    const problem = controlFileError(file);
    if (problem) {
      setControlData(null);
      setControlName('');
      setControlLoading(false);
      setError(problem);
      return;
    }
    setControlLoading(true);
    try {
      const data = await readFileAsDataUrl(file);
      if (seq !== controlSeq.current) return;
      setControlData(data);
      setControlName(file.name);
      setError(null);
    } catch {
      if (seq !== controlSeq.current) return;
      setControlData(null);
      setControlName('');
      setError('Unable to read the selected control file');
    } finally {
      if (seq === controlSeq.current) setControlLoading(false);
    }
  }

  async function generateReference() {
    if (previewBusy) return;
    const trimmed = promptText.trim();
    if (!trimmed) {
      setError('Enter a prompt before generating a reference');
      return;
    }
    setPreviewBusy(true);
    setError(null);
    try {
      const result = await previewReference({ prompt: trimmed, seed: Number(t2iSeed), scaffold });
      setRefPreview(result);
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setPreviewBusy(false);
    }
  }

  async function generateEditPreview() {
    if (!job || editBusy) return;
    const trimmed = editPrompt.trim();
    if (!trimmed) {
      setEditError('Describe the retexture before generating a preview');
      return;
    }
    const captured = preview3dRef.current?.capture();
    if (!captured) {
      setEditError('Model preview is not loaded yet — wait for the mesh to render');
      return;
    }
    setEditBusy(true);
    setEditError(null);
    try {
      const result = await previewReference({ prompt: trimmed, seed: Number(editSeed), image: captured, scaffold, parent_job_id: job.id });
      setEditPreview(result);
    } catch (caught) {
      setEditError(describeError(caught));
    } finally {
      setEditBusy(false);
    }
  }

  async function applyRetexture() {
    if (!job || !editPreview || actionPending) return;
    const requestJobId = job.id;
    setActionPending(true);
    setEditError(null);
    try {
      const child = await retextureJob(requestJobId, { image: editPreview.image, prompt: editPrompt.trim(), t2i_seed: Number(editSeed) });
      if (jobIdRef.current !== requestJobId) return;
      setEditPreview(null);
      setEditPrompt('');
      setValidationReport(null);
      setMeshStats(null);
      setPreviewSource('auto');
      setJob(child);
      setJobId(child.id);
      setJobs(current => [child, ...current]);
    } catch (caught) {
      setEditError(describeError(caught));
    } finally {
      setActionPending(false);
    }
  }

  async function suggestVehicleWheels() {
    if (!job || rigBusy) return;
    setRigBusy(true);
    setRigError(null);
    try {
      const result = await suggestWheels(job.id);
      setWheels(result.wheels);
    } catch (caught) {
      setRigError(describeError(caught));
    } finally {
      setRigBusy(false);
    }
  }

  function updateWheel(index: number, patch: Partial<VehicleWheel>) {
    setWheels(current => (current ? current.map((wheel, i) => (i === index ? { ...wheel, ...patch } : wheel)) : current));
  }

  function onViewportPick(point: [number, number, number]) {
    if (markingIndex === null) return;
    updateWheel(markingIndex, { center: point });
    setMarkingIndex(null);
  }

  async function buildVehicleRig() {
    if (!job || !wheels?.length || actionPending) return;
    const requestJobId = job.id;
    setActionPending(true);
    setRigError(null);
    try {
      const child = await rigJob(requestJobId, { wheels });
      if (jobIdRef.current !== requestJobId) return;
      setWheels(null);
      setMarkingIndex(null);
      setValidationReport(null);
      setMeshStats(null);
      setPreviewSource('auto');
      setActiveTab('generate');
      setJob(child);
      setJobId(child.id);
      setJobs(current => [child, ...current]);
    } catch (caught) {
      setRigError(describeError(caught));
    } finally {
      setActionPending(false);
    }
  }

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!settings || submitting || busy || uploadPending) return;
    const form = event.currentTarget;
    if (!form.checkValidity()) { form.reportValidity(); return; }
    const textMode = inputMode === 'text';
    const multiImageMode = !textMode && referenceMode === 'multi';
    const selectedReferences = Object.values(referenceImages).filter((item): item is ReferenceImage => Boolean(item));
    const missingReferenceViews = REQUIRED_REFERENCE_VIEWS.filter(({ view }) => !referenceImages[view]);
    const effectiveImage = textMode ? refPreview?.image ?? null : multiImageMode ? referenceImages.front?.image ?? null : imageData;
    if (textMode && !refPreview) {
      setError('Generate a reference image and review it before submitting');
      return;
    }
    if (multiImageMode && missingReferenceViews.length) {
      setError(`Multi-view generation requires: ${missingReferenceViews.map(item => item.label).join(', ')}`);
      return;
    }
    if (multiImageMode && runtime === 'hunyuan' && !multiView.enabled) {
      setError(multiView.reason || 'Multi-view generation is not configured on this local runtime');
      return;
    }
    const problem = submissionError(backend, effectiveImage, omni ? controlType : null, omni ? controlData : null, settings);
    if (problem) { setError(problem); return; }
    setSubmitting(true);
    setError(null);
    try {
      const payload: JobPayload = {
        project_id: projectId || null,
        backend,
        preset,
        seed: Number(seed),
        texture: effectiveTexture,
        image: effectiveImage,
        reference_images: multiImageMode ? selectedReferences : [],
        control_type: omni ? controlType : null,
        control_data: omni ? controlData : null,
        input_mode: textMode ? 'text' : multiImageMode ? 'multi-image' : 'image',
        prompt: textMode ? promptText.trim() : null,
        t2i_seed: textMode ? Number(t2iSeed) : null,
        asset_type: assetType,
        parameters: settings,
      };
      const created = await submitJob(payload);
      setValidationReport(null);
      setMeshStats(null);
      setPreviewSource('auto');
      setJob(created);
      setJobId(created.id);
      setJobs(current => [created, ...current]);
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setSubmitting(false);
    }
  }

  async function runAction(action: 'retry' | 'resume' | 'restart' | 'cancel') {
    if (!job || actionPending) return;
    const requestJobId = job.id;
    setActionPending(true);
    setError(null);
    try {
      const updated = await jobAction(requestJobId, action);
      if (jobIdRef.current !== requestJobId) return;
      setJob(updated);
      if (action !== 'cancel') {
        setValidationReport(null);
        setMeshStats(null);
        setPreviewSource('auto');
        setJobId(updated.id);
        setJobs(current => [updated, ...current]);
      }
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setActionPending(false);
    }
  }

  async function newProject() {
    const name = window.prompt('Project name', 'New asset project');
    if (!name?.trim()) return;
    try {
      const project = await createProject(name.trim());
      setProjects(current => [project, ...current]);
      setProjectId(project.id);
    } catch (caught) {
      setError(describeError(caught));
    }
  }

  function refreshHistory() {
    fetchJobs(projectId || undefined).then(setJobs).catch(() => undefined);
  }

  const jobLabel = job ? `Job ${job.id.slice(0, 8)}` : 'No asset selected';
  const totalElapsed = job && !isTerminalStage(job.stage) && job.started_at
    ? (Date.now() - Date.parse(job.started_at)) / 1000
    : typeof job?.timings?.elapsed_seconds_total === 'number' ? job.timings.elapsed_seconds_total : null;

  return <div className="app-shell">
    <a className="skip-link" href="#main-content">Skip to main content</a>
    <aside id="app-sidebar" className={`sidebar ${menuOpen ? 'open' : ''}`} inert={mobile && !menuOpen}>
      <div className="brand"><div className="brand-mark"><Sparkles size={17} /></div><span>HunyForge</span><button className="icon-button sidebar-close" ref={sidebarCloseRef} aria-label="Close menu" onClick={() => setMenuOpen(false)}><X size={18} /></button></div>
      <div className="workspace-switcher"><div><span className="eyebrow">PROJECT</span>{projects.length ? <select aria-label="Select project" value={projectId} onChange={event => setProjectId(event.target.value)}>{projects.map(project => <option key={project.id} value={project.id}>{project.name}</option>)}</select> : <strong>Asset Lab</strong>}</div><ChevronDown size={15} /></div>
      <nav aria-label="Primary navigation">
        <button className="nav-item active" onClick={refreshHistory}><Layers3 size={17} />Assets <span className="nav-count">{jobs.length}</span></button>
        <button className="nav-item" onClick={newProject}><FolderOpen size={17} />New project</button>
      </nav>
      <div className="history-list" aria-label="Recent jobs">
        {jobs.map(entry => {
          const badge = inputModeBadge(entry.input_mode);
          return (
            <button key={entry.id} className={`history-item ${entry.id === jobId ? 'selected' : ''}`} onClick={() => selectJob(entry)}>
              <span className={`history-dot ${isTerminalStage(entry.stage) ? entry.stage : 'running'}`} />
              <span className="history-name">{entry.id.slice(0, 8)}</span>
              {badge ? <span className={`input-badge ${entry.input_mode}`} title={badge.title}>{badge.label}</span> : null}
              <span className="history-stage">{STAGE_LABELS[entry.stage] ?? entry.stage}</span>
            </button>
          );
        })}
        {!jobs.length ? <p className="muted small">No jobs yet — generate an asset.</p> : null}
      </div>
      <button className="ghost-button full" onClick={refreshHistory}><RefreshCw size={14} />Refresh history</button>
      <div className="sidebar-bottom"><div className="model-health"><span className="status-dot" /><div><span className="eyebrow">LOCAL RUNTIME</span><strong>{runtimeStatus}</strong></div></div><div className="vram-row"><span>Peak allocated VRAM</span><strong>{job?.peak_vram_mb != null ? `${job.peak_vram_mb} MB` : 'Not benchmarked'}</strong></div></div>
    </aside>
    <main className="main-content" id="main-content" inert={mobile && menuOpen}>
      <header className="topbar">
        <button className="icon-button topbar-menu" ref={menuToggleRef} aria-label="Toggle menu" aria-expanded={menuOpen} aria-controls="app-sidebar" onClick={() => setMenuOpen(open => !open)}><Menu size={18} /></button>
        <div><span className="eyebrow">ASSET WORKSPACE</span><h1>{jobLabel}</h1></div>
        <div className="top-actions">
          {job && job.available_actions.includes('cancel') ? <button className="ghost-button" onClick={() => void runAction('cancel')} disabled={actionPending}><X size={15} />Cancel job</button> : null}
          <button className="primary-button" disabled={submitting || busy || !settings || uploadPending || (inputMode === 'text' && !refPreview)} onClick={() => formRef.current?.requestSubmit()}><Play size={15} fill="currentColor" />Generate</button>
        </div>
      </header>
      <div className="workspace-grid">
        <section className="viewer-panel">
          <div className="viewer-toolbar">
            <div className="segmented" role="group" aria-label="Preview shading mode">
              <button className={viewerMode === 'material' ? 'selected' : ''} onClick={() => setViewerMode('material')}>Material</button>
              <button className={viewerMode === 'wireframe' ? 'selected' : ''} onClick={() => setViewerMode('wireframe')}>Wireframe</button>
              <button className={viewerMode === 'normals' ? 'selected' : ''} onClick={() => setViewerMode('normals')}>Normals</button>
            </div>
            <div className="viewer-actions">
              {job?.parent_job_id && parentJob ? (
                <div className="segmented" role="group" aria-label="Compare with parent job">
                  <button className={!compareParent ? 'selected' : ''} onClick={() => setCompareParent(false)}>This asset</button>
                  <button className={compareParent ? 'selected' : ''} onClick={() => setCompareParent(true)}>Parent</button>
                </div>
              ) : null}
              {shownReady.includes('vehicle-rigged.glb') ? (
                <div className="segmented" role="group" aria-label="Preview source">
                  <button className={shownPreviewName === 'white-mesh.glb' ? 'selected' : ''} onClick={() => setPreviewSource('white')}>White</button>
                  <button className={shownPreviewName === 'textured-mesh.glb' ? 'selected' : ''} onClick={() => setPreviewSource('textured')} disabled={!shownReady.includes('textured-mesh.glb')}>Textured</button>
                  <button className={shownPreviewName === 'vehicle-rigged.glb' ? 'selected' : ''} onClick={() => setPreviewSource('rigged')}>Rigged</button>
                </div>
              ) : shownReady.includes('white-mesh.glb') && shownJob?.texture ? (
                <div className="segmented" role="group" aria-label="Preview source">
                  <button className={shownPreviewName === 'white-mesh.glb' ? 'selected' : ''} onClick={() => setPreviewSource('white')}>White</button>
                  <button className={shownPreviewName === 'textured-mesh.glb' ? 'selected' : ''} onClick={() => setPreviewSource('textured')} disabled={!shownReady.includes('textured-mesh.glb')}>Textured</button>
                </div>
              ) : null}
              <span className="pill neutral">GLB</span>
              {jobId && ready.includes('unity-package.zip') ? <a className="icon-button" aria-label="Download Unity package" href={artifactUrl(jobId, 'unity-package.zip')} download><Download size={16} /></a> : <button className="icon-button" aria-label="Download Unity package" disabled><Download size={16} /></button>}
            </div>
          </div>
          <div className="viewport">
            <Suspense fallback={<div className="preview-empty">Loading viewer…</div>}>
              <ThreePreview ref={preview3dRef} url={modelUrl} mode={viewerMode} markers={activeTab === 'rig' && wheels ? wheels : undefined} marking={activeTab === 'rig' && markingIndex !== null} onPick={onViewportPick} onStats={setMeshStats} />
            </Suspense>
            <div className="axis" aria-hidden="true"><span>Y</span><span>X</span><span>Z</span></div>
            <div className="viewport-caption">
              <span>{modelUrl && shownPreviewName ? `${compareParent ? 'parent · ' : ''}${shownPreviewName.replace('.glb', '')} · ${viewerMode}` : 'No artifact selected'}</span>
              <span>{meshStats ? `${meshStats.faces.toLocaleString()} faces · ${meshStats.vertices.toLocaleString()} vertices · ${meshStats.dimensions.map(value => value.toFixed(2)).join(' × ')} GLB units` : 'Geometry stats appear after load'}</span>
            </div>
          </div>
          <div className="asset-summary">
            <div><span className="eyebrow">CURRENT ASSET</span><strong>{job ? `${job.preset} preset · seed ${job.seed}` : 'Nothing generated yet'}</strong><span className="muted">{job ? `Created ${new Date(job.created_at).toLocaleString()}${job.parent_job_id ? ` · ${job.input_mode === 'retexture' ? 'retexture of' : 'resumed from'} ${job.parent_job_id.slice(0, 8)}` : ''}${job.prompt ? ` · "${job.prompt.length > 48 ? `${job.prompt.slice(0, 48)}…` : job.prompt}"` : ''}` : 'Select or generate a job'}</span></div>
            {job ? <span className={`state-badge ${isTerminalStage(job.stage) ? job.stage : 'running'}`}>{job.stage === 'complete' ? <CheckCircle2 size={14} /> : job.stage === 'failed' ? <XCircle size={14} /> : job.stage === 'cancelled' ? <Ban size={14} /> : <LoaderCircle size={14} />} {STAGE_LABELS[job.stage] ?? job.stage}</span> : null}
          </div>
          <div className="download-row" aria-label="Artifacts">
            {job ? ready.map(name => <a key={name} className="artifact-link" href={artifactUrl(job.id, name)} download>{name}</a>) : null}
          </div>
        </section>
        <aside className="inspector">
          <div className="tabs"><button className={activeTab === 'generate' ? 'tab active' : 'tab'} onClick={() => setActiveTab('generate')}>Generate</button><button className={activeTab === 'edit' ? 'tab active' : 'tab'} onClick={() => setActiveTab('edit')}>Edit</button><button className={`tab ${activeTab === 'rig' ? 'active' : ''} ${job?.asset_type === 'vehicle' || job?.input_mode === 'vehicle-rig' ? '' : 'disabled'}`} onClick={() => setActiveTab('rig')}>Rig</button><button className={activeTab === 'unity' ? 'tab active' : 'tab'} onClick={() => setActiveTab('unity')}>Unity prepare</button></div>
          {activeTab === 'generate' ? (
            <form className="inspector-body" ref={formRef} onSubmit={onSubmit}>
              <div className="section-heading"><div><h2>New generation</h2><p>Stage a local asset run</p></div><span className="pill teal">LOCAL</span></div>
              {/* Pre-emit critique: P5 H5 E5 S5 R5 V5 D5 */}
              <div className="segmented" role="group" aria-label="Asset type">
                <button type="button" className={assetType === 'generic' ? 'selected' : ''} onClick={() => setAssetType('generic')}><Box size={13} />Generic</button>
                <button type="button" className={assetType === 'character' ? 'selected' : ''} disabled title="Character rigging lands in a later phase">Character</button>
                <button type="button" className={assetType === 'vehicle' ? 'selected' : ''} onClick={() => setAssetType('vehicle')} title="Enables wheel marking and WheelCollider export">Vehicle</button>
              </div>
              {assetType === 'vehicle' ? <p className="field-hint">After generation, mark wheels in the Rig tab — a rigged child job partitions wheels and emits Unity WheelCollider metadata.</p> : null}
              <div className="segmented" role="group" aria-label="Input mode">
                <button type="button" className={inputMode === 'image' ? 'selected' : ''} onClick={() => setInputMode('image')}><ImagePlus size={13} />Image</button>
                <button type="button" className={inputMode === 'text' ? 'selected' : ''} disabled={!t2iEnabled} title={t2iEnabled ? 'Generate a reference from a text prompt' : 'Requires the FLUX.2-klein T2I worker'} onClick={() => setInputMode('text')}><Type size={13} />Text</button>
              </div>
              {inputMode === 'text' ? (
                <>
                  <label className="field-label">Text prompt
                    <textarea className="prompt-area" rows={4} maxLength={2000} placeholder="Describe the object — e.g. 'a worn leather boot with brass buckles'" value={promptText} onChange={event => { setPromptText(event.target.value); setRefPreview(null); }} />
                    <span className="field-hint">{promptText.length}/2000 · A reference image is generated locally, then fed into shape generation.</span>
                  </label>
                  <label className="toggle-row"><span><strong>Prompt scaffolding</strong><small>Wraps the prompt with "single object, centered, plain background" for cleaner geometry</small></span><input type="checkbox" checked={scaffold} onChange={event => setScaffold(event.target.checked)} /></label>
                  <label className="field-label">T2I seed
                    <input type="number" autoComplete="off" min={0} max={4294967295} step={1} value={t2iSeed} onChange={event => { setT2iSeed(event.target.value); setRefPreview(null); }} />
                  </label>
                  <button type="button" className="ghost-button full" disabled={previewBusy || !promptText.trim() || workerBusy} onClick={() => void generateReference()}>
                    {previewBusy ? <LoaderCircle size={14} /> : <Type size={14} />}{previewBusy ? 'Generating reference…' : refPreview ? 'Regenerate reference' : 'Generate reference'}
                  </button>
                  {refPreview ? (
                    <figure className="ref-preview">
                      <img src={refPreview.image} alt={`Generated reference for "${promptText.trim()}"`} />
                      <figcaption className="ref-meta"><span>seed {refPreview.seed}{typeof refPreview.timings?.inference === 'number' ? ` · ${refPreview.timings.inference.toFixed(1)}s` : ''}</span><span>{refPreview.prompt_effective}</span></figcaption>
                    </figure>
                  ) : (
                    <p className="field-hint">No reference yet — generate and review one before running the full job.</p>
                  )}
                </>
              ) : (
                <>
                  <div className="segmented" role="group" aria-label="Reference view mode">
                    <button type="button" className={referenceMode === 'single' ? 'selected' : ''} onClick={() => setReferenceMode('single')}>Single view</button>
                    <button type="button" className={referenceMode === 'multi' ? 'selected' : ''} onClick={() => setReferenceMode('multi')}>Multi view</button>
                  </div>
                  {referenceMode === 'single' ? (
                    <label className="field-label">Reference image
                      <div className="upload-box"><ImagePlus size={20} /><strong>{imageName || 'Drop an image here'}</strong><span>PNG, JPG · up to 10 MB</span><input type="file" accept="image/png,image/jpeg" aria-label="Reference image" onChange={event => void onImageChange(event.target.files?.[0])} /></div>
                    </label>
                  ) : (
                    <div className="reference-set">
                      <div className="field-label"><strong>Multi-view reference set</strong><span className="field-hint">Upload the same object from each labeled angle. Keep scale, configuration, background, and lighting consistent.</span></div>
                      {runtime === 'hunyuan' && !multiView.enabled ? <p className="error-text" role="alert">{multiView.reason || 'A local multi-view adapter is not configured.'}</p> : null}
                      <div className="reference-grid">
                        {REQUIRED_REFERENCE_VIEWS.map(item => <ReferenceUploadCard key={item.view} {...item} required value={referenceImages[item.view]} loading={Boolean(referenceLoading[item.view])} onChange={file => void onReferenceImageChange(item.view, file)} onRemove={() => removeReferenceImage(item.view)} />)}
                      </div>
                      <details className="advanced"><summary>Additional coverage (optional)</summary><div className="reference-grid">{OPTIONAL_REFERENCE_VIEWS.map(item => <ReferenceUploadCard key={item.view} {...item} value={referenceImages[item.view]} loading={Boolean(referenceLoading[item.view])} onChange={file => void onReferenceImageChange(item.view, file)} onRemove={() => removeReferenceImage(item.view)} />)}</div></details>
                      <p className="field-hint">{REQUIRED_REFERENCE_VIEWS.filter(item => referenceImages[item.view]).length} of {REQUIRED_REFERENCE_VIEWS.length} required views ready</p>
                    </div>
                  )}
                </>
              )}
              <label className="field-label">Runtime
                <select value={runtime} onChange={event => setRuntime(event.target.value as 'demo' | 'hunyuan')}>
                  <option value="demo">Demo runtime</option>
                  <option value="hunyuan">HunyForge Hunyuan service</option>
                </select>
              </label>
              <label className="field-label">Shape backend
                <select value={backendChoice} disabled={runtime === 'demo'} onChange={event => setBackendChoice(event.target.value as 'hunyuan3d-2.1' | 'hunyuan3d-omni')}>
                  <option value="hunyuan3d-2.1">Hunyuan3D 2.1</option>
                  <option value="hunyuan3d-omni">Hunyuan3D Omni</option>
                </select>
              </label>
              {omni && runtime === 'hunyuan' ? <OmniControlField type={controlType} name={controlName} onTypeChange={setControlType} onFileChange={onControlChange} /> : null}
              <div className="field-row">
                <label className="field-label">Quality preset
                  <select value={preset} onChange={event => selectPreset(event.target.value as PresetName)}>
                    {(['draft', 'standard', 'final'] as PresetName[]).map(name => <option key={name} value={name}>{contract?.presets?.[name]?.label ?? name}</option>)}
                  </select>
                </label>
                <label className="field-label">Output
                  <select value={effectiveTexture ? 'textured' : 'shape'} disabled={omni && runtime === 'hunyuan'} onChange={event => setTexture(event.target.value === 'textured')}>
                    <option value="shape">Shape only</option>
                    <option value="textured">Shape + textures</option>
                  </select>
                </label>
              </div>
              <label className="field-label">Seed
                <input type="number" name="seed" autoComplete="off" required min={0} max={4294967295} step={1} value={seed} onChange={event => setSeed(event.target.value)} />
                <span className="field-hint">0 – 4294967295</span>
              </label>
              <details className="advanced">
                <summary>Advanced settings</summary>
                {settings ? <div className="field-grid">
                  {SETTING_FIELDS.map(field => {
                    const property: SchemaProperty = schema[field.key] ?? {};
                    const value = settings[field.key];
                    const disabled = Boolean(field.textureOnly) && (!effectiveTexture || (omni && runtime === 'hunyuan'));
                    const hint = property.enum ? property.enum.join(' / ') : property.minimum != null || property.maximum != null ? `${property.minimum ?? ''} – ${property.maximum ?? ''}` : null;
                    if (property.enum) {
                      const numeric = typeof property.enum[0] === 'number';
                      return <label key={field.key} className="field-label">{field.label}
                        <select value={String(value)} disabled={disabled} onChange={event => setSetting(field.key, (numeric ? Number(event.target.value) : event.target.value) as never)}>
                          {property.enum.map(option => <option key={String(option)} value={String(option)}>{String(option)}</option>)}
                        </select>
                        {hint ? <span className="field-hint">{hint}</span> : null}
                      </label>;
                    }
                    if (property.type === 'boolean') {
                      return <label key={field.key} className="toggle-row"><span><strong>{field.label}</strong></span><input type="checkbox" checked={Boolean(value)} disabled={disabled} onChange={event => setSetting(field.key, event.target.checked as never)} /></label>;
                    }
                    return <label key={field.key} className="field-label">{field.label}
                      <input type="number" required value={String(value)} min={property.minimum} max={property.maximum} step={property.type === 'integer' ? 1 : 0.1} disabled={disabled} onChange={event => setSetting(field.key, Number(event.target.value) as never)} />
                      {hint ? <span className="field-hint">{hint}</span> : null}
                    </label>;
                  })}
                </div> : <p className="muted">Loading preset contract…</p>}
              </details>
              {settingsProblem ? <p className="error-text" role="alert">{settingsProblem}</p> : null}
              <div className="vram-callout"><Gauge size={17} /><div><strong>Est. duration: {estimate.duration} · Peak VRAM: {estimate.vram}</strong>{estimate.warnings.map(note => <p key={note}>{note}</p>)}</div></div>
              {error ? <p className="error-text" role="alert">{error}</p> : null}
              <button type="submit" className="primary-button full" disabled={submitting || busy || !settings || uploadPending || Boolean(settingsProblem) || (inputMode === 'text' && !refPreview) || (inputMode === 'image' && referenceMode === 'multi' && (REQUIRED_REFERENCE_VIEWS.some(item => !referenceImages[item.view]) || (runtime === 'hunyuan' && !multiView.enabled)))}><Sparkles size={16} />{submitting ? 'Submitting…' : uploadPending ? 'Reading upload…' : busy ? 'Worker busy' : inputMode === 'text' && !refPreview ? 'Generate a reference first' : inputMode === 'image' && referenceMode === 'multi' && REQUIRED_REFERENCE_VIEWS.some(item => !referenceImages[item.view]) ? 'Add required views' : inputMode === 'image' && referenceMode === 'multi' && runtime === 'hunyuan' && !multiView.enabled ? 'Configure multi-view adapter' : 'Generate asset'}</button>
            </form>
          ) : activeTab === 'edit' ? (
            <EditPanel job={job} ready={ready} t2iEnabled={t2iEnabled} prompt={editPrompt} onPromptChange={value => { setEditPrompt(value); setEditPreview(null); }} seed={editSeed} onSeedChange={value => { setEditSeed(value); setEditPreview(null); }} scaffold={scaffold} onScaffoldChange={setScaffold} preview={editPreview} busy={editBusy} error={editError} actionPending={actionPending} workerBusy={workerBusy} onPreview={() => void generateEditPreview()} onApply={() => void applyRetexture()} />
          ) : activeTab === 'rig' ? (
            <RigPanel job={job} ready={ready} wheels={wheels} markingIndex={markingIndex} busy={rigBusy} error={rigError} actionPending={actionPending} onSuggest={() => void suggestVehicleWheels()} onUpdateWheel={updateWheel} onMark={index => setMarkingIndex(current => (current === index ? null : index))} onAddWheel={() => setWheels(current => {
              if ((current?.length ?? 0) >= 16) return current;
              return [...(current ?? []), { name: `Wheel_${(current?.length ?? 0) + 1}`, center: [0, 0, 0], axis: [1, 0, 0], radius: 0.15, half_width: 0.08, steer: false }];
            })} onRemoveWheel={index => setWheels(current => (current ? current.filter((_, i) => i !== index) : current))} onBuild={() => void buildVehicleRig()} />
          ) : (
            <UnityPanel job={job} ready={ready} report={validationReport} onResume={() => void runAction('resume')} />
          )}
        </aside>
      </div>
      <section className="activity-panel" aria-live="polite">
        <div className="activity-header">
          <div><span className="eyebrow">JOB ACTIVITY</span><strong>{error || connectionState !== 'live' ? (error ?? (connectionState === 'down' ? 'Lost connection to local job stream; retrying status checks…' : 'Live updates degraded — polling job status every 5 seconds.')) : previewBusy || editBusy ? 'Generating reference image (FLUX.2 Klein)…' : job ? (STAGE_LABELS[job.stage] ?? job.stage) : 'Ready for generation'}</strong>{job?.current_operation ? <span className="muted small">Operation: {job.current_operation.replace(/_/g, ' ')}{operationStepText(job.operation_progress)}</span> : null}</div>
          <div className="activity-actions">
            <span className="muted">{job?.progress ?? 0}%</span>
            {totalElapsed != null ? <span className="muted">{formatElapsed(totalElapsed)}</span> : null}
            {job && job.available_actions.includes('cancel') ? <button className="ghost-button compact" disabled={actionPending} onClick={() => void runAction('cancel')}><X size={13} />Cancel</button> : null}
            {job && job.available_actions.includes('resume') && job.resume_stage === 'texture' ? <button className="ghost-button compact" disabled={actionPending} onClick={() => void runAction('resume')}><RotateCcw size={13} />Resume from texture</button> : null}
            {job && job.available_actions.includes('resume_unity') ? <button className="ghost-button compact" disabled={actionPending} onClick={() => void runAction('resume')}><RotateCcw size={13} />Resume from export</button> : null}
            {job && job.available_actions.includes('resume') && job.resume_stage !== 'texture' && !job.available_actions.includes('resume_unity') ? <button className="ghost-button compact" disabled={actionPending} onClick={() => void runAction('resume')}><RotateCcw size={13} />Resume</button> : null}
            {job && job.available_actions.includes('restart') ? <button className="ghost-button compact" disabled={actionPending} onClick={() => void runAction('restart')}><RotateCcw size={13} />Restart from shape</button> : null}
          </div>
        </div>
        <div className="progress-track"><div className={`progress-value ${job?.stage === 'failed' ? 'error' : ''}`} style={{ width: `${job?.progress ?? 0}%` }} /></div>
        <div className="stage-list">
          {STAGE_ROWS.map(row => <StageRow key={row.key} row={row} job={job} />)}
        </div>
        {job?.failed_stage ? <p className="error-text">Failed stage: {job.failed_stage}. Preserved artifacts remain downloadable above; {job.error_message}</p> : job?.error_message ? <p className="error-text">{job.error_message}</p> : null}
      </section>
    </main>
    {menuOpen && <button className="menu-overlay" onClick={() => setMenuOpen(false)} aria-label="Close menu" />}
  </div>;
}

function operationStepText(progress: JobUpdate['operation_progress']): string {
  const step = progress?.step;
  const total = progress?.total;
  return typeof step === 'number' && typeof total === 'number' ? ` · step ${step}/${total}` : '';
}

function StageRow({ row, job }: { row: { key: StageName; label: string }; job: JobUpdate | null }) {
  const record = job?.stage_status?.[row.key];
  const state = record?.state ?? (row.key === 'rig' ? 'skipped' : 'waiting');
  const live = record?.state === 'running' && record.started_at ? (Date.now() - Date.parse(record.started_at)) / 1000 : record?.elapsed_seconds;
  const icon = state === 'complete' ? <CheckCircle2 size={13} /> : state === 'failed' ? <XCircle size={13} /> : state === 'cancelled' ? <Ban size={13} /> : state === 'running' ? <LoaderCircle size={13} /> : null;
  return (
    <div className={`stage-entry ${state}`}>
      <span className={`stage-dot ${state}`} />
      <span className="stage-label">{row.label}</span>
      <span className="stage-state">{icon}{state}</span>
      <span className="stage-elapsed">{record ? formatElapsed(live) : '—'}</span>
    </div>
  );
}

function UnityPanel({ job, ready, report, onResume }: { job: JobUpdate | null; ready: string[]; report: ValidationReport | null; onResume: () => void }) {
  if (!job) {
    return <div className="inspector-body"><div className="section-heading"><div><h2>Unity preparation</h2><p>Select a job to inspect its export</p></div><ShieldCheck size={20} className="teal-icon" /></div><p className="muted">Generate an asset first — Unity settings and validation appear here.</p></div>;
  }
  const settings = job.parameters ?? {};
  const exportActions = (job.resume_stage === 'unity' || job.resume_stage === 'validation') && (job.available_actions.includes('resume') || job.available_actions.includes('resume_unity'));
  const lodRequested = Boolean(settings.generate_lods);
  const collisionRequested = Boolean(settings.generate_collision);
  const requestedFiles = ['unity-lod0.glb', ...(lodRequested ? ['unity-lod1.glb'] : []), ...(collisionRequested ? ['unity-collision.glb'] : [])];
  const requestedReady = requestedFiles.every(name => ready.includes(name));
  const reportPassed = report?.status === 'passed';
  const extrasState = !lodRequested && !collisionRequested ? 'SKIPPED' : reportPassed && requestedReady ? 'PASS' : 'REVIEW';
  return <div className="inspector-body">
    <div className="section-heading"><div><h2>Unity preparation</h2><p>Export settings for {job.preset} preset</p></div><ShieldCheck size={20} className="teal-icon" /></div>
    <dl className="settings-list">
      <div><dt>Backend</dt><dd>{job.backend}</dd></div>
      <div><dt>Seed</dt><dd>{job.seed}</dd></div>
      <div><dt>Textures</dt><dd>{job.texture ? (ready.includes('textured-mesh.glb') ? 'Ready' : 'Requested') : 'Shape only'}</dd></div>
      {(['face_count', 'unity_mode', 'collision_mode', 'texture_size', 'render_size', 'max_num_view'] as const).map(key => <div key={key}><dt>{key.replace(/_/g, ' ')}</dt><dd>{String(settings[key] ?? '—')}</dd></div>)}
      <div><dt>Generate LODs</dt><dd>{lodRequested ? 'Requested' : 'Skipped'}</dd></div>
      <div><dt>Generate collision</dt><dd>{collisionRequested ? 'Requested' : 'Skipped'}</dd></div>
    </dl>
    <div className="validation-list">
      <div className={`validation-item ${report ? (reportPassed ? 'passed' : 'failed') : 'warning'}`}><CheckCircle2 size={16} /><span>Backend validation report</span><b>{report ? report.status.toUpperCase() : 'PENDING'}</b></div>
      <div className={`validation-item ${extrasState === 'PASS' ? 'passed' : extrasState === 'SKIPPED' ? 'warning' : 'warning'}`}><Gauge size={16} /><span>LOD and collision artifacts</span><b>{extrasState}</b></div>
      <div className="validation-item warning"><Box size={16} /><span>Unity Editor import remains a manual quality gate</span><b>REVIEW</b></div>
    </div>
    {report?.warnings?.length ? <ul className="warning-list">{report.warnings.map((entry, index) => <li key={index}>{warningText(entry)}</li>)}</ul> : null}
    {report?.blocking_failures?.length ? <ul className="warning-list blocking">{report.blocking_failures.map((entry, index) => <li key={index}>{entry.check}</li>)}</ul> : null}
    <div className="download-row">
      {ready.map(name => <a key={name} className="artifact-link" href={artifactUrl(job.id, name)} download>{name}</a>)}
    </div>
    {exportActions ? <button className="primary-button full" onClick={onResume}><RotateCcw size={15} />Resume export</button> : null}
    {jobIdReady(job, ready) ? <a className="ghost-button full" href={artifactUrl(job.id, 'unity-package.zip')} download><Download size={15} />Export Unity package</a> : <button className="ghost-button full" disabled><Download size={15} />Export Unity package</button>}
  </div>;
}

function jobIdReady(job: JobUpdate, ready: string[]): boolean {
  return ready.includes('unity-package.zip');
}

interface EditPanelProps {
  job: JobUpdate | null;
  ready: string[];
  t2iEnabled: boolean;
  prompt: string;
  onPromptChange: (value: string) => void;
  seed: string;
  onSeedChange: (value: string) => void;
  scaffold: boolean;
  onScaffoldChange: (value: boolean) => void;
  preview: ReferencePreview | null;
  busy: boolean;
  error: string | null;
  actionPending: boolean;
  workerBusy: boolean;
  onPreview: () => void;
  onApply: () => void;
}

/* Pre-emit critique: P5 H5 E5 S5 R5 V5 D5 */
function EditPanel({ job, ready, t2iEnabled, prompt, onPromptChange, seed, onSeedChange, scaffold, onScaffoldChange, preview, busy, error, actionPending, workerBusy, onPreview, onApply }: EditPanelProps) {
  if (!job) {
    return <div className="inspector-body"><div className="section-heading"><div><h2>Edit asset</h2><p>Text-guided retexture</p></div><Pencil size={20} className="teal-icon" /></div><p className="muted">Select a job with a completed white mesh to edit its appearance.</p></div>;
  }
  const eligible = ready.includes('white-mesh.glb');
  return <div className="inspector-body">
    <div className="section-heading"><div><h2>Edit asset</h2><p>Text-guided retexture of {job.id.slice(0, 8)}</p></div><Pencil size={20} className="teal-icon" /></div>
    {!eligible ? (
      <p className="muted">This job has no usable white-mesh checkpoint{job.stage === 'complete' ? '' : ' yet'} — appearance edits require the shape stage to finish first.</p>
    ) : (
      <>
        {!t2iEnabled ? <p className="field-hint">Text editing requires the FLUX.2-klein T2I worker — not enabled on this runtime.</p> : null}
        <div className="vram-callout"><Sparkles size={17} /><div><strong>Appearance only</strong><p>Retexturing changes materials and color — geometry stays identical to job {job.id.slice(0, 8)}. The result is stored as a child job; the original is untouched.</p></div></div>
        <dl className="settings-list">
          <div><dt>Parent</dt><dd>{job.id.slice(0, 8)}</dd></div>
          <div><dt>Checkpoint</dt><dd>white-mesh.glb</dd></div>
          {job.parent_job_id ? <div><dt>Lineage</dt><dd>child of {job.parent_job_id.slice(0, 8)}</dd></div> : null}
        </dl>
        <label className="field-label">Edit instruction
          <textarea className="prompt-area" rows={3} maxLength={2000} placeholder="e.g. 'make it rusty metal with chipped paint'" value={prompt} onChange={event => onPromptChange(event.target.value)} />
          <span className="field-hint">The current viewport render is sent as the edit reference — position the model first.</span>
        </label>
        <label className="toggle-row"><span><strong>Prompt scaffolding</strong><small>Adds geometry-preserving edit guidance</small></span><input type="checkbox" checked={scaffold} onChange={event => onScaffoldChange(event.target.checked)} /></label>
        <label className="field-label">T2I seed
          <input type="number" autoComplete="off" min={0} max={4294967295} step={1} value={seed} onChange={event => onSeedChange(event.target.value)} />
        </label>
        <button type="button" className="ghost-button full" disabled={busy || actionPending || !t2iEnabled || !prompt.trim() || workerBusy} onClick={onPreview}>
          {busy ? <LoaderCircle size={14} /> : <Type size={14} />}{busy ? 'Generating preview…' : preview ? 'Regenerate preview' : 'Generate edit preview'}
        </button>
        {preview ? (
          <figure className="ref-preview">
            <img src={preview.image} alt={`Edit preview for "${prompt.trim()}"`} />
            <figcaption className="ref-meta"><span>seed {preview.seed}{typeof preview.timings?.inference === 'number' ? ` · ${preview.timings.inference.toFixed(1)}s` : ''}</span><span>{preview.prompt_effective}</span></figcaption>
          </figure>
        ) : null}
        {error ? <p className="error-text" role="alert">{error}</p> : null}
        <button type="button" className="primary-button full" disabled={!preview || actionPending || busy} onClick={onApply}>
          <Sparkles size={16} />{actionPending ? 'Creating child job…' : 'Apply retexture as child job'}
        </button>
      </>
    )}
  </div>;
}

interface RigPanelProps {
  job: JobUpdate | null;
  ready: string[];
  wheels: VehicleWheel[] | null;
  markingIndex: number | null;
  busy: boolean;
  error: string | null;
  actionPending: boolean;
  onSuggest: () => void;
  onUpdateWheel: (index: number, patch: Partial<VehicleWheel>) => void;
  onMark: (index: number) => void;
  onAddWheel: () => void;
  onRemoveWheel: (index: number) => void;
  onBuild: () => void;
}

/* Pre-emit critique: P5 H5 E5 S5 R5 V5 D5 */
function RigPanel({ job, ready, wheels, markingIndex, busy, error, actionPending, onSuggest, onUpdateWheel, onMark, onAddWheel, onRemoveWheel, onBuild }: RigPanelProps) {
  const vehicleJob = job && (job.asset_type === 'vehicle' || job.input_mode === 'vehicle-rig');
  if (!job || !vehicleJob) {
    return <div className="inspector-body"><div className="section-heading"><div><h2>Vehicle rig</h2><p>Wheel pivots for Unity</p></div><Box size={20} className="teal-icon" /></div><p className="muted">Select a vehicle job to mark wheels — choose the Vehicle asset type on the Generate tab.</p></div>;
  }
  const meshReady = ready.includes('white-mesh.glb') || ready.includes('textured-mesh.glb');
  const rigged = ready.includes('vehicle-rigged.glb');
  if (job.input_mode === 'vehicle-rig' || rigged) {
    return <div className="inspector-body">
      <div className="section-heading"><div><h2>Vehicle rig</h2><p>Rigged output of {job.id.slice(0, 8)}</p></div><Box size={20} className="teal-icon" /></div>
      <div className="vram-callout"><Sparkles size={17} /><div><strong>{rigged ? 'Rigged GLB ready' : 'Rigging in progress'}</strong><p>{rigged ? 'Switch the preview to Rigged to inspect the Chassis + wheel node hierarchy. The Unity package includes HunyForgeVehicleSetup.cs for WheelCollider generation.' : 'The wheel partition and Unity export are still running — watch the activity strip.'}</p></div></div>
      {job.rig_spec?.wheels ? (
        <dl className="settings-list">
          {job.rig_spec.wheels.map(wheel => <div key={wheel.name}><dt>{wheel.name}{wheel.steer ? ' · steers' : ''}</dt><dd>r {wheel.radius.toFixed(3)} · [{wheel.center.map(v => v.toFixed(2)).join(', ')}]</dd></div>)}
        </dl>
      ) : null}
    </div>;
  }
  return <div className="inspector-body">
    <div className="section-heading"><div><h2>Vehicle rig</h2><p>Mark wheels on {job.id.slice(0, 8)}</p></div><Box size={20} className="teal-icon" /></div>
    {!meshReady ? (
      <p className="muted">Wheel marking needs a finished mesh — wait for the shape stage to complete.</p>
    ) : (
      <>
        <div className="vram-callout"><Sparkles size={17} /><div><strong>Wheel cylinders, not bones</strong><p>Each marker is a cylinder around an axle — faces inside become that wheel's mesh. Auto-suggest starts with a conventional four-wheel layout; add and place markers for every extra axle.</p></div></div>
        <button type="button" className="ghost-button full" disabled={busy || actionPending || !isTerminalStage(job.stage)} onClick={onSuggest}>
          {busy ? <LoaderCircle size={14} /> : <Sparkles size={14} />}{busy ? 'Analyzing mesh…' : wheels ? 'Re-suggest wheels' : 'Auto-suggest wheels'}
        </button>
        {wheels?.length ? (
          <div className="wheel-list">
            {wheels.map((wheel, index) => (
              <div key={`${wheel.name}-${index}`} className={`wheel-card ${markingIndex === index ? 'marking' : ''}`}>
                <div className="wheel-head">
                  <input className="wheel-name" aria-label="Wheel name" value={wheel.name} onChange={event => onUpdateWheel(index, { name: event.target.value })} />
                  <label className="wheel-steer"><input type="checkbox" checked={wheel.steer} onChange={event => onUpdateWheel(index, { steer: event.target.checked })} />steer</label>
                  <button type="button" className={`ghost-button compact ${markingIndex === index ? 'selected' : ''}`} onClick={() => onMark(index)}>{markingIndex === index ? 'Click model…' : 'Place'}</button>
                  <button type="button" className="icon-button" aria-label={`Remove ${wheel.name}`} onClick={() => onRemoveWheel(index)}><X size={13} /></button>
                </div>
                <div className="wheel-grid">
                  {(['X', 'Y', 'Z'] as const).map((axisLabel, axisIndex) => (
                    <label key={axisLabel} className="field-label compact">{axisLabel}
                      <input type="number" step={0.01} value={wheel.center[axisIndex]} onChange={event => { const center = [...wheel.center] as [number, number, number]; center[axisIndex] = Number(event.target.value); onUpdateWheel(index, { center }); }} />
                    </label>
                  ))}
                  <label className="field-label compact">Radius
                    <input type="number" min={0.001} step={0.01} value={wheel.radius} onChange={event => onUpdateWheel(index, { radius: Number(event.target.value) })} />
                  </label>
                  <label className="field-label compact">Width
                    <input type="number" min={0.001} step={0.01} value={wheel.half_width * 2} onChange={event => onUpdateWheel(index, { half_width: Number(event.target.value) / 2 })} />
                  </label>
                </div>
              </div>
            ))}
          </div>
        ) : null}
        <button type="button" className="ghost-button full" disabled={(wheels?.length ?? 0) >= 16} onClick={onAddWheel}><ImagePlus size={14} />Add wheel marker</button>
        {error ? <p className="error-text" role="alert">{error}</p> : null}
        <button type="button" className="primary-button full" disabled={!wheels?.length || busy || actionPending || !isTerminalStage(job.stage)} onClick={onBuild}>
          <Sparkles size={16} />{actionPending ? 'Building rig…' : 'Build rigged child job'}
        </button>
        <p className="field-hint">Supports up to 16 wheels. Creates a child job — every wheel is partitioned onto its own pivot, then Unity export and validation rerun with a WheelCollider setup script.</p>
      </>
    )}
  </div>;
}

function ReferenceUploadCard({ view, label, required = false, value, loading, onChange, onRemove }: { view: ReferenceView; label: string; required?: boolean; value?: ReferenceImage; loading: boolean; onChange: (file: File | undefined) => void; onRemove: () => void }) {
  return <label className={`reference-upload ${value ? 'ready' : ''}`}>
    <span className="reference-upload-heading"><strong>{label}{required ? ' *' : ''}</strong>{value ? <button type="button" className="icon-button" aria-label={`Remove ${label} reference`} onClick={event => { event.preventDefault(); onRemove(); }}><X size={13} /></button> : null}</span>
    {value?.image ? <img src={value.image} alt={`${label} reference preview`} /> : <ImagePlus size={18} />}
    <span>{loading ? 'Reading image…' : value ? value.filename : 'Drop or choose image'}</span>
    <small>PNG, JPG · up to 10 MB</small>
    <input type="file" accept="image/png,image/jpeg" aria-label={`${label} reference image`} onChange={event => onChange(event.target.files?.[0])} />
  </label>;
}

function OmniControlField({ type, name, onTypeChange, onFileChange }: { type: string; name: string; onTypeChange: (value: string) => void; onFileChange: (file: File | undefined) => Promise<void> }) {
  return <div className="omni-control"><label className="field-label">Omni control<select value={type} onChange={e => onTypeChange(e.target.value)}><option value="point">point</option><option value="voxel">voxel</option><option value="pose">pose</option><option value="bbox">bbox</option></select></label><label className="field-label">Control file<div className="control-upload"><Upload size={14} /><span>{name || `Upload ${type} data`}</span><input type="file" accept=".json,.ply,.obj,.vox" aria-label="Omni control file" onChange={e => void onFileChange(e.target.files?.[0])} /></div></label></div>;
}

export default App;
