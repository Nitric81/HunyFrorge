import { useEffect, useState } from 'react';
import { X } from 'lucide-react';
import {
  AppSettings,
  DiskUsage,
  HealthInfo,
  PresetContract,
  PresetName,
  SweepReport,
  describeError,
  fetchSettings,
  formatBytes,
  saveSettings,
  sweepStorage,
} from './job-utils';

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  presets: PresetContract | null;
  health: HealthInfo | null;
  onSaved: (settings: AppSettings) => void;
}

function NumberField({ label, hint, value, min = 0, onChange }: { label: string; hint?: string; value: number; min?: number; onChange: (value: number) => void }) {
  return (
    <label className="field-label">
      {label}
      <input type="number" min={min} value={value} onChange={event => onChange(Number(event.target.value))} />
      {hint ? <span className="field-hint">{hint}</span> : null}
    </label>
  );
}

export function SettingsPanel({ open, onClose, presets, health, onSaved }: SettingsPanelProps) {
  const [form, setForm] = useState<AppSettings | null>(null);
  const [usage, setUsage] = useState<DiskUsage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [sweeping, setSweeping] = useState(false);
  const [report, setReport] = useState<SweepReport | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setError(null);
    fetchSettings()
      .then(response => {
        if (cancelled) return;
        setForm(response.settings);
        setUsage(response.usage);
      })
      .catch(reason => !cancelled && setError(describeError(reason)));
    return () => {
      cancelled = true;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const patch = (changes: Partial<AppSettings>) => setForm(current => (current ? { ...current, ...changes } : current));

  const save = async () => {
    if (!form) return;
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      const response = await saveSettings(form);
      setForm(response.settings);
      setUsage(response.usage);
      onSaved(response.settings);
      setNotice('Settings saved');
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setSaving(false);
    }
  };

  const sweepNow = async () => {
    setSweeping(true);
    setError(null);
    try {
      const result = await sweepStorage();
      setReport(result);
      const refreshed = await fetchSettings();
      setUsage(refreshed.usage);
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setSweeping(false);
    }
  };

  const presetNames = presets ? (Object.keys(presets.presets ?? {}) as PresetName[]) : [];

  return (
    <div className="settings-backdrop" onClick={onClose}>
      <section className="settings-panel" role="dialog" aria-modal="true" aria-label="Settings" onClick={event => event.stopPropagation()}>
        <header className="settings-head">
          <div>
            <span className="eyebrow">HUNYFORGE</span>
            <h2>Settings</h2>
          </div>
          <button className="icon-button" aria-label="Close settings" onClick={onClose}><X size={16} /></button>
        </header>

        {error ? <p className="settings-error" role="alert">{error}</p> : null}
        {notice ? <p className="settings-note" role="status">{notice}</p> : null}
        {!form && !error ? <p className="muted small">Loading settings…</p> : null}

        {form ? (
          <>
            <div className="settings-section">
              <h3>Storage</h3>
              <p className="muted small">
                {usage ? `${formatBytes(usage.total_bytes)} used across ${usage.job_count} job${usage.job_count === 1 ? '' : 's'}` : 'Calculating usage…'}
              </p>
              <div className="field-grid">
                <NumberField label="Keep completed jobs (days)" hint="0 = keep forever" value={form.job_max_age_days} onChange={value => patch({ job_max_age_days: value })} />
                <NumberField label="Keep failed/cancelled jobs (days)" hint="0 = keep forever" value={form.failed_job_max_age_days} onChange={value => patch({ failed_job_max_age_days: value })} />
                <NumberField label="Max data size (GB)" hint="0 = unlimited; oldest terminal jobs evicted first" value={form.data_max_gb} onChange={value => patch({ data_max_gb: value })} />
              </div>
              <button className="ghost-button compact" onClick={() => void sweepNow()} disabled={sweeping}>
                {sweeping ? 'Cleaning…' : 'Clean up now'}
              </button>
              {report ? (
                <p className="settings-note">
                  Removed {report.deleted_jobs.length} job{report.deleted_jobs.length === 1 ? '' : 's'}, freed {formatBytes(report.freed_bytes)}
                  {report.tmp_files_removed ? `, reaped ${report.tmp_files_removed} temp file${report.tmp_files_removed === 1 ? '' : 's'}` : ''}.
                </p>
              ) : null}
              <p className="field-hint">Retention applies to terminal jobs only — a running job is never deleted.</p>
            </div>

            <div className="settings-section">
              <h3>Generation defaults</h3>
              <label className="field-label">
                Default preset
                <select value={form.default_preset ?? ''} onChange={event => patch({ default_preset: (event.target.value || null) as PresetName | null })}>
                  <option value="">App default (standard)</option>
                  {presetNames.map(name => <option key={name} value={name}>{name}</option>)}
                </select>
              </label>
              <label className="toggle-row">
                <input type="checkbox" checked={form.default_unreal_export} onChange={event => patch({ default_unreal_export: event.target.checked })} />
                Enable Unreal package on new jobs
              </label>
            </div>

            <div className="settings-section">
              <h3>Runtime</h3>
              <NumberField label="Min free memory to start a stage (MB)" hint="Stages are rejected with insufficient_memory below this headroom" value={form.min_available_mb} onChange={value => patch({ min_available_mb: value })} />
              <label className="toggle-row">
                <input type="checkbox" checked={form.stage_isolation} onChange={event => patch({ stage_isolation: event.target.checked })} />
                Run each stage in an isolated subprocess
              </label>
              <p className="field-hint">Runtime changes apply to the next stage, not to a stage already running.</p>
            </div>

            <div className="settings-section">
              <h3>About</h3>
              <div className="settings-about">
                <div><span className="muted">Version</span><strong>0.1.0</strong></div>
                <div><span className="muted">Inference mode</span><strong>{health?.inference_mode ?? 'unknown'}</strong></div>
                <div><span className="muted">Runtime</span><strong>{health ? (health.runtime_ready ? 'ready' : 'not ready') : 'unknown'}</strong></div>
                <div><span className="muted">GPU</span><strong>{health?.gpu_available ? 'available' : 'not detected'}</strong></div>
              </div>
            </div>

            <div className="settings-foot">
              <button className="primary-button" onClick={() => void save()} disabled={saving}>{saving ? 'Saving…' : 'Save settings'}</button>
              <button className="ghost-button" onClick={onClose}>Close</button>
            </div>
          </>
        ) : null}
      </section>
    </div>
  );
}
