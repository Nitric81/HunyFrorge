$ErrorActionPreference = 'Stop'
$health = Invoke-RestMethod 'http://127.0.0.1:8081/health'
$health | ConvertTo-Json
if ($health.inference_mode -eq 'demo') { Write-Host 'HunyForge is in demo mode.' }
if (-not $health.gpu_available) { Write-Warning 'CUDA is not available in the current Python environment.' }
if (-not $health.model_paths_configured) { Write-Warning 'HUNYUAN_ROOT is not configured.' }
if ($health.model_snapshot_present) { Write-Host "Hunyuan 2.1 weights found: $($health.model_snapshot_path)" }
elseif ($env:HUNYUAN_MODEL_PATH) { Write-Warning "HUNYUAN_MODEL_PATH does not contain the expected 2.1 shape checkpoint: $env:HUNYUAN_MODEL_PATH" }
if (-not $health.runtime_ready -and $health.inference_mode -ne 'demo') { Write-Warning 'Real Hunyuan runtime is not ready: CUDA, source checkout, and weights are all required.' }
if ($health.inference_mode -ne 'demo') {
  $hunyuanUrl = if ($env:HUNYUAN_API_URL) { $env:HUNYUAN_API_URL } else { 'http://127.0.0.1:8082' }
  try { Invoke-WebRequest "$hunyuanUrl/health" -TimeoutSec 3 | Out-Null; Write-Host "Hunyuan service reachable: $hunyuanUrl" }
  catch { Write-Warning "Hunyuan service is not reachable at $hunyuanUrl" }
}
