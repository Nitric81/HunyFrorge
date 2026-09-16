param(
  [string]$HunyuanRoot = $env:HUNYUAN_ROOT,
  [int]$Port = 8082,
  [string]$ModelPath = $(if ($env:HUNYUAN_MODEL_PATH) { $env:HUNYUAN_MODEL_PATH } else { 'tencent/Hunyuan3D-2.1' }),
  [string]$PythonPath = $(if ($env:HUNYUAN_PYTHON) { $env:HUNYUAN_PYTHON } else { 'python' })
)

$ErrorActionPreference = 'Stop'
if (-not $HunyuanRoot) { throw 'Set HUNYUAN_ROOT to the official Hunyuan3D checkout before starting the service.' }
$server = Join-Path $HunyuanRoot 'api_server.py'
if (-not (Test-Path -LiteralPath $server)) { throw "Hunyuan API server not found: $server" }

Write-Host "Starting Hunyuan service from $HunyuanRoot"
Write-Host "Model: $ModelPath"
Write-Host 'Low-VRAM mode: enabled'
& $PythonPath $server --host 127.0.0.1 --port $Port --model_path $ModelPath --low_vram_mode
