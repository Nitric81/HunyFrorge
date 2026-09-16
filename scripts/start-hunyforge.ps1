param([int]$UiPort = 5173)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$api = Start-Process python -ArgumentList '-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8081' -WorkingDirectory $projectRoot -PassThru
$ui = Start-Process npm -ArgumentList 'run','dev','--','--host','127.0.0.1','--port',$UiPort -WorkingDirectory $projectRoot -PassThru
try {
  $ready = $false
  for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
      $health = Invoke-RestMethod "http://127.0.0.1:8081/health" -TimeoutSec 1
      if ($health.status -eq 'ok') { $ready = $true; break }
    } catch { }
    Start-Sleep -Milliseconds 250
  }
  if (-not $ready) { throw 'HunyForge API did not become healthy on port 8081.' }
  Write-Host 'HunyForge API: http://127.0.0.1:8081'
  Write-Host "HunyForge UI:  http://127.0.0.1:$UiPort"
  Write-Host 'Press Ctrl+C to stop both services.'
  Wait-Process -Id $api.Id,$ui.Id
} finally {
  Stop-Process -Id $api.Id,$ui.Id -Force -ErrorAction SilentlyContinue
}
