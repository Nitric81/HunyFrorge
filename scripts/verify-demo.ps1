param([string]$ApiUrl = 'http://127.0.0.1:8081')

$ErrorActionPreference = 'Stop'
$health = Invoke-RestMethod "$ApiUrl/health"
if ($health.status -ne 'ok') { throw 'HunyForge health check failed.' }
$project = Invoke-RestMethod "$ApiUrl/api/projects" -Method Post -ContentType 'application/json' -Body (@{ name = 'Smoke test project' } | ConvertTo-Json)
$job = Invoke-RestMethod "$ApiUrl/api/jobs" -Method Post -ContentType 'application/json' -Body (@{ project_id = $project.id; backend = 'demo'; texture = $true; seed = 48291 } | ConvertTo-Json)

for ($attempt = 0; $attempt -lt 120; $attempt++) {
  Start-Sleep -Milliseconds 250
  $current = Invoke-RestMethod "$ApiUrl/api/jobs/$($job.id)"
  if ($current.stage -in @('complete', 'failed', 'cancelled')) { break }
}
if ($current.stage -ne 'complete') { throw "Demo job ended in stage: $($current.stage)" }
$validation = Invoke-RestMethod "$ApiUrl/api/jobs/$($job.id)/validation"
if ($validation.status -ne 'passed') { throw 'Unity validation did not pass.' }
$artifacts = Invoke-RestMethod "$ApiUrl/api/jobs/$($job.id)/artifacts"
if ($artifacts.artifacts -notcontains 'artifacts/unity-package.zip') { throw 'Unity package artifact is missing.' }
[pscustomobject]@{ project_id = $project.id; job_id = $job.id; stage = $current.stage; validation = $validation.status; artifact_count = $artifacts.artifacts.Count } | ConvertTo-Json
