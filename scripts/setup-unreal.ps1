param(
  [string]$ProjectPath,
  [string]$PackageZip,
  [string]$PackageDir,
  [string]$EnginePath,
  [string]$ReportPath,
  [int]$TimeoutSeconds = 1200,
  [switch]$ProbeOnly
)

$ErrorActionPreference = 'Stop'

# HunyForge Unreal bootstrap: resolves the Unreal Engine install, enables the
# plugins HunyForge needs inside the target .uproject (or a disposable probe
# project), extracts an unreal-package.zip if given, then runs
# HunyForgeUnrealSetup.py headless and prints the JSON result report.
#
# Examples:
#   .\scripts\setup-unreal.ps1                                  # probe only
#   .\scripts\setup-unreal.ps1 -ProjectPath D:\UE\Game\Game.uproject -PackageZip .\unreal-package.zip
#   .\scripts\setup-unreal.ps1 -ProjectPath D:\UE\Game          -PackageDir D:\pkg\HunyForge

$repoRoot = Split-Path -Parent $PSScriptRoot
$repoSetupScript = Join-Path $repoRoot 'backend\unreal\HunyForgeUnrealSetup.py'

function Resolve-UProject([string]$path) {
  if (-not $path) { return $null }
  $item = Get-Item $path
  if ($item.PSIsContainer) {
    $found = Get-ChildItem $item.FullName -Filter *.uproject
    if ($found.Count -eq 0) { throw "No .uproject found under $($item.FullName)" }
    if ($found.Count -gt 1) { throw "Multiple .uproject files under $($item.FullName); pass the file path." }
    return $found[0].FullName
  }
  if ($item.Extension -ne '.uproject') { throw "Expected a .uproject path, got: $path" }
  return $item.FullName
}

function Get-InstalledEngines {
  $candidates = @{}
  $roots = 'HKLM:\SOFTWARE\Epic Games\Unreal Engine', 'HKCU:\SOFTWARE\Epic Games\Unreal Engine'
  foreach ($root in $roots) {
    if (Test-Path $root) {
      Get-ChildItem $root | ForEach-Object {
        $key = $_.PSChildName
        $dir = ($_ | Get-ItemProperty -ErrorAction SilentlyContinue).InstalledDirectory
        if ($dir -and (Test-Path $dir)) { $candidates[$key] = $dir }
      }
    }
  }
  $builds = 'HKCU:\SOFTWARE\Epic Games\Unreal Engine\Builds'
  if (Test-Path $builds) {
    $props = Get-ItemProperty $builds
    $props.PSObject.Properties | ForEach-Object {
      if ($_.Value -is [string] -and (Test-Path $_.Value)) { $candidates[$_.Name] = $_.Value }
    }
  }
  foreach ($drive in (Get-PSDrive -PSProvider FileSystem).Root) {
    foreach ($root in @("$($drive)Program Files", "$($drive)Epic Games")) {
      Get-ChildItem $root -Directory -Filter 'UE_*' -ErrorAction SilentlyContinue | ForEach-Object {
        $key = $_.Name -replace '^UE_', ''
        if (-not $candidates.ContainsKey($key)) { $candidates[$key] = $_.FullName }
      }
    }
  }
  return $candidates
}

function Resolve-EngineRoot([string]$association, [hashtable]$engines, [string]$override) {
  if ($override) {
    if (-not (Test-Path (Join-Path $override 'Engine'))) { throw "-EnginePath '$override' has no Engine\ subfolder" }
    return $override
  }
  if ($association -and $engines.ContainsKey($association)) { return $engines[$association] }
  if ($association -and $engines.Count -gt 0) {
    Write-Warning "EngineAssociation '$association' not found locally; using the newest installed engine."
  }
  if ($engines.Count -eq 0) { return $null }
  $best = $engines.GetEnumerator() | Sort-Object {
    $v = $null; [void][version]::TryParse(($_.Key -replace '[^0-9.]', ''), [ref]$v); if ($v) { $v } else { [version]'0.0' }
  } -Descending | Select-Object -First 1
  return $best.Value
}

function Find-NeededPlugins([string]$engineRoot, [string]$projectDir) {
  $found = @{}
  $scan = @((Join-Path $engineRoot 'Engine\Plugins'))
  if ($projectDir) { $scan += (Join-Path $projectDir 'Plugins') }
  foreach ($dir in $scan) {
    if (-not (Test-Path $dir)) { continue }
    Get-ChildItem $dir -Recurse -Filter *.uplugin -ErrorAction SilentlyContinue | ForEach-Object {
      try { $meta = Get-Content $_.FullName -Raw | ConvertFrom-Json } catch { return }
      $name = "$($meta.Name)"
      $label = "$name $($meta.FriendlyName) $($meta.Description)"
      if ($label -match 'python|editorscripting|editor scripting|gltf|interchange' -and $name -notmatch 'test|example|sample|platform|remote|trace|datasmith') {
        $found[$name] = $_.FullName
      }
    }
  }
  return $found
}

function Ensure-UProjectPlugins([string]$uproject, [string[]]$names) {
  $json = Get-Content $uproject -Raw | ConvertFrom-Json
  $plugins = @()
  if ($json.PSObject.Properties['Plugins'] -and $json.Plugins) { $plugins = @($json.Plugins) }
  $existing = @($plugins | ForEach-Object { $_.Name })
  $added = @()
  foreach ($name in $names) {
    if ($existing -notcontains $name) {
      $plugins += [pscustomobject]@{ Name = $name; Enabled = $true }
      $added += $name
    }
  }
  if ($added.Count -eq 0) { return @() }
  $backup = "$uproject.hunyforge-bak"
  if (-not (Test-Path $backup)) { Copy-Item $uproject $backup }
  $json | Add-Member -NotePropertyName Plugins -NotePropertyValue $plugins -Force
  $json | ConvertTo-Json -Depth 32 | Set-Content $uproject -Encoding UTF8
  return $added
}

# --- resolve package -----------------------------------------------------
$stageDir = $null
if ($PackageZip) {
  $stageDir = Join-Path $env:TEMP ("HunyForgeUE_" + [guid]::NewGuid().ToString('N').Substring(0, 8))
  Expand-Archive -Path $PackageZip -DestinationPath $stageDir -Force
  Write-Host "Extracted package to $stageDir"
} elseif ($PackageDir) {
  $stageDir = (Get-Item $PackageDir).FullName
}
# No package content to install -> capability probe only.
if (-not $stageDir) { $ProbeOnly = $true }
$sourceDir = $stageDir
$setupScript = $repoSetupScript
if ($stageDir) {
  $glb = Get-ChildItem $stageDir -Recurse -Filter *.glb | Select-Object -First 1
  if ($glb) { $sourceDir = $glb.DirectoryName }
  $packaged = Get-ChildItem $stageDir -Recurse -Filter 'HunyForgeUnrealSetup.py' | Select-Object -First 1
  if ($packaged) { $setupScript = $packaged.FullName }
}
if (-not (Test-Path $setupScript)) { throw "HunyForgeUnrealSetup.py not found at $setupScript" }

# --- resolve project -----------------------------------------------------
$tempProject = $false
$uproject = Resolve-UProject $ProjectPath
$engines = Get-InstalledEngines
if (-not $uproject) {
  $ProbeOnly = $true
  $engineRoot = Resolve-EngineRoot '' $engines $EnginePath
  if (-not $engineRoot) { throw 'No Unreal Engine install found (registry + drive scan for UE_* folders). Install UE 5.x or pass -EnginePath.' }
  $probeDir = Join-Path $env:TEMP 'HunyForgeUEProbe'
  New-Item -ItemType Directory -Force $probeDir | Out-Null
  $uproject = Join-Path $probeDir 'HunyForgeProbe.uproject'
  $association = ($engines.GetEnumerator() | Where-Object { $_.Value -eq $engineRoot } | Select-Object -First 1).Key
  if (-not $association) { $association = '' }
  [pscustomobject]@{ FileVersion = 3; EngineAssociation = $association; Category = ''; Description = 'HunyForge capability probe project'; Plugins = @() } |
    ConvertTo-Json -Depth 8 | Set-Content $uproject -Encoding UTF8
  $tempProject = $true
  Write-Host "No -ProjectPath given; probing with a disposable project at $probeDir"
} else {
  $json = Get-Content $uproject -Raw | ConvertFrom-Json
  $engineRoot = Resolve-EngineRoot "$($json.EngineAssociation)" $engines $EnginePath
  if (-not $engineRoot) { throw "No Unreal Engine install matches EngineAssociation '$($json.EngineAssociation)'; pass -EnginePath." }
}
$projectDir = Split-Path -Parent $uproject

$editorCmd = Join-Path $engineRoot 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
if (-not (Test-Path $editorCmd)) { $editorCmd = Join-Path $engineRoot 'Engine\Binaries\Win64\UnrealEditor.exe' }
if (-not (Test-Path $editorCmd)) { throw "No UnrealEditor binary under $engineRoot\Engine\Binaries\Win64" }
Write-Host "Engine: $engineRoot"
Write-Host "Project: $uproject"

# --- enable plugins ------------------------------------------------------
$needed = Find-NeededPlugins $engineRoot $projectDir
if ($needed.Count -eq 0) { Write-Warning 'No scripting/glTF plugins were found in this engine install; the probe will report what is missing.' }
$requiredNames = @($needed.Keys | Where-Object { $_ -match 'python|editorscripting|gltf|interchange' })
$added = Ensure-UProjectPlugins $uproject $requiredNames
if ($added.Count -gt 0) { Write-Host "Enabled plugins in .uproject: $($added -join ', ') (backup: $uproject.hunyforge-bak)" }
if ((Get-Process -Name 'UnrealEditor*' -ErrorAction SilentlyContinue) -and -not $tempProject) {
  Write-Warning 'An Unreal Editor is running. Plugin changes require it to be closed before the headless run can load them.'
}

# --- headless run --------------------------------------------------------
if (-not $ReportPath) { $ReportPath = Join-Path ($(if ($sourceDir) { $sourceDir } else { $env:TEMP })) 'hunyforge-unreal-result.json' }
$env:HUNYFORGE_SOURCE_DIR = $sourceDir
$env:HUNYFORGE_REPORT_PATH = $ReportPath
$env:HUNYFORGE_PROBE_ONLY = $(if ($ProbeOnly) { '1' } else { '0' })
$env:HUNYFORGE_QUIT_EDITOR = '1'
$logFile = Join-Path $env:TEMP 'hunyforge-unreal-setup.log'
$args = @("`"$uproject`"", "-ExecutePythonScript=`"$setupScript`"", '-unattended', '-nop4', '-nosplash', '-stdout', '-FullStdoutLogOutput', "-Log=`"$logFile`"")
Write-Host "Running setup $(if ($ProbeOnly) { '(probe only) ' })headless - this can take several minutes on first project open..."
$process = Start-Process -FilePath $editorCmd -ArgumentList $args -PassThru -NoNewWindow
if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
  Stop-Process -Id $process.Id -Force
  throw "Headless Unreal run timed out after $TimeoutSeconds s; see $logFile"
}

# --- report --------------------------------------------------------------
if (-not (Test-Path $ReportPath)) { throw "No result report was written; see editor log at $logFile" }
$report = Get-Content $ReportPath -Raw | ConvertFrom-Json
Write-Host ''
Write-Host "Unreal $($report.engine_version) - result: $(if ($report.ok) { 'OK' } else { 'INCOMPLETE' })  (report: $ReportPath)"
if ($report.capabilities) {
  $report.capabilities.PSObject.Properties | ForEach-Object { Write-Host ("  {0,-22} {1}" -f $_.Name, $_.Value) }
}
foreach ($a in @($report.actions)) { Write-Host "  done: $a" }
foreach ($w in @($report.warnings)) { Write-Warning $w }
foreach ($m in @($report.manual_steps)) { Write-Host "  MANUAL STEP: $m" -ForegroundColor Yellow }
Write-Host "Editor log: $logFile"
if ($report.ok) { exit 0 } else { exit 1 }
