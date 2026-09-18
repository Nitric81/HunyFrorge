# Queue HunyForge generation jobs for Project Ironhound and import each
# resulting unreal-package.zip into the project. Sequential — the service
# has a single-GPU busy lock (POST /api/jobs returns 409 while busy).
#
# Reliability: the container's worker gets OOM-killed when a second heavy
# job runs against accumulated memory (~21 GB RSS observed), so we restart
# the container before every asset and wait for runtime_ready.
#
# Workarounds baked in:
#  - packaged HunyForgeUnrealSetup.py inside each zip still ships the LOD0
#    pick() bug (fixed copy lives in the repo backend); we overwrite it in
#    the extracted package dir before running setup.
#  - staging dir names must not start with 't' after a backslash ("\t" is
#    parsed as a tab when the editor resolves -ExecutePythonScript); use
#    pkg-* names.

$ErrorActionPreference = 'Stop'
$api = 'http://127.0.0.1:8081'
$projectId = '1ac1e848-585b-4bd1-a286-9290a8f188f6'
$outDir = 'D:\Projects\Software Dev\HunyForge\ironhound-assets'
$setupPs1 = 'D:\Projects\Software Dev\HunyForge\scripts\setup-unreal.ps1'
$fixedSetupPy = 'D:\Projects\Software Dev\HunyForge\backend\unreal\HunyForgeUnrealSetup.py'
$uproject = 'D:\Projects\Software Dev\Project Ironhound\unreal\Project_Ironhound.uproject'
$engine = 'D:\Program Files\UE_5.8'

$style = ', post-apocalyptic wasteland, weathered and rusted, muted desert palette, game asset, three-quarter view'

$assets = @(
    @{ key = 'hub-kiosk';  type = 'generic'; seed = 44;
       prompt = "Small standalone roadside trade kiosk, armored service booth with shuttered window, breaker panel, small roof overhang, exposed cables and pipes, corrugated metal walls$style" },
    @{ key = 'hub-mast';   type = 'generic'; seed = 45;
       prompt = "Tall lattice radio mast tower landmark, rusted steel truss structure with small service platform and antenna array on top, guy-wire anchor lugs at the base$style" },
    @{ key = 'wreck';      type = 'generic'; seed = 46;
       prompt = "Burnt-out rusted cargo truck wreck, collapsed cab with no doors, melted body panels, missing wheels, charred flatbed frame, scattered scrap around it$style" },
    @{ key = 'dog';        type = 'generic'; seed = 47;
       prompt = "Alert wasteland shepherd-mix dog, medium size, upright ears, short sandy-brown coat with darker muzzle and saddle, standing alert pose wearing a simple utility harness$style" },
    @{ key = 'door-slab';  type = 'generic'; seed = 48;
       prompt = "Heavy industrial rolling shutter door, sealed metal doorway gate, corrugated steel slats with faded yellow-black warning stripe along the bottom edge, side guide rails$style" }
)

function Wait-RuntimeReady([int]$Minutes = 8) {
    $deadline = (Get-Date).AddMinutes($Minutes)
    while ((Get-Date) -lt $deadline) {
        try {
            $h = Invoke-RestMethod "$api/health" -TimeoutSec 20
            if ($h.runtime_ready) { return $true }
        } catch { }
        Start-Sleep -Seconds 15
    }
    return $false
}

$results = @()

foreach ($a in $assets) {
    $key = $a.key
    try {
        # 0. fresh container for this asset
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$key] restarting container..."
        docker restart hunyforge | Out-Null
        if (-not (Wait-RuntimeReady)) { throw 'container did not become ready' }
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$key] runtime ready"

        # 1. reference image via local FLUX
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$key] reference-preview..."
        $ref = Invoke-RestMethod -Method Post -Uri "$api/api/reference-preview" `
            -ContentType 'application/json' -TimeoutSec 600 `
            -Body (@{ prompt = $a.prompt; seed = $a.seed; size = 1024 } | ConvertTo-Json)
        if (-not $ref.image) { throw 'reference-preview returned no image' }

        # 2. submit job
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$key] submitting job..."
        $body = @{
            project_id    = $projectId
            backend       = 'hunyuan3d-2.1'
            preset        = 'standard'
            seed          = $a.seed
            texture       = $true
            input_mode    = 'text'
            asset_type    = $a.type
            unreal_export = $true
            t2i_seed      = $a.seed
            prompt        = $a.prompt
            image         = $ref.image
        } | ConvertTo-Json
        $job = Invoke-RestMethod -Method Post -Uri "$api/api/jobs" `
            -ContentType 'application/json' -TimeoutSec 90 -Body $body
        $id = $job.id
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$key] job $id"

        # 3. poll — transient errors tolerated (API stalls under GPU load)
        $last = ''; $deadline = (Get-Date).AddMinutes(40); $stage = ''
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 30
            try {
                $j = Invoke-RestMethod "$api/api/jobs/$id" -TimeoutSec 60
                $stage = $j.stage
                $line = "$($j.stage) $($j.progress)%"
                if ($line -ne $last) {
                    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$key] $line $($j.error_message)"
                    $last = $line
                }
                if ($stage -in @('complete','completed','failed','cancelled','canceled')) { break }
            } catch {
                Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$key] poll timeout (job continues)"
            }
        }
        if ($stage -ne 'complete' -and $stage -ne 'completed') { throw "job ended in stage '$stage'" }

        # 4. download package
        $zip = Join-Path $outDir "$key-unreal-package.zip"
        Invoke-RestMethod "$api/api/jobs/$id/artifacts/unreal-package.zip" -OutFile $zip -TimeoutSec 180
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$key] package downloaded"

        # 5. extract + patch packaged setup script
        $pkg = Join-Path $outDir "pkg-$key"
        Remove-Item $pkg -Recurse -Force -ErrorAction SilentlyContinue
        Expand-Archive -Path $zip -DestinationPath $pkg -Force
        $packaged = Get-ChildItem $pkg -Recurse -Filter 'HunyForgeUnrealSetup.py' | Select-Object -First 1
        if ($packaged) { Copy-Item $fixedSetupPy $packaged.FullName -Force }

        # 6. headless import
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$key] importing into project..."
        & $setupPs1 -ProjectPath $uproject -PackageDir $pkg -EnginePath $engine `
            -ReportPath (Join-Path $outDir "$key-import-report.json") | Select-Object -Last 8
        $rep = Get-Content (Join-Path $outDir "$key-import-report.json") -Raw | ConvertFrom-Json
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$key] import ok=$($rep.ok) verts=$($rep.verification.verts.lod0)/$($rep.verification.verts.lod1) imported=$($rep.imported -join '; ')"
        $results += [pscustomobject]@{ key = $key; job = $id; import_ok = $rep.ok }
    } catch {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$key] FAILED: $($_.Exception.Message)"
        $results += [pscustomobject]@{ key = $key; job = $null; import_ok = $false }
    }
}

Write-Host "`n=== QUEUE SUMMARY ==="
$results | Format-Table -AutoSize
