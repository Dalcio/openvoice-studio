# OpenVoice Studio installer for Windows (PowerShell 5.1+).
#   Remote: powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Dalcio/openvoice-studio/main/install.ps1 | iex"
#   Local:  $env:OVS_SOURCE="."; powershell -ExecutionPolicy ByPass -File .\install.ps1
# Env: OVS_SOURCE (path or git URL), OVS_SKIP_SETUP=1, OVS_EVALUATION=1, OVS_PYTHON (default 3.11)
$ErrorActionPreference = "Stop"

$Source = $env:OVS_SOURCE
if (-not $Source) { $Source = "git+https://github.com/Dalcio/openvoice-studio" }
elseif (Test-Path $Source) { $Source = (Resolve-Path $Source).Path }
$Python = $env:OVS_PYTHON
if (-not $Python) { $Python = "3.11" }

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Step "Installing uv (astral.sh)"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not found. Install it with: winget install --id Gyan.FFmpeg -e   (then open a new terminal)"
}

Step "Installing openvoice-studio from $Source (Python $Python)"
uv tool install --force --python $Python --reinstall-package openvoice-studio $Source
if ($LASTEXITCODE -ne 0) { throw "uv tool install failed" }

Step "Making sure the uv tool bin directory is on PATH"
uv tool update-shell | Out-Null
$bin = (uv tool dir --bin).Trim()
if (($env:Path -split ";") -notcontains $bin) { $env:Path = "$bin;$env:Path" }

if ($env:OVS_SKIP_SETUP -ne "1") {
    Step "Setup: models + MCP registration in Claude"
    $setupArgs = @("setup", "--yes")
    if ($env:OVS_EVALUATION -eq "1") { $setupArgs += "--evaluation" }
    & openvoice-studio @setupArgs
    if ($LASTEXITCODE -ne 0) { throw "openvoice-studio setup failed" }
}

Step "Doctor"
& openvoice-studio doctor
Write-Host ""
Write-Host "Done. Open the UI with:  openvoice-studio ui" -ForegroundColor Green
Write-Host "(If the command is not found, open a new terminal so PATH is refreshed.)"
