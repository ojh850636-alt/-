Param(
    [int]$Port = 8002,
    [int]$WaitSeconds = 2
)

# Starts the server using quick_start.ps1, waits a bit, runs pytest, then stops the server.
$qs = Join-Path $PSScriptRoot 'quick_start.ps1'
if (-not (Test-Path $qs)) {
    Write-Error "quick_start.ps1 not found in tools folder."
    exit 1
}

Write-Output "Starting local server (background)..."
$proc = Start-Process -FilePath pwsh -ArgumentList "-NoProfile -NoLogo -Command \"& '$qs' -Port $Port\"" -PassThru
Start-Sleep -Seconds $WaitSeconds

Write-Output "Running pytest..."
$py = "python -m pytest -q"
& cmd /c $py
$rc = $LASTEXITCODE

Write-Output "Stopping server (if running)..."
try {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
} catch { }

if ($rc -ne 0) { exit $rc } else { exit 0 }
