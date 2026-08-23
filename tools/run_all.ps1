Param(
  [switch]$SkipDocker
)

Write-Host "Running tests, linter, and optional Docker build..."

$python = "C:/Users/Hi/AppData/Local/Programs/Python/Python313/python.exe"

Write-Host "1) Running tests (pytest)..."
& $python -m pytest -q
$lastTest = $LASTEXITCODE

Write-Host "`n2) Running ruff --fix..."
& $python -m ruff --fix .
$lastRuff = $LASTEXITCODE

if (-not $SkipDocker) {
  Write-Host "`n3) Building Docker image (if docker is available)..."
  if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker build -t lucia-app .
  } else {
    Write-Host "Docker not found; skipping Docker build."
  }
}

Write-Host "`nDone. Test exit:$lastTest, Ruff exit:$lastRuff"
exit $lastTest
