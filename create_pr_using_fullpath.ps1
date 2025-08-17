# Create PR using full path to gh.exe if gh is not on PATH
$ghFull = 'C:\Program Files\GitHub CLI\gh.exe'
$bodyFile = Join-Path (Get-Location) 'PR_BODY_FINAL.md'
if (-Not (Test-Path $ghFull)) {
    Write-Error "gh not found at $ghFull; please install or add to PATH."
    exit 1
}

& $ghFull pr create --fill --body-file $bodyFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "gh pr create failed with code $LASTEXITCODE"
} else {
    Write-Host "PR created (or gh interactive)."
}
