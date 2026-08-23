# PowerShell helper: stage, commit, and push changes to origin/fix/quick-start
param(
    [string]$Branch = "fix/quick-start",
    [string]$Message = "chore: add lucia_quantum_fusion, run helpers, and quick-start artifacts"
)

Write-Host "Current branch:"; git rev-parse --abbrev-ref HEAD
Write-Host "Staging changes..."; git add -A
Write-Host "Committing..."; git commit -m $Message
if ($LASTEXITCODE -ne 0) {
    Write-Host "Commit failed or nothing to commit. Aborting push."; exit 1
}
Write-Host "Pushing to origin/$Branch..."; git push --set-upstream origin $Branch
if ($LASTEXITCODE -eq 0) {
    Write-Host "Push succeeded"
} else {
    Write-Host "Push failed. Check remote/credentials and try again."; exit 1
}
