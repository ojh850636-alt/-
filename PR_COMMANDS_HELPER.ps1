# PR helper for Windows PowerShell
# Usage: .\PR_COMMANDS_HELPER.ps1 -BranchName fix/quick-start -PushToRemote
param(
    [string]$BranchName = "fix/quick-start",
    [switch]$PushToRemote
)

Write-Host "Creating branch: $BranchName"
git checkout -b $BranchName

Write-Host "Staging changes"
git add -A

Write-Host "Committing"
git commit -m "chore: modularize providers and stabilize tests (ci/mock fixes)"

if ($PushToRemote) {
    Write-Host "Pushing to origin/$BranchName"
    git push -u origin $BranchName
    Write-Host "If you have gh installed, create a PR with: gh pr create --fill --body-file PR_BODY_DRAFT.md"
} else {
    Write-Host "Branch created locally. To push run: git push -u origin $BranchName"
}
