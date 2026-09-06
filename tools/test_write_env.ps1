<#
tools/test_write_env.ps1 — smoke test for write_env_noninteractive.ps1

This script creates a temporary working directory, copies the noninteractive writer
into it, runs it with test values, and verifies a .env file was created containing
expected keys. Use for local quick checks or in CI.

Exit codes:
  0 - success (file created and contains expected entries)
  1 - .env not created
  2 - validation or writer returned error
  3 - other failures
#>

try {
    $root = Split-Path -Parent $MyInvocation.MyCommand.Definition
    $tmp = Join-Path $root 'tmp_test_env'
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    New-Item -ItemType Directory -Path $tmp | Out-Null

    # Copy writer script into temp dir so it treats temp as repo root
    Copy-Item -Path (Join-Path $root 'write_env_noninteractive.ps1') -Destination $tmp -ErrorAction Stop

    Push-Location $tmp

    # Run the writer with explicit safe test keys
    $openai = 'sk-test-XXXXXXXXXXXXXXXX'
    $groq = 'groq-test-XXXXXXXXXXXXXXXX'
    & .\write_env_noninteractive.ps1 -OpenAIKey $openai -GroqKey $groq -Port 12345 -LocalIP '127.0.0.1'
    $rc = $LASTEXITCODE
    if ($rc -ne 0) { Write-Host "Writer exited with code $rc"; Pop-Location; exit 2 }

    if (-not (Test-Path '.env')) { Write-Host '.env not found' ; Pop-Location; exit 1 }

    $content = Get-Content -Raw -ErrorAction Stop '.env'
    if ($content -notmatch 'OPENAI_API_KEY=sk-test-') { Write-Host 'OPENAI key missing in .env'; Pop-Location; exit 1 }
    if ($content -notmatch 'GROQ_API_KEY=groq-test-') { Write-Host 'GROQ key missing in .env'; Pop-Location; exit 1 }
    if ($content -notmatch 'SERVER_PORT=12345') { Write-Host 'PORT missing/incorrect in .env'; Pop-Location; exit 1 }

    Write-Host 'TEST PASSED: .env created and contains expected entries.'
    Pop-Location
    exit 0
} catch {
    Write-Host 'TEST FAILED:' $_.Exception.Message
    if ($tmp -and (Test-Path $tmp)) { Pop-Location -ErrorAction SilentlyContinue }
    exit 3
}
