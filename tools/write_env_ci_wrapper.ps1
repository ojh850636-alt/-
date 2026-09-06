<#
write_env_ci_wrapper.ps1 — CI-friendly wrapper to call write_env_noninteractive.ps1

Usage examples:
  # Clear all keys via switches and run writer
  .\write_env_ci_wrapper.ps1 -ClearOpenAI -ClearXAI -ClearGROQ -Port 8002 -LocalIP '127.0.0.1'

This wrapper avoids passing empty string arguments on the command line (which
can be parsed as "missing argument" in some PowerShell invocation contexts).
Instead it sets environment variables explicitly (empty or provided) and then
calls the underlying writer script.
#>

param(
    [switch]$ClearOpenAI,
    [switch]$ClearXAI,
    [switch]$ClearGROQ,
    [int]$Port = $env:SERVER_PORT,
    [string]$LocalIP = $env:LOCAL_IP
)

# set defaults
if (-not $Port) { $Port = 8002 }
if (-not $LocalIP) { $LocalIP = '127.0.0.1' }

# clear env vars explicitly when requested
if ($ClearOpenAI) { $env:OPENAI_API_KEY = '' }
if ($ClearXAI) { $env:XAI_API_KEY = '' }
if ($ClearGROQ) { $env:GROQ_API_KEY = '' }

# call the underlying script from the same tools folder
$scriptPath = Join-Path $PSScriptRoot 'write_env_noninteractive.ps1'
if (-not (Test-Path $scriptPath)) {
    Write-Host "Wrapper: underlying script not found at $scriptPath" -ForegroundColor Red
    exit 2
}

# invoke writer (env variables will be read by the writer)
& $scriptPath -Port $Port -LocalIP $LocalIP
exit $LASTEXITCODE
