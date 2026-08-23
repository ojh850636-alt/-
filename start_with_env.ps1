# start_with_env.ps1
# Loads .env (if present) and starts the FastAPI server with those environment variables.
# Usage: .\start_with_env.ps1

$envFile = Join-Path $PSScriptRoot '.env'
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*#') { return }
        if ($_ -match '^\s*$') { return }
        $parts = $_ -split '=', 2
        if ($parts.Count -eq 2) {
            $k = $parts[0].Trim()
            $v = $parts[1].Trim()
            if ($v -ne '') {
                # Use Set-Item to set process-scoped environment variable safely
                Set-Item -Path "Env:\$k" -Value $v -ErrorAction SilentlyContinue
            }
        }
    }
} else {
    Write-Output "No .env file found; using defaults or already-set environment variables."
}

# fallbacks
if (-not $env:SERVER_PORT) { $env:SERVER_PORT = '8002' }
if (-not $env:LOCAL_IP) { $env:LOCAL_IP = '127.0.0.1' }
if (-not $env:OPENAI_API_KEY) { $env:OPENAI_API_KEY = '' }

# start server in background
Write-Output "Starting server on $($env:LOCAL_IP):$($env:SERVER_PORT) (uvicorn)"
Start-Process -NoNewWindow -FilePath python -ArgumentList '-m','uvicorn','lucia_ultimate_quantum_integrated:app','--host',$env:LOCAL_IP,'--port',$env:SERVER_PORT,'--log-level','info' -PassThru | Out-Null
Write-Output "Server start command issued. Use Get-Process or netstat to confirm."
