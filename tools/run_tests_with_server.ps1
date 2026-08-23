# Start Lucia server on PORT 8002, run pytest, then stop the server.
# Usage: Open PowerShell in repo root and run: .\tools\run_tests_with_server.ps1

$ErrorActionPreference = 'Stop'
$cwd = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $cwd

Write-Host "Starting lucia server on PORT=8002..."
$env:PORT = '8002'

# Launch using uvicorn so the app object is imported directly from the module
$uvicornArgs = '-m', 'uvicorn', 'lucia_ultimate_quantum_integrated_fixed:app', '--host', '127.0.0.1', '--port', $env:PORT
$proc = Start-Process -FilePath python -ArgumentList $uvicornArgs -WorkingDirectory $cwd -PassThru

# Wait for the server port to become ready (timeout after 15s)
$portReady = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $sock = New-Object System.Net.Sockets.TcpClient
        $async = $sock.BeginConnect('127.0.0.1', [int]$env:PORT, $null, $null)
        $completed = $async.AsyncWaitHandle.WaitOne(500)
        if ($completed -and $sock.Connected) {
            $portReady = $true
            $sock.EndConnect($async)
            $sock.Close()
            break
        }
        $sock.Close()
    } catch {
        # ignore connection attempts that fail
    }
    Start-Sleep -Milliseconds 500
}

if (-not $portReady) {
    Write-Host "ERROR: server did not become ready on port $env:PORT within timeout."
    Write-Host "Stopping lucia server (pid $($proc.Id))..."
    try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
    exit 1
}

try {
    Write-Host "Running pytest (single-threaded)..."
    pytest -q
} finally {
    Write-Host "Stopping lucia server (pid $($proc.Id))..."
    try {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    } catch {
        # ignore
    }
}

Write-Host "Done."
