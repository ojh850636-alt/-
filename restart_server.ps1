$port = 8002
$line = (netstat -ano | Select-String ":$port" | Select-Object -First 1)
if ($line) {
    $parts = $line -split '\s+'
    $pid = $parts[-1]
    if ($pid -match '^\d+$') {
        Write-Output "Killing PID $pid"
        taskkill /PID $pid /F | Out-Null
    }
}

# Start the server using the project's helper script (loads .env)
Write-Output "Starting server via start_with_env.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_with_env.ps1

Start-Sleep -s 1

try {
    (Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3) | ConvertTo-Json
} catch {
    Write-Output 'health-failed'
}

try {
    (Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$port/ai/status" -TimeoutSec 3) | ConvertTo-Json -Depth 4
} catch {
    Write-Output 'ai-status-failed'
}

# Run pytest
python -m pytest -q
