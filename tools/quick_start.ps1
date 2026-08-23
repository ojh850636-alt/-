Param(
    [int]$Port = 8002
)

$python = "python"
$module = "lucia_ultimate_quantum_integrated_fixed:app"
$args = "-m uvicorn $module --host 127.0.0.1 --port $Port"

Write-Output "Starting server: $python $args"
$proc = Start-Process -FilePath $python -ArgumentList $args -WindowStyle Hidden -PassThru

$timeout = 30
$elapsed = 0
$open = $false
Write-Output "Waiting up to $timeout seconds for http://127.0.0.1:$Port to become available..."
while ($elapsed -lt $timeout) {
    try {
        $open = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -InformationLevel Quiet
    } catch {
        $open = $false
    }
    if ($open) { break }
    Start-Sleep -Seconds 1
    $elapsed++
}

if ($open) {
    Write-Output "Server is up: http://127.0.0.1:$Port"
    Start-Process "http://127.0.0.1:$Port/docs"
    Write-Output "To stop the server run: Stop-Process -Id $($proc.Id)"
} else {
    Write-Output "Server did not become available after $timeout seconds. See uvicorn output in the started process."
    if ($proc -and -not $proc.HasExited) { Write-Output "Server PID: $($proc.Id)" }
}
