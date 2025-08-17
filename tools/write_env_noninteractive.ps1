<#
write_env_noninteractive.ps1 — write a .env from environment variables or CLI args for CI
Usage examples (PowerShell):
  # from env vars
  .\write_env_noninteractive.ps1
  # with explicit values
  .\write_env_noninteractive.ps1 -OpenAIKey 'sk-...' -GroqKey '...' -Port 8002 -LocalIP '127.0.0.1'

Notes:
- This script is intended for automated environments (CI) where interactive prompts are not possible.
- It will NOT echo secret values to stdout.
- It creates a timestamped backup of existing .env before overwriting.
#>

param(
    [string]$OpenAIKey = $env:OPENAI_API_KEY,
    [string]$XAIKey = $env:XAI_API_KEY,
    [string]$GroqKey = $env:GROQ_API_KEY,
    [int]$Port = ([int]($env:SERVER_PORT) -as [int]) ,
    [string]$LocalIP = $env:LOCAL_IP
)

# Respect explicitly passed empty string parameters (e.g. -OpenAIKey "") by
# checking $PSBoundParameters rather than truthiness.
if ($PSBoundParameters.ContainsKey('OpenAIKey')) { $OpenAIKey = $OpenAIKey }
if ($PSBoundParameters.ContainsKey('XAIKey')) { $XAIKey = $XAIKey }
if ($PSBoundParameters.ContainsKey('GroqKey')) { $GroqKey = $GroqKey }
if ($PSBoundParameters.ContainsKey('Port')) { $Port = $Port }
if ($PSBoundParameters.ContainsKey('LocalIP')) { $LocalIP = $LocalIP }

if (-not $Port) { $Port = 8002 }
if (-not $LocalIP) { $LocalIP = '127.0.0.1' }

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$envPath = Join-Path $repoRoot '.env'
$logDir = Join-Path $repoRoot 'tools'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logPath = Join-Path $logDir 'store_keys.log'

function Validate-SecretValue($k, $v) {
    if (-not $v) { return $true }
    if ($v -match '\s') { Write-Host "[ERROR] $k contains whitespace"; return $false }
    # Avoid regex/quote patterns which can be fragile in different PowerShell invocation contexts.
    # Use IndexOf to detect quote characters safely.
    if (($v.IndexOf('"') -ge 0) -or ($v.IndexOf("'") -ge 0)) { Write-Host "[ERROR] $k contains quote characters"; return $false }
    if ($v.Length -lt 8) { Write-Host "[WARN] $k looks short" }
    return $true
}

if (-not (Validate-SecretValue 'OPENAI_API_KEY' $OpenAIKey)) { exit 2 }
if (-not (Validate-SecretValue 'XAI_API_KEY' $XAIKey)) { exit 2 }
if (-not (Validate-SecretValue 'GROQ_API_KEY' $GroqKey)) { exit 2 }

# Build lines without printing secrets
$lines = @()
if ($OpenAIKey) { $lines += "OPENAI_API_KEY=$OpenAIKey" }
if ($XAIKey) { $lines += "XAI_API_KEY=$XAIKey" }
if ($GroqKey) { $lines += "GROQ_API_KEY=$GroqKey" }
$lines += "SERVER_PORT=$Port"
$lines += "LOCAL_IP=$LocalIP"
$lines += "ENABLE_AI=true"
$lines += "AI_DAILY_CALL_LIMIT=100"
$lines += "AI_MAX_CALLS_PER_MINUTE=10"

try {
    if (Test-Path $envPath) {
        $bak = "$envPath.bak.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Copy-Item -Path $envPath -Destination $bak -ErrorAction SilentlyContinue
        Add-Content -Path $logPath -Value "$(Get-Date -Format o) - BACKUP - $bak" -Encoding UTF8
    } else {
        Add-Content -Path $logPath -Value "$(Get-Date -Format o) - CREATE new .env" -Encoding UTF8
    }

    $enc = if ($PSVersionTable.PSVersion.Major -ge 7) { 'utf8' } else { 'UTF8' }
    Set-Content -Path $envPath -Value ($lines -join "`n") -Encoding $enc
    Add-Content -Path $logPath -Value "$(Get-Date -Format o) - WROTE_ENV (noninteractive)" -Encoding UTF8
    # Ensure .env is in .gitignore
    $gitignorePath = Join-Path $repoRoot '.gitignore'
    if (Test-Path $gitignorePath) {
        $gi = Get-Content $gitignorePath -Raw -ErrorAction SilentlyContinue
        if ($gi -notmatch "(?m)^\s*\.env\s*$") {
            if ([string]::IsNullOrEmpty($gi)) { Set-Content -Path $gitignorePath -Value ".env`n" -Encoding UTF8 }
            else {
                if ($gi -notmatch "`n$") { Add-Content -Path $gitignorePath -Value "`n" -Encoding UTF8 }
                Add-Content -Path $gitignorePath -Value ".env" -Encoding UTF8
            }
            Add-Content -Path $logPath -Value "$(Get-Date -Format o) - ADDED_GITIGNORE_ENV" -Encoding UTF8
        }
    } else {
        Set-Content -Path $gitignorePath -Value ".env`n" -Encoding UTF8
        Add-Content -Path $logPath -Value "$(Get-Date -Format o) - CREATED_GITIGNORE" -Encoding UTF8
    }

    Write-Host "Wrote .env (secrets not shown); see $logPath for details." -ForegroundColor Green
    exit 0
} catch {
    Write-Host "Failed to write .env: $_.Exception.Message" -ForegroundColor Red
    Add-Content -Path $logPath -Value "$(Get-Date -Format o) - ERROR - $_.Exception.Message" -Encoding UTF8
    exit 3
}
