<#
store_keys.ps1 — interactively store API keys into a local .env file (never commit .env)
Usage: run in PowerShell: .\store_keys.ps1

IMPORTANT: Do NOT paste API keys into source files or chat. This script
prompts you interactively and writes keys into a local .env file which is
ignored by git. If you've accidentally exposed keys (for example in chat),
revoke/rotate them immediately with the provider.
#>

param(
    [switch]$NonInteractive,
    [string]$OpenAIKey = $env:OPENAI_API_KEY,
    [string]$XAIKey = $env:XAI_API_KEY,
    [string]$GroqKey = $env:GROQ_API_KEY,
    [int]$Port = $env:SERVER_PORT,
    [string]$LocalIP = $env:LOCAL_IP
)

Write-Host "This helper will store keys into .env in the repo root. The .env file is added to .gitignore by default."

# If called in non-interactive mode, forward to tools/write_env_noninteractive.ps1
if ($NonInteractive) {
    $scriptPath = Join-Path $PSScriptRoot 'tools\write_env_noninteractive.ps1'
    if (-not (Test-Path $scriptPath)) {
        Write-Host "Non-interactive helper not found at $scriptPath" -ForegroundColor Red
        exit 2
    }
    & $scriptPath -OpenAIKey $OpenAIKey -XAIKey $XAIKey -GroqKey $GroqKey -Port $Port -LocalIP $LocalIP
    exit $LASTEXITCODE
}

$envPath = Join-Path $PSScriptRoot '.env'

function Read-Secret([string]$prompt) {
    # Use Read-Host -Prompt so the prompt appears in a consistent way with SecureString input
    $sec = Read-Host -Prompt $prompt -AsSecureString
    return [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
}

# Load existing .env values (for prefilling) and set up logging
$existing = @{}
if (Test-Path $envPath) {
    try {
        # Read lines, strip comments, handle quoted values, and trim whitespace
        Get-Content $envPath -ErrorAction SilentlyContinue | ForEach-Object {
            $line = $_ -replace "^\s*#.*$", ''
            if ($line -match "^\s*([^=]+?)\s*=\s*(.+)\s*$") {
                $k = $Matches[1].Trim()
                $v = $Matches[2].Trim()
                # remove surrounding single or double quotes if present
                if ($v -match '^"(.*)"$') { $v = $Matches[1] }
                elseif ($v -match "^'(.*)'") { $v = $Matches[1] }
                $existing[$k] = $v
            }
        }
    } catch {
        # ignore parse errors
    }
}

$logDir = Join-Path $PSScriptRoot 'tools'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logPath = Join-Path $logDir 'store_keys.log'

function MaskValue([string]$v) {
    if (-not $v) { return '' }
    if ($v.Length -le 8) { return (-join (1..$v.Length | ForEach-Object { '*' })) }
    return $v.Substring(0,4) + '...' + $v.Substring($v.Length -4)
}

function ValidateKey([string]$k, [string]$v) {
    if (-not $v) { return $true } # empty is allowed (user may disable)
    if ($v -match '\s') { Write-Host "Value for $k contains whitespace/newline; unlikely valid."; return $false }
    if ($v -match '["\']') { Write-Host "Value for $k contains quote characters; please enter the raw key without quotes."; return $false }
    if ($v.Length -lt 20) { Write-Host "Value for $k looks too short (<20 chars); please verify."; return $false }
    return $true
}

Write-Host "Press Enter to keep an existing value where shown (value will be masked)."

# Prefill from CLI params if provided (supports scripts that pass values without NonInteractive)
if ($OpenAIKey) { $openai = $OpenAIKey }
if ($XAIKey) { $xai = $XAIKey }
if ($GroqKey) { $groq = $GroqKey }

# Collect keys (prefill with existing masked values when no param provided)
# OPENAI
if (-not $openai) {
    $attempts = 0
    do {
        $openai_prompt = if ($existing.ContainsKey('OPENAI_API_KEY')) { "Enter OPENAI_API_KEY (existing: $(MaskValue $existing['OPENAI_API_KEY'])): " } else { "Enter OPENAI_API_KEY: " }
        $openai = Read-Secret $openai_prompt
        if ((-not $openai) -and $existing.ContainsKey('OPENAI_API_KEY')) { $openai = $existing['OPENAI_API_KEY'] }
        if (-not (ValidateKey 'OPENAI_API_KEY' $openai)) { $attempts++ } else { break }
    } while ($attempts -lt 3)
} else {
    if (-not (ValidateKey 'OPENAI_API_KEY' $openai)) { Write-Host "Provided OPENAI_API_KEY failed validation."; exit 2 }
}

# XAI (optional)
if (-not $xai) {
    $attempts = 0
    do {
        $xai_prompt = if ($existing.ContainsKey('XAI_API_KEY')) { "Enter XAI_API_KEY (existing: $(MaskValue $existing['XAI_API_KEY'])) (or leave empty): " } else { "Enter XAI_API_KEY (or leave empty): " }
        $xai = Read-Secret $xai_prompt
        if ((-not $xai) -and $existing.ContainsKey('XAI_API_KEY')) { $xai = $existing['XAI_API_KEY'] }
        if (-not (ValidateKey 'XAI_API_KEY' $xai)) { $attempts++ } else { break }
    } while ($attempts -lt 3)
} else {
    if (-not (ValidateKey 'XAI_API_KEY' $xai)) { Write-Host "Provided XAI_API_KEY failed validation."; exit 2 }
}

# GROQ (optional)
if (-not $groq) {
    $attempts = 0
    do {
        $groq_prompt = if ($existing.ContainsKey('GROQ_API_KEY')) { "Enter GROQ_API_KEY (existing: $(MaskValue $existing['GROQ_API_KEY'])) (or leave empty): " } else { "Enter GROQ_API_KEY (or leave empty): " }
        $groq = Read-Secret $groq_prompt
        if ((-not $groq) -and $existing.ContainsKey('GROQ_API_KEY')) { $groq = $existing['GROQ_API_KEY'] }
        if (-not (ValidateKey 'GROQ_API_KEY' $groq)) { $attempts++ } else { break }
    } while ($attempts -lt 3)
} else {
    if (-not (ValidateKey 'GROQ_API_KEY' $groq)) { Write-Host "Provided GROQ_API_KEY failed validation."; exit 2 }
}

# Other settings (prefill from existing .env when present)
$port_default = if ($existing.ContainsKey('SERVER_PORT')) { $existing['SERVER_PORT'] } else { '8002' }
$port = Read-Host "SERVER_PORT (default $port_default)"
if (-not $port) { $port = $port_default }
$localip_default = if ($existing.ContainsKey('LOCAL_IP')) { $existing['LOCAL_IP'] } else { '127.0.0.1' }
$localip = Read-Host "LOCAL_IP (default $localip_default)"
if (-not $localip) { $localip = $localip_default }

# Show a summary (mask secrets) and confirm
Write-Host "\nSummary (secrets masked):"
Write-Host "OPENAI_API_KEY: $(MaskValue $openai)"
Write-Host "XAI_API_KEY: $(MaskValue $xai)"
Write-Host "GROQ_API_KEY: $(MaskValue $groq)"
Write-Host "SERVER_PORT: $port"
Write-Host "LOCAL_IP: $localip"

# Offer storage option: .env file (default) or SecretManagement vault
$storage_choice = Read-Host "Choose storage: (E)nv file (default), (S)ecretStore (PowerShell SecretManagement) [E/S]"
$storage_choice = if (-not $storage_choice) { 'E' } else { $storage_choice.Substring(0,1).ToUpper() }
if ($storage_choice -eq 'S') {
    Add-Content -Path $logPath -Value "$(Get-Date -Format o) - STORE_CHOICE - SecretStore" -Encoding UTF8
    # Check for SecretManagement availability
    $sm = Get-Module -ListAvailable -Name Microsoft.PowerShell.SecretManagement -ErrorAction SilentlyContinue
    $ss = Get-Module -ListAvailable -Name Microsoft.PowerShell.SecretStore -ErrorAction SilentlyContinue
    if (-not $sm) {
        $install = Read-Host "SecretManagement module not found. Install Microsoft.PowerShell.SecretManagement & SecretStore to current user? (Y/N) [N]"
        if ($install -and $install.Substring(0,1).ToUpper() -eq 'Y') {
            try {
                Install-Module Microsoft.PowerShell.SecretManagement -Scope CurrentUser -Force -ErrorAction Stop
                Install-Module Microsoft.PowerShell.SecretStore -Scope CurrentUser -Force -ErrorAction Stop
                Write-Host "Installed SecretManagement/SecretStore modules to CurrentUser scope."
            } catch {
                Write-Host "Failed to install SecretManagement modules:" $_.Exception.Message
                Add-Content -Path $logPath -Value "$(Get-Date -Format o) - SECRET_INSTALL_FAILED - $_.Exception.Message" -Encoding UTF8
                return
            }
        } else {
            Write-Host "SecretManagement not installed; aborting secret-store write."
            return
        }
    }
    try {
        Import-Module Microsoft.PowerShell.SecretManagement -ErrorAction Stop
        Import-Module Microsoft.PowerShell.SecretStore -ErrorAction Stop
    } catch {
        Write-Host "Failed to import SecretManagement modules:" $_.Exception.Message
        Add-Content -Path $logPath -Value "$(Get-Date -Format o) - SECRET_IMPORT_FAILED - $_.Exception.Message" -Encoding UTF8
        return
    }
    # Ensure a vault is registered (SecretStore as default)
    $vaults = Get-SecretVault -ErrorAction SilentlyContinue
    if (-not $vaults) {
        try {
            Register-SecretVault -Name 'LocalSecretStore' -ModuleName Microsoft.PowerShell.SecretStore -DefaultVault -ErrorAction Stop
            Write-Host "Registered SecretStore vault 'LocalSecretStore'."
        } catch {
            Write-Host "Failed to register SecretStore vault:" $_.Exception.Message
            Add-Content -Path $logPath -Value "$(Get-Date -Format o) - SECRET_VAULT_REG_FAILED - $_.Exception.Message" -Encoding UTF8
            return
        }
    }

    # Save secrets to the vault (do not log secret contents)
    try {
        if ($openai) { Set-Secret -Name OPENAI_API_KEY -Secret (ConvertTo-SecureString $openai -AsPlainText -Force) -ErrorAction Stop }
        if ($xai) { Set-Secret -Name XAI_API_KEY -Secret (ConvertTo-SecureString $xai -AsPlainText -Force) -ErrorAction Stop }
        if ($groq) { Set-Secret -Name GROQ_API_KEY -Secret (ConvertTo-SecureString $groq -AsPlainText -Force) -ErrorAction Stop }
        Add-Content -Path $logPath -Value "$(Get-Date -Format o) - STORED_IN_SECRETSTORE" -Encoding UTF8
        Write-Host "Secrets stored in SecretManagement vault (LocalSecretStore)."
    } catch {
        Write-Host "Failed to save secrets to vault:" $_.Exception.Message
        Add-Content -Path $logPath -Value "$(Get-Date -Format o) - SECRET_STORE_WRITE_FAILED - $_.Exception.Message" -Encoding UTF8
    }
    return
}

if (Test-Path $envPath) {
    $choice = Read-Host "Existing .env found. Choose: (Y)es overwrite, (B)ackup then overwrite, (N) cancel [default B]"
    if (-not $choice) { $choice = 'B' }
    $choice = $choice.Substring(0,1).ToUpper()
    if ($choice -eq 'N') {
        Write-Host "Aborted by user. No changes made."
        Add-Content -Path $logPath -Value "$(Get-Date -Format o) - ABORT - user cancelled write" -Encoding UTF8
        return
    }
    if ($choice -eq 'B') {
        $bak = "$envPath.bak.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Copy-Item -Path $envPath -Destination $bak -ErrorAction SilentlyContinue
        Write-Host "Existing .env backed up to $bak"
        Add-Content -Path $logPath -Value "$(Get-Date -Format o) - BACKUP - $bak" -Encoding UTF8
    } else {
        Add-Content -Path $logPath -Value "$(Get-Date -Format o) - OVERWRITE without backup" -Encoding UTF8
    }
} else {
    Add-Content -Path $logPath -Value "$(Get-Date -Format o) - CREATE new .env" -Encoding UTF8
}

# write .env (use the values the user entered; do not store example keys in source)
$lines = @()
if ($openai) { $lines += "OPENAI_API_KEY=$openai" }
if ($xai) { $lines += "XAI_API_KEY=$xai" }
if ($groq) { $lines += "GROQ_API_KEY=$groq" }
$lines += "SERVER_PORT=$port"
$lines += "LOCAL_IP=$localip"
$lines += "ENABLE_AI=true"
$lines += "AI_DAILY_CALL_LIMIT=100"
$lines += "AI_MAX_CALLS_PER_MINUTE=10"

# Confirm before persisting (final chance to abort)
$confirm = Read-Host "Proceed to write .env with the above values? (Y/N) [Y]"
$confirm = if (-not $confirm) { 'Y' } else { $confirm.Substring(0,1).ToUpper() }
if ($confirm -ne 'Y') {
    Write-Host "Aborted by user before writing .env. No changes made."
    Add-Content -Path $logPath -Value "$(Get-Date -Format o) - ABORT - user cancelled at final confirmation" -Encoding UTF8
    return
}

# ensure .env is ignored by git (append safely, preserving newline semantics)
$gitignorePath = Join-Path $PSScriptRoot '.gitignore'
try {
    if (Test-Path $gitignorePath) {
        $gi = Get-Content $gitignorePath -Raw -ErrorAction SilentlyContinue
        if ($gi -notmatch "(?m)^\s*\.env\s*$") {
            if ([string]::IsNullOrEmpty($gi)) {
                Set-Content -Path $gitignorePath -Value ".env`n" -Encoding UTF8
            } else {
                # ensure file ends with a newline before appending
                if ($gi -notmatch "`n$") { Add-Content -Path $gitignorePath -Value "`n" -Encoding UTF8 }
                Add-Content -Path $gitignorePath -Value ".env" -Encoding UTF8
            }
            Write-Host "Added .env to .gitignore"
        }
    } else {
        # create a minimal .gitignore including .env
        Set-Content -Path $gitignorePath -Value ".env`n" -Encoding UTF8
        Write-Host "Created .gitignore with .env entry"
    }
} catch {
    # non-fatal: continue even if we can't modify .gitignore
    Write-Host "Could not update .gitignore:" $_.Exception.Message
}

# persist .env without echoing secrets to console
try {
    # If an existing .env exists, create a timestamped backup before overwriting
    if (Test-Path $envPath) {
        $bak = "$envPath.bak.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Copy-Item -Path $envPath -Destination $bak -ErrorAction SilentlyContinue
        Write-Host "Existing .env backed up to $bak"
    }

    # Choose an encoding name compatible with installed PowerShell
    # PowerShell 7+ understands 'utf8' (no BOM); Windows PowerShell 5.1 uses 'UTF8'
    $enc = if ($PSVersionTable.PSVersion.Major -ge 7) { 'utf8' } else { 'UTF8' }

    Set-Content -Path $envPath -Value ($lines -join "`n") -Encoding $enc
    Write-Host ".env written to $envPath (do NOT commit this file)."
} catch {
    Write-Host "Failed to write .env:" $_.Exception.Message
}

    # Optional: tighten NTFS ACL so current user is the only explicit principal (may require admin)
    $acl_choice = Read-Host "Tighten file ACL (make .env accessible only to current user)? (Y/N) [N]"
    if ($acl_choice -and $acl_choice.Substring(0,1).ToUpper() -eq 'Y') {
        try {
            $username = "$env:USERDOMAIN\$env:USERNAME"
            icacls $envPath /inheritance:r /grant:r "$username:(R,W)" /c | Out-Null
            Write-Host "Applied ACL restrictions to $envPath for $username"
            Add-Content -Path $logPath -Value "$(Get-Date -Format o) - ACL_APPLIED - $username" -Encoding UTF8
        } catch {
            Write-Host "Failed to apply ACLs (may require admin):" $_.Exception.Message
            Add-Content -Path $logPath -Value "$(Get-Date -Format o) - ACL_FAILED - $_.Exception.Message" -Encoding UTF8
        }
    }

    # Gentle OpenAI key pattern check (warn if doesn't start like common cloud keys)
    if ($openai -and ($openai -notmatch '^(sk-|oai-|fl-).{10,}')) {
        Write-Host "Warning: OPENAI_API_KEY does not match common provider prefixes (sk-, oai-, fl-). Please verify you pasted the correct key."
        Add-Content -Path $logPath -Value "$(Get-Date -Format o) - WARN - openai key pattern mismatch" -Encoding UTF8
    }
