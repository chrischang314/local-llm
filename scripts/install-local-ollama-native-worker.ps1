param(
    [string]$WorkerName = "chris-pc-1",
    [string]$Namespace = "local-llm",
    [string]$SwitchDeployment = "chris-pc-1-ollama-switch",
    [string]$InstallRoot = "$env:ProgramData\LocalLlmWorker",
    [string]$OllamaDownloadUrl = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip",
    [string]$ModelsDir = "$env:ProgramData\Ollama\models",
    [string[]]$Models = @("llama3.2:3b"),
    [string]$KubeconfigSource = "$env:USERPROFILE\.kube\config",
    [string]$KubectlPath = "",
    [string]$OllamaTaskName = "Local LLM Native Ollama",
    [string]$OllamaTaskUserId = "SYSTEM",
    [ValidateSet("ServiceAccount", "S4U", "Interactive")]
    [string]$OllamaTaskLogonType = "ServiceAccount",
    [string]$OllamaProfileRoot = "$env:ProgramData\LocalLlmWorker\profile",
    [string]$ControllerTaskName = "Local LLM Native Worker Controller",
    [switch]$SkipDownload,
    [switch]$SkipModelPull,
    [int]$PollSeconds = 10
)

$ErrorActionPreference = "Stop"

function Resolve-KubectlPath {
    param([string]$RequestedPath)

    if ($RequestedPath -and (Test-Path $RequestedPath)) {
        return (Resolve-Path $RequestedPath).Path
    }

    $fromPath = Get-Command kubectl -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }

    $dockerKubectl = Join-Path $env:ProgramFiles "Docker\Docker\resources\bin\kubectl.exe"
    if (Test-Path $dockerKubectl) {
        return $dockerKubectl
    }

    throw "kubectl.exe was not found. Install kubectl or pass -KubectlPath."
}

function Wait-Ollama {
    param([int]$TimeoutSeconds = 90)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 | Out-Null
            return $true
        } catch {
            Start-Sleep -Seconds 3
        }
    }
    return $false
}

New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null

$ollamaZip = Join-Path $InstallRoot "ollama-windows-amd64.zip"
$ollamaDir = Join-Path $InstallRoot "ollama"
if (-not $SkipDownload -or -not (Test-Path (Join-Path $ollamaDir "ollama.exe"))) {
    Invoke-WebRequest -Uri $OllamaDownloadUrl -OutFile $ollamaZip
    if (Test-Path $ollamaDir) {
        Remove-Item -Path $ollamaDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $ollamaDir -Force | Out-Null
    Expand-Archive -Path $ollamaZip -DestinationPath $ollamaDir -Force
}
$ollamaExe = Get-ChildItem -Path $ollamaDir -Recurse -Filter "ollama.exe" | Select-Object -First 1
if (-not $ollamaExe) {
    throw "ollama.exe was not found after extracting $ollamaZip"
}

$startScript = Join-Path $InstallRoot "start-ollama.ps1"
$ollamaLocalAppData = Join-Path $OllamaProfileRoot "AppData\Local"
$ollamaAppData = Join-Path $OllamaProfileRoot "AppData\Roaming"
$ollamaTemp = Join-Path $OllamaProfileRoot "Temp"
New-Item -ItemType Directory -Path $OllamaProfileRoot,$ollamaLocalAppData,$ollamaAppData,$ollamaTemp -Force | Out-Null
$ollamaExeDir = Split-Path $ollamaExe.FullName -Parent
$startScriptContent = @'
$ErrorActionPreference = "Stop"
$launchLog = '__INSTALL_ROOT__\ollama-launch.log'
$stdoutLog = '__INSTALL_ROOT__\ollama-stdout.log'
$stderrLog = '__INSTALL_ROOT__\ollama-stderr.log'
"[$(Get-Date -Format o)] launching native Ollama" | Out-File -FilePath $launchLog -Append -Encoding utf8
$env:USERPROFILE = '__PROFILE_ROOT__'
$env:HOME = $env:USERPROFILE
$env:LOCALAPPDATA = '__LOCAL_APP_DATA__'
$env:APPDATA = '__APP_DATA__'
$env:TEMP = '__TEMP_DIR__'
$env:TMP = $env:TEMP
$env:OLLAMA_HOST = "0.0.0.0:11434"
$env:OLLAMA_MODELS = '__MODELS_DIR__'
New-Item -ItemType Directory -Path $env:OLLAMA_MODELS,$env:USERPROFILE,$env:LOCALAPPDATA,$env:APPDATA,$env:TEMP -Force | Out-Null

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 | Out-Null
    "[$(Get-Date -Format o)] native Ollama is already answering" | Out-File -FilePath $launchLog -Append -Encoding utf8
    exit 0
} catch {
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -ieq "ollama.exe" -and $_.CommandLine -match "\bserve\b" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

$process = Start-Process `
    -FilePath '__OLLAMA_EXE__' `
    -ArgumentList "serve" `
    -WorkingDirectory '__OLLAMA_EXE_DIR__' `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru
Start-Sleep -Seconds 2
if ($process.HasExited) {
    "[$(Get-Date -Format o)] native Ollama exited immediately with $($process.ExitCode)" | Out-File -FilePath $launchLog -Append -Encoding utf8
    throw "native Ollama exited immediately with $($process.ExitCode)"
}
"[$(Get-Date -Format o)] native Ollama launched as pid $($process.Id)" | Out-File -FilePath $launchLog -Append -Encoding utf8
'@
$startScriptContent = $startScriptContent.
    Replace("__INSTALL_ROOT__", $InstallRoot).
    Replace("__PROFILE_ROOT__", $OllamaProfileRoot).
    Replace("__LOCAL_APP_DATA__", $ollamaLocalAppData).
    Replace("__APP_DATA__", $ollamaAppData).
    Replace("__TEMP_DIR__", $ollamaTemp).
    Replace("__MODELS_DIR__", $ModelsDir).
    Replace("__OLLAMA_EXE_DIR__", $ollamaExeDir).
    Replace("__OLLAMA_EXE__", $ollamaExe.FullName)
Set-Content -Path $startScript -Value $startScriptContent -Encoding ascii

$resolvedKubectl = Resolve-KubectlPath -RequestedPath $KubectlPath
$kubeconfigDest = Join-Path $InstallRoot "kubeconfig"
if (-not (Test-Path $KubeconfigSource)) {
    throw "Kubeconfig source not found: $KubeconfigSource"
}
Copy-Item -Path $KubeconfigSource -Destination $kubeconfigDest -Force

$controllerPrincipal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest
$ollamaPrincipal = New-ScheduledTaskPrincipal `
    -UserId $OllamaTaskUserId `
    -LogonType $OllamaTaskLogonType `
    -RunLevel Highest
$taskSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

$ollamaAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$startScript`""
$startupTrigger = New-ScheduledTaskTrigger -AtStartup
Register-ScheduledTask `
    -TaskName $OllamaTaskName `
    -Action $ollamaAction `
    -Trigger $startupTrigger `
    -Principal $ollamaPrincipal `
    -Settings $taskSettings `
    -Description "Runs native Ollama for the Local LLM optional worker." `
    -Force | Out-Null

$controllerScript = Resolve-Path (Join-Path $PSScriptRoot "local-ollama-native-worker-controller.ps1")
$controllerArguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$controllerScript`"",
    "-WorkerName", "`"$WorkerName`"",
    "-Namespace", "`"$Namespace`"",
    "-SwitchDeployment", "`"$SwitchDeployment`"",
    "-Kubeconfig", "`"$kubeconfigDest`"",
    "-KubectlPath", "`"$resolvedKubectl`"",
    "-OllamaTaskName", "`"$OllamaTaskName`"",
    "-PollSeconds", $PollSeconds
) -join " "
$controllerAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $controllerArguments `
    -WorkingDirectory (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Register-ScheduledTask `
    -TaskName $ControllerTaskName `
    -Action $controllerAction `
    -Trigger $startupTrigger `
    -Principal $controllerPrincipal `
    -Settings $taskSettings `
    -Description "Watches the Kubernetes $SwitchDeployment deployment and starts/stops native Ollama." `
    -Force | Out-Null

if (-not (Get-NetFirewallRule -DisplayName "Local-LLM-Ollama-In-TCP" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule `
        -DisplayName "Local-LLM-Ollama-In-TCP" `
        -Direction Inbound `
        -Protocol TCP `
        -LocalPort 11434 `
        -Action Allow | Out-Null
}

Start-ScheduledTask -TaskName $OllamaTaskName
if (-not (Wait-Ollama -TimeoutSeconds 120)) {
    throw "Native Ollama did not become ready on http://127.0.0.1:11434"
}

$env:OLLAMA_HOST = "127.0.0.1:11434"
if (-not $SkipModelPull) {
    $env:OLLAMA_HOST = "127.0.0.1:11434"
    foreach ($model in $Models) {
        & $ollamaExe.FullName pull $model
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to pull Ollama model $model"
        }
    }
}

Start-ScheduledTask -TaskName $ControllerTaskName

[pscustomobject]@{
    WorkerName = $WorkerName
    OllamaExe = $ollamaExe.FullName
    ModelsDir = $ModelsDir
    Kubeconfig = $kubeconfigDest
    OllamaTask = $OllamaTaskName
    ControllerTask = $ControllerTaskName
    Endpoint = "http://127.0.0.1:11434"
}
