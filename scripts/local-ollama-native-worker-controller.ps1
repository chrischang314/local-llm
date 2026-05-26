param(
    [string]$WorkerName = "chris-pc-1",
    [string]$Namespace = "local-llm",
    [string]$SwitchDeployment = "chris-pc-1-ollama-switch",
    [string]$Kubeconfig = "$env:ProgramData\LocalLlmWorker\kubeconfig",
    [string]$KubectlPath = "kubectl",
    [string]$OllamaTaskName = "Local LLM Native Ollama",
    [string]$Endpoint = "http://127.0.0.1:11434/api/tags",
    [int]$PollSeconds = 10,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

function Invoke-Kubectl {
    param([string[]]$Arguments)

    $output = & $KubectlPath --kubeconfig $Kubeconfig @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl failed: $output"
    }
    return $output
}

function Get-DesiredWorkerState {
    $deploymentJson = Invoke-Kubectl -Arguments @("-n", $Namespace, "get", "deployment", $SwitchDeployment, "-o", "json")
    $deployment = $deploymentJson | ConvertFrom-Json
    $replicas = 0
    if ($null -ne $deployment.spec.replicas) {
        $replicas = [int]$deployment.spec.replicas
    }
    return $replicas -gt 0
}

function Get-OllamaRunning {
    try {
        Invoke-RestMethod -Uri $Endpoint -TimeoutSec 5 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Start-Worker {
    Start-ScheduledTask -TaskName $OllamaTaskName
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        if (Get-OllamaRunning) {
            return
        }
    }
}

function Stop-Worker {
    Stop-ScheduledTask -TaskName $OllamaTaskName -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -ieq "ollama.exe" -and $_.CommandLine -match "\bserve\b" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Set-WorkerStatusAnnotation {
    param(
        [string]$Desired,
        [string]$Actual
    )

    $timestamp = (Get-Date).ToUniversalTime().ToString("o")
    Invoke-Kubectl -Arguments @(
        "-n", $Namespace,
        "annotate", "deployment", $SwitchDeployment,
        "local-llm.io/desired-state=$Desired",
        "local-llm.io/actual-state=$Actual",
        "local-llm.io/last-observed-at=$timestamp",
        "--overwrite"
    ) | Out-Null
}

function Sync-Worker {
    $desiredOn = Get-DesiredWorkerState
    $running = Get-OllamaRunning
    $desiredText = if ($desiredOn) { "on" } else { "off" }

    if ($desiredOn -and -not $running) {
        Write-Host "[$(Get-Date -Format o)] Kubernetes switch is ON; starting $WorkerName native Ollama..."
        Start-Worker
        $running = Get-OllamaRunning
    } elseif (-not $desiredOn -and $running) {
        Write-Host "[$(Get-Date -Format o)] Kubernetes switch is OFF; stopping $WorkerName native Ollama..."
        Stop-Worker
        $running = Get-OllamaRunning
    } else {
        Write-Host "[$(Get-Date -Format o)] $WorkerName already matches Kubernetes switch: $desiredText."
    }

    $actualText = if ($running) { "on" } else { "off" }
    $statusKey = "$desiredText/$actualText"
    $now = (Get-Date).ToUniversalTime()
    if (
        $statusKey -ne $script:LastStatusKey -or
        ($now - $script:LastAnnotationAt).TotalSeconds -ge 60
    ) {
        Set-WorkerStatusAnnotation -Desired $desiredText -Actual $actualText
        $script:LastStatusKey = $statusKey
        $script:LastAnnotationAt = $now
    }
}

if (-not (Test-Path $Kubeconfig)) {
    throw "Kubeconfig not found: $Kubeconfig"
}

$script:LastStatusKey = $null
$script:LastAnnotationAt = [datetime]::MinValue

$mutex = [System.Threading.Mutex]::new($false, "Local\LocalLlm-$WorkerName-NativeWorkerController")
if (-not $mutex.WaitOne(0)) {
    Write-Host "A $WorkerName native worker controller is already running."
    exit 0
}

try {
    do {
        try {
            Sync-Worker
        } catch {
            Write-Warning $_.Exception.Message
        }

        if (-not $Once) {
            Start-Sleep -Seconds $PollSeconds
        }
    } while (-not $Once)
} finally {
    if ($mutex) {
        $mutex.ReleaseMutex()
        $mutex.Dispose()
    }
}
