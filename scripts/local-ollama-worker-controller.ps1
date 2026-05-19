param(
    [string]$WorkerName = "chris-pc-2",
    [string]$Namespace = "local-llm",
    [string]$SwitchDeployment = "chris-pc-2-ollama-switch",
    [string]$ComposeFile = "docker-compose.pc-worker.yml",
    [string]$Service = "chris-pc-2-ollama",
    [string]$Container = "local-llm-chris-pc-2-ollama",
    [int]$PollSeconds = 10,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

function Get-ScriptRoot {
    if ($PSScriptRoot) {
        return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    }
    return (Get-Location).Path
}

function Get-DesiredWorkerState {
    $deploymentJson = kubectl -n $Namespace get deployment $SwitchDeployment -o json 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot read Kubernetes switch deployment '$Namespace/$SwitchDeployment': $deploymentJson"
    }

    $deployment = $deploymentJson | ConvertFrom-Json
    $replicas = 0
    if ($null -ne $deployment.spec.replicas) {
        $replicas = [int]$deployment.spec.replicas
    }
    return $replicas -gt 0
}

function Get-ContainerRunning {
    try {
        $state = docker inspect -f "{{.State.Running}}" $Container 2>$null
        return $state -eq "true"
    } catch {
        return $false
    }
}

function Set-WorkerStatusAnnotation {
    param(
        [string]$Desired,
        [string]$Actual
    )

    $timestamp = (Get-Date).ToUniversalTime().ToString("o")
    $annotateOutput = kubectl -n $Namespace annotate deployment $SwitchDeployment `
        "local-llm.io/desired-state=$Desired" `
        "local-llm.io/actual-state=$Actual" `
        "local-llm.io/last-observed-at=$timestamp" `
        --overwrite 2>&1

    if ($LASTEXITCODE -ne 0) {
        throw "Cannot update Kubernetes switch status annotation '$Namespace/$SwitchDeployment': $annotateOutput"
    }
}

function Start-Worker {
    docker volume create local_llm_chris_pc_2_ollama | Out-Null
    docker compose -f $ComposeFile up -d $Service | Out-Host
}

function Stop-Worker {
    docker compose -f $ComposeFile stop $Service | Out-Host
}

function Sync-Worker {
    $desiredOn = Get-DesiredWorkerState
    $running = Get-ContainerRunning
    $desiredText = if ($desiredOn) { "on" } else { "off" }

    if ($desiredOn -and -not $running) {
        Write-Host "[$(Get-Date -Format o)] Kubernetes switch is ON; starting $WorkerName..."
        Start-Worker
        $running = Get-ContainerRunning
    } elseif (-not $desiredOn -and $running) {
        Write-Host "[$(Get-Date -Format o)] Kubernetes switch is OFF; stopping $WorkerName..."
        Stop-Worker
        $running = Get-ContainerRunning
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

$repoRoot = Get-ScriptRoot
Set-Location $repoRoot
$script:LastStatusKey = $null
$script:LastAnnotationAt = [datetime]::MinValue

$mutex = [System.Threading.Mutex]::new($false, "Local\LocalLlm-$WorkerName-WorkerController")
if (-not $mutex.WaitOne(0)) {
    Write-Host "A $WorkerName worker controller is already running."
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
