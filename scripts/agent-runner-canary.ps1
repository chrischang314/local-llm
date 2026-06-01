param(
    [string]$Namespace = "local-llm",
    [string]$SecretName = "local-llm-agent-runner-auth",
    [string]$Deployment = "local-llm-agent-runner",
    [string]$Service = "local-llm-agent-runner",
    [int]$LocalPort = 18082,
    [string]$Token = $env:AGENT_RUNNER_TOKEN,
    [switch]$ApplyManifest,
    [switch]$CreateSecretIfMissing
)

$ErrorActionPreference = "Stop"

function New-RunnerToken {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($bytes)
    } finally {
        $rng.Dispose()
    }
    return [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
}

function Get-RunnerToken {
    $secretJsonPath = "{.data.token}"
    $encoded = kubectl -n $Namespace get secret $SecretName -o "jsonpath=$secretJsonPath" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $encoded) {
        return $null
    }
    return [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($encoded))
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $repoRoot "k8s\local-llm\agent-runner.yaml"

if (-not $Token) {
    $Token = Get-RunnerToken
}

if (-not $Token -and $CreateSecretIfMissing) {
    $Token = New-RunnerToken
    kubectl -n $Namespace create secret generic $SecretName `
        "--from-literal=token=$Token" `
        --dry-run=client -o yaml | kubectl apply -f -
}

if (-not $Token) {
    throw "No runner token found. Set AGENT_RUNNER_TOKEN or create $Namespace/$SecretName."
}

if ($ApplyManifest) {
    kubectl apply -f $manifest
}

kubectl -n $Namespace rollout status "deployment/$Deployment" --timeout=180s

$stdoutPath = Join-Path $env:TEMP "local-llm-agent-runner-port-forward.out"
$stderrPath = Join-Path $env:TEMP "local-llm-agent-runner-port-forward.err"
$portForward = Start-Process `
    -FilePath "kubectl" `
    -ArgumentList @("-n", $Namespace, "port-forward", "svc/$Service", "${LocalPort}:8080") `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath

try {
    Start-Sleep -Seconds 3
    $headers = @{ Authorization = "Bearer $Token" }
    $body = @{
        argv = @("python", "-c", "print('agent-runner-canary')")
        timeout_seconds = 10
    } | ConvertTo-Json -Depth 5

    $result = Invoke-RestMethod `
        -Method Post `
        -Uri "http://127.0.0.1:$LocalPort/runs" `
        -Headers $headers `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 30

    if ($result.exit_code -ne 0 -or $result.stdout.Trim() -ne "agent-runner-canary") {
        throw "Canary failed: exit=$($result.exit_code), stdout='$($result.stdout)', stderr='$($result.stderr)'"
    }

    $result | ConvertTo-Json -Depth 5
} finally {
    if ($portForward -and -not $portForward.HasExited) {
        Stop-Process -Id $portForward.Id -Force
    }
}
