param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("on", "off", "status", "pull")]
    [string]$Mode,

    [string]$ComposeFile = "docker-compose.pc-worker.yml",
    [string]$Service = "chris-pc-2-ollama",
    [string]$Container = "local-llm-chris-pc-2-ollama",
    [string]$Model = "llama3.2:3b",
    [string]$Endpoint = "http://localhost:11434/api/tags"
)

$ErrorActionPreference = "Stop"

function Show-WorkerStatus {
    docker ps -a `
        --filter "name=$Container" `
        --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

    try {
        Invoke-RestMethod -Uri $Endpoint -TimeoutSec 5 | ConvertTo-Json -Depth 5
    } catch {
        Write-Host "Ollama endpoint unavailable: $($_.Exception.Message)"
    }
}

switch ($Mode) {
    "on" {
        docker volume create local_llm_chris_pc_2_ollama | Out-Null
        docker compose -f $ComposeFile up -d $Service
        Show-WorkerStatus
    }
    "off" {
        docker compose -f $ComposeFile stop $Service
        Show-WorkerStatus
    }
    "status" {
        Show-WorkerStatus
    }
    "pull" {
        docker volume create local_llm_chris_pc_2_ollama | Out-Null
        docker compose -f $ComposeFile up -d $Service
        docker exec $Container ollama pull $Model
        Show-WorkerStatus
    }
}
