param(
    [Parameter(Mandatory = $true)]
    [string[]]$Node,

    [Parameter(Mandatory = $true)]
    [ValidateSet("on", "off", "status")]
    [string]$Mode,

    [string]$Namespace = "local-llm"
)

$ErrorActionPreference = "Stop"

function Get-OptionalLabel {
    param([string]$NodeName)

    $nodeJson = kubectl get node $NodeName -o json | ConvertFrom-Json
    $label = $nodeJson.metadata.labels.PSObject.Properties |
        Where-Object { $_.Name -eq "local-llm.io/optional" } |
        Select-Object -First 1

    if ($null -eq $label) {
        return $null
    }
    return [string]$label.Value
}

function Assert-OptionalNode {
    param([string]$NodeName)

    $optional = Get-OptionalLabel -NodeName $NodeName
    if ($optional -ne "true") {
        throw "Refusing to modify '$NodeName'. Add label local-llm.io/optional=true only to non-essential personal PCs."
    }
}

function Show-NodeStatus {
    param([string]$NodeName)

    kubectl get node $NodeName `
        -o custom-columns="NAME:.metadata.name,READY:.status.conditions[?(@.type=='Ready')].status,SCHEDULABLE:.spec.unschedulable,OPTIONAL:.metadata.labels.local-llm\.io/optional,OLLAMA:.metadata.labels.local-llm\.io/ollama,GAMING:.spec.taints[?(@.key=='local-llm.io/gaming')].value"
}

foreach ($nodeName in $Node) {
    Assert-OptionalNode -NodeName $nodeName

    if ($Mode -eq "status") {
        Show-NodeStatus -NodeName $nodeName
        continue
    }

    if ($Mode -eq "off") {
        Write-Host "Disabling optional Local LLM worker '$nodeName'..."
        kubectl cordon $nodeName
        kubectl taint node $nodeName "local-llm.io/gaming=true:NoExecute" --overwrite
        kubectl delete pod `
            --namespace $Namespace `
            --field-selector "spec.nodeName=$nodeName" `
            --selector "local-llm.io/interruptible=true" `
            --ignore-not-found
        Show-NodeStatus -NodeName $nodeName
        continue
    }

    if ($Mode -eq "on") {
        Write-Host "Enabling optional Local LLM worker '$nodeName'..."
        kubectl taint node $nodeName "local-llm.io/gaming=true:NoExecute-" --ignore-not-found
        kubectl uncordon $nodeName
        Show-NodeStatus -NodeName $nodeName
    }
}
