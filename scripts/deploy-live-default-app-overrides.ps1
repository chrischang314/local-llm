param(
    [string]$AppNamespace = "default",
    [string]$ControlNamespace = "local-llm",
    [string]$BackendDeployment = "local-llm-backend",
    [string]$FrontendDeployment = "local-llm-frontend"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $repoRoot "k8s\local-llm\live-default-app-overrides.yaml"

kubectl apply -f $manifest

kubectl -n $AppNamespace set env "deployment/$BackendDeployment" `
    "KUBERNETES_NAMESPACE=$ControlNamespace" `
    "OLLAMA_BACKENDS-"

$patchPath = Join-Path $env:TEMP "local-llm-backend-live-overrides.yaml"
@"
spec:
  template:
    spec:
      serviceAccountName: local-llm-backend
      containers:
        - name: backend
          envFrom:
            - configMapRef:
                name: local-llm-routing
"@ | Set-Content -LiteralPath $patchPath -Encoding UTF8

kubectl -n $AppNamespace patch deployment $BackendDeployment --type strategic --patch-file $patchPath
kubectl -n $AppNamespace rollout status "deployment/$BackendDeployment" --timeout=180s

if (kubectl -n $AppNamespace get deployment $FrontendDeployment 2>$null) {
    kubectl -n $AppNamespace rollout restart "deployment/$FrontendDeployment"
    kubectl -n $AppNamespace rollout status "deployment/$FrontendDeployment" --timeout=180s
}
