# Local LLM Handoff

## Current State

- `chris-pc-1` is registered as an optional external Ollama worker at
  `http://192.168.4.27:11434`.
- `chris-pc-1-ollama-switch` lives in the `local-llm` namespace and is currently
  the Kubernetes on/off switch for that PC.
- The CHRIS-PC-1 controller runs as the Windows scheduled task
  `Local LLM CHRIS-PC-1 Worker Controller`.
- Docker Desktop is started at logon by the `Start Docker Desktop` scheduled
  task so the controller can use the Docker named pipe from the interactive
  Windows session.
- `llama3.2:3b` is installed on CHRIS-PC-1 and was verified with a direct
  generate request from inside the Kubernetes backend pod.

## Safe Continuation Notes

- Do not commit kubeconfigs, service-account tokens, SSH keys, Docker auth
  files, or anything under `.docker-worker/`.
- If CHRIS-PC-1 appears offline, first check Docker Desktop and the scheduled
  tasks on that PC, then check the switch annotations:

```powershell
kubectl -n local-llm get deploy chris-pc-1-ollama-switch -o yaml
```

- To reapply routing after config changes:

```powershell
kubectl apply -f .\k8s\local-llm\ollama-backends-configmap.yaml
.\scripts\deploy-live-default-app-overrides.ps1
```

- To refresh the CHRIS-PC-1 controller files, copy the scripts to
  `C:\Users\chris\Projects\local-llm\scripts` on that host and rerun
  `install-local-ollama-worker-controller.ps1` with the CHRIS-PC-1 parameters in
  `docs/k8s-personal-workers.md`.
