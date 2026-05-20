# Local LLM Handoff

## Current State

- `chris-pc-1` is registered as an optional external Ollama worker at
  `http://192.168.4.27:11434`.
- `/health` now includes a `workers` summary with total, enabled, available,
  busy, and loaded model counts. The frontend sidebar renders this as
  model count plus worker availability.
- `chris-pc-1-ollama-switch` lives in the `local-llm` namespace and is currently
  the Kubernetes on/off switch for that PC.
- Login tokens are stored in browser `localStorage`. The backend signs them
  with `JWT_SECRET` when set, otherwise it generates and reuses
  `/app/data/jwt_secret` on the persistent chat-data volume.
- The CHRIS-PC-1 controller runs as the Windows scheduled task
  `Local LLM CHRIS-PC-1 Worker Controller`.
- Docker Desktop is started at logon by the `Start Docker Desktop` scheduled
  task so the controller can use the Docker named pipe from the interactive
  Windows session.
- `llama3.2:3b` is installed on CHRIS-PC-1 and was verified with a direct
  generate request from inside the Kubernetes backend pod.

## Safe Continuation Notes

- Do not commit kubeconfigs, service-account tokens, SSH keys, Docker auth
  files, anything under `.docker-worker/`, or any generated `data/jwt_secret`
  file.
- If an existing browser token was signed before this persistence change, the
  user may need to sign in once after deployment. New tokens should survive tab
  closes and backend restarts until logout or token expiry.
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

## Verification Notes

- Focused health unit coverage lives in `tests/test_health.py`.
- For a live smoke check, call `http://localllm.lan/health` and confirm the
  `workers` object is present, then open the chat UI and inspect the sidebar
  health label.
