# Agent Runner Sandbox

The agent runner is a small internal HTTP service for executing bounded commands
inside a Kubernetes sandbox. It is intended for backend-owned agent workflows,
not for direct browser access.

## Security Model

- The runner requires a bearer token in Kubernetes.
- Commands are executed without a shell. The first argv entry must be a bare
  executable name from `RUNNER_ALLOWED_COMMANDS`.
- `cwd` is resolved under `/workspace`; path traversal outside that emptyDir is
  rejected.
- stdin, argv, timeout, and stdout/stderr are bounded by environment settings.
- The pod runs as UID/GID `10001`, drops all Linux capabilities, forbids
  privilege escalation, uses the runtime-default seccomp profile, and mounts a
  read-only root filesystem.
- The Kubernetes service account does not mount an API token.
- The NetworkPolicy denies all egress and allows ingress only from backend pods
  on port `8080`.

## Deploy

Create a random token before applying the runner Deployment:

```powershell
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$token = [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
kubectl -n local-llm create secret generic local-llm-agent-runner-auth `
  --from-literal=token=$token `
  --dry-run=client -o yaml | kubectl apply -f -
```

Apply the sandbox resources:

```powershell
kubectl apply -f .\k8s\local-llm\agent-runner.yaml
kubectl -n local-llm rollout status deployment/local-llm-agent-runner --timeout=180s
```

The backend manifest sets:

- `AGENT_RUNNER_URL=http://local-llm-agent-runner.local-llm.svc.cluster.local:8080`
- `AGENT_RUNNER_TOKEN` from the same Kubernetes secret when present.

If the live backend runs in another namespace, mirror the token secret into that
namespace before wiring backend agent endpoints to the client.

## Canary

Run the canary after the image is available in GHCR and the manifest has been
applied:

```powershell
.\scripts\agent-runner-canary.ps1
```

For a first deploy where the token secret does not exist yet:

```powershell
.\scripts\agent-runner-canary.ps1 -CreateSecretIfMissing -ApplyManifest
```

The canary port-forwards the service, submits:

```json
{"argv":["python","-c","print('agent-runner-canary')"],"timeout_seconds":10}
```

and fails unless the runner returns exit code `0` with the expected stdout.

## Backend Client

Backend code should use `backend/agent_executor_client.py` rather than building
runner HTTP calls inline:

```python
from agent_executor_client import agent_executor_client

result = await agent_executor_client.run(
    ["python", "-c", "print('ok')"],
    timeout_seconds=10,
)
```

The client raises `AgentExecutorNotConfigured` when `AGENT_RUNNER_URL` is absent
and wraps runner HTTP failures as `AgentExecutorRequestError`.
