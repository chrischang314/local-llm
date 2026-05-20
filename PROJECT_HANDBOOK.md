# Local LLM Project Handbook

## Design Overview

The Local LLM app separates model routing from worker ownership. The frontend
lets users choose a model; the backend decides which reachable Ollama backend
should serve it.

Essential always-on capacity lives on stable machines such as the Mac Mini.
Personal Windows PCs are optional capacity: they can add GPU throughput when
available, but the system must keep working when they are off, gaming, or
rebooting.

## Optional Windows Workers

Windows PCs are not joined as native K3s nodes in this setup. Instead, each PC
runs Docker Desktop and an Ollama container locally. Kubernetes represents the
PC with a tiny "switch" Deployment:

- `replicas: 1` means the user wants that PC worker on.
- `replicas: 0` means the user wants that PC worker off.

A scheduled PowerShell controller on the PC watches the switch Deployment and
starts or stops the matching Docker Compose service. It writes status
annotations back to Kubernetes so dashboards can distinguish desired state from
actual state.

## Routing Rules

Backend routing config lives in:

- `k8s/local-llm/ollama-backends-configmap.yaml`
- `k8s/local-llm/live-default-app-overrides.yaml`

The router should only choose a backend that is enabled, reachable, and has the
requested model installed. Optional PC workers should appear after essential
capacity for small models unless a larger GPU is intentionally preferred.

## Health Reporting

The unauthenticated `/health` route is intentionally lightweight enough for
dashboards and launchpad checks. It reports backend status, Ollama model count,
and a compact worker summary derived from the router's backend health cache.
The frontend sidebar uses that same response to show `available/enabled`
workers and active routed requests.

## Operational Safety

Secrets and local runtime state stay out of git. In particular, do not commit
kube tokens, SSH keys, Docker credential files, or `.docker-worker/`.

When adding another Windows worker, copy the CHRIS-PC-1 pattern: add a switch
Deployment, a restricted worker-controller service account, routing config, docs,
and an interactive Windows scheduled task for Docker Desktop plus the controller.
