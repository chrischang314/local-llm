# Kubernetes Personal Worker Nodes

This setup lets personal PCs contribute Ollama capacity when they are available, while keeping essential nodes such as Raspberry Pis and the Mac mini out of the gaming/offline controls.

## Node Labels

Label essential nodes so automation can identify them but never disable them:

```powershell
kubectl label node raspberry-pi-1 local-llm.io/optional=false --overwrite
kubectl label node mac-mini local-llm.io/optional=false --overwrite
```

Label personal PCs as optional LLM workers when they are real Linux K8s nodes:

```powershell
kubectl label node gaming-pc-5080 local-llm.io/optional=true --overwrite
kubectl label node gaming-pc-5080 local-llm.io/ollama=true --overwrite
kubectl label node gaming-pc-5080 local-llm.io/gpu=nvidia --overwrite
kubectl label node gaming-pc-5080 local-llm.io/perf-tier=large --overwrite

kubectl label node spare-pc local-llm.io/optional=true --overwrite
kubectl label node spare-pc local-llm.io/ollama=true --overwrite
kubectl label node spare-pc local-llm.io/perf-tier=medium --overwrite
```

The `scripts/k8s-worker-mode.ps1` script refuses to change any node that does not have `local-llm.io/optional=true`.

## CHRIS-PC-2 Optional Worker

`CHRIS-PC-2` is currently integrated as an external optional Ollama backend instead of a native K8s node because this machine is Windows 11 and only has Docker Desktop's internal WSL distro available. K3s worker nodes need a durable Linux environment, but the backend can still route model requests to this PC over the LAN.

Current endpoint:

```text
http://192.168.4.24:11434
```

Hardware detected:

```text
NVIDIA GeForce RTX 5070, 12 GB VRAM
```

Start it:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local-ollama-worker-mode.ps1 -Mode on
```

Stop it before gaming:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local-ollama-worker-mode.ps1 -Mode off
```

Pull the default routed model:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local-ollama-worker-mode.ps1 -Mode pull -Model llama3.2:3b
```

The backend config map lists `chris-pc-2` first for `llama3.2:3b`, `llama3.1:8b`, and `qwen2.5:14b`. The router still confirms the model is actually installed before choosing the PC.

## Dashboard On/Off Control

`CHRIS-PC-2` also has a Kubernetes dashboard switch:

```text
namespace: local-llm
deployment: chris-pc-2-ollama-switch
```

Scale that deployment from the Kubernetes dashboard:

- `replicas: 1` turns the local PC worker on.
- `replicas: 0` turns the local PC worker off.

The switch pod is intentionally tiny and does not run the model. It gives the Kubernetes control panel a normal Deployment object to scale. A watcher on `CHRIS-PC-2` observes that desired replica count and starts or stops the local Docker Ollama container.

Install the watcher on `CHRIS-PC-2`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-local-ollama-worker-controller.ps1
```

Run one sync manually:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local-ollama-worker-controller.ps1 -Once
```

Check the switch state from kubectl:

```powershell
kubectl -n local-llm get deploy chris-pc-2-ollama-switch
kubectl -n local-llm describe deploy chris-pc-2-ollama-switch
```

## Join a PC to the Cluster

Use the join command from your cluster control plane. For kubeadm clusters it usually looks like:

```powershell
kubeadm token create --print-join-command
```

Run the printed join command on the PC. After the node appears:

```powershell
kubectl get nodes -o wide
kubectl label node <pc-node-name> local-llm.io/optional=true local-llm.io/ollama=true --overwrite
```

For k3s, the control-plane node usually provides a command similar to:

```powershell
curl -sfL https://get.k3s.io | K3S_URL=https://<server>:6443 K3S_TOKEN=<token> sh -
```

## Turning a PC Worker Off for Gaming

```powershell
.\scripts\k8s-worker-mode.ps1 -Node gaming-pc-5080 -Mode off
```

That command:

- Confirms the node is optional.
- Cordons the node.
- Applies `local-llm.io/gaming=true:NoExecute`.
- Deletes interruptible Local LLM pods from that node so GPU/CPU resources are freed.

Bring it back:

```powershell
.\scripts\k8s-worker-mode.ps1 -Node gaming-pc-5080 -Mode on
```

That command removes the gaming taint, uncordons the node, and lets the Ollama DaemonSet recreate the pod.

## LLM Routing

The backend reads `OLLAMA_BACKENDS_FILE` or `OLLAMA_BACKENDS`. The Kubernetes config map in `k8s/local-llm/ollama-backends-configmap.yaml` shows the expected shape.

For example, prioritize the RTX 5080 machine for large models:

```json
{
  "model_preferences": {
    "gemma2:27b": ["gaming-pc-5080"],
    "qwen2.5:14b": ["gaming-pc-5080"],
    "llama3.2:3b": ["mac-mini", "gaming-pc-5080"]
  }
}
```

When a preferred PC is off, NotReady, cordoned, or has Ollama stopped, the backend health probe marks it unavailable and routes to the next viable backend without requiring a frontend change.

## Frontend Behavior

Users still select the model in the UI. The backend decides which worker should serve that model for the current request. The UI does not need to know which machine was used.

For inspection:

```powershell
curl http://localhost/routing/status
```
