"""Kubernetes launcher for agentic code jobs."""

import base64
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import httpx

from agent_services import callback_token
from models import AgentJob

K8S_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
K8S_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"


@dataclass(frozen=True)
class AgentLaunchResult:
    job_name: str
    namespace: str


def sandbox_namespace() -> str:
    return os.getenv("AGENT_SANDBOX_NAMESPACE", "local-llm-sandbox")


def runner_image() -> str:
    return os.getenv(
        "AGENT_RUNNER_IMAGE",
        "ghcr.io/chrischang314/local-llm/agent-runner:main",
    )


def internal_backend_url() -> str:
    return os.getenv(
        "AGENT_INTERNAL_BACKEND_URL",
        "http://backend.default.svc.cluster.local:8000",
    )


def agent_secret() -> str:
    return os.getenv("AGENT_SECRET_KEY", "")


def _k8s_base_url() -> str:
    host = os.getenv("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
    port = os.getenv("KUBERNETES_SERVICE_PORT", "443")
    return f"https://{host}:{port}"


async def _k8s_request(
    method: str,
    path: str,
    *,
    body: dict | None = None,
    content_type: str = "application/json",
) -> dict[str, Any]:
    if not os.path.exists(K8S_TOKEN_PATH):
        raise RuntimeError("Kubernetes service account token is unavailable")
    token = pathlib.Path(K8S_TOKEN_PATH).read_text(encoding="utf-8").strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": content_type,
    }
    verify: str | bool = K8S_CA_PATH if os.path.exists(K8S_CA_PATH) else True
    async with httpx.AsyncClient(timeout=15.0, verify=verify) as client:
        response = await client.request(
            method,
            f"{_k8s_base_url()}{path}",
            headers=headers,
            json=body,
        )
        response.raise_for_status()
        return response.json() if response.content else {}


def _resource_name(job_id: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]", "-", job_id.lower())
    return f"agent-job-{cleaned[:24]}"


def _b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


class KubernetesAgentExecutor:
    async def launch(self, job: AgentJob, github_token: str) -> AgentLaunchResult:
        if not agent_secret():
            raise RuntimeError("AGENT_SECRET_KEY is required before agent jobs can run")
        namespace = sandbox_namespace()
        job_name = _resource_name(job.id)
        secret_name = f"{job_name}-spec"
        owner, repo = job.repo_full_name.split("/", 1)

        await _k8s_request(
            "POST",
            f"/api/v1/namespaces/{namespace}/secrets",
            body={
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": secret_name,
                    "labels": {
                        "app.kubernetes.io/name": "local-llm-agent-runner",
                        "local-llm.io/agent-job-id": job.id,
                    },
                },
                "type": "Opaque",
                "data": {
                    "github-token": _b64(github_token),
                    "agent-callback-token": _b64(callback_token(job.id)),
                    "task": _b64(job.task),
                },
            },
        )

        env = [
            {"name": "AGENT_JOB_ID", "value": job.id},
            {"name": "REPO_FULL_NAME", "value": job.repo_full_name},
            {"name": "REPO_OWNER", "value": owner},
            {"name": "REPO_NAME", "value": repo},
            {"name": "BASE_BRANCH", "value": job.base_branch},
            {"name": "WORK_BRANCH", "value": job.work_branch or f"agent/{job.id[:12]}"},
            {"name": "MODEL", "value": job.model},
            {"name": "TEST_COMMAND", "value": job.test_command or ""},
            {"name": "HOME", "value": "/workspace/home"},
            {"name": "TMPDIR", "value": "/workspace/tmp"},
            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
            {"name": "LOCAL_LLM_API_URL", "value": f"{internal_backend_url()}/v1/chat/completions"},
            {"name": "LOCAL_LLM_CALLBACK_URL", "value": f"{internal_backend_url()}/agent/internal/jobs/{job.id}"},
            {"name": "GITHUB_TOKEN_FILE", "value": "/agent-secrets/github-token"},
            {"name": "AGENT_CALLBACK_TOKEN_FILE", "value": "/agent-secrets/agent-callback-token"},
            {"name": "TASK_FILE", "value": "/agent-secrets/task"},
        ]
        init_env = [
            {
                "name": "GITHUB_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "github-token"}},
            },
            {
                "name": "AGENT_CALLBACK_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "agent-callback-token"}},
            },
            {
                "name": "TASK",
                "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "task"}},
            },
        ]

        job_body = {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": job_name,
                    "labels": {
                        "app.kubernetes.io/name": "local-llm-agent-runner",
                        "local-llm.io/agent-job-id": job.id,
                    },
                },
                "spec": {
                    "backoffLimit": 0,
                    "activeDeadlineSeconds": 1800,
                    "ttlSecondsAfterFinished": 600,
                    "template": {
                        "metadata": {
                            "labels": {
                                "app.kubernetes.io/name": "local-llm-agent-runner",
                                "local-llm.io/agent-job-id": job.id,
                            }
                        },
                        "spec": {
                            "automountServiceAccountToken": False,
                            "restartPolicy": "Never",
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 10001,
                                "runAsGroup": 10001,
                                "fsGroup": 10001,
                                "seccompProfile": {"type": "RuntimeDefault"},
                            },
                            "initContainers": [
                                {
                                    "name": "copy-secrets",
                                    "image": "busybox:1.36",
                                    "imagePullPolicy": "IfNotPresent",
                                    "env": init_env,
                                    "command": [
                                        "/bin/sh",
                                        "-c",
                                        (
                                            "set -eu; umask 077; "
                                            "printf '%s' \"$GITHUB_TOKEN\" > /agent-secrets/github-token; "
                                            "printf '%s' \"$AGENT_CALLBACK_TOKEN\" > /agent-secrets/agent-callback-token; "
                                            "printf '%s' \"$TASK\" > /agent-secrets/task"
                                        ),
                                    ],
                                    "resources": {
                                        "requests": {
                                            "cpu": "10m",
                                            "memory": "32Mi",
                                            "ephemeral-storage": "16Mi",
                                        },
                                        "limits": {
                                            "cpu": "100m",
                                            "memory": "64Mi",
                                            "ephemeral-storage": "32Mi",
                                        },
                                    },
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "capabilities": {"drop": ["ALL"]},
                                        "readOnlyRootFilesystem": True,
                                    },
                                    "volumeMounts": [
                                        {"name": "agent-secrets", "mountPath": "/agent-secrets"}
                                    ],
                                }
                            ],
                            "containers": [
                                {
                                    "name": "runner",
                                    "image": runner_image(),
                                    "imagePullPolicy": "Always",
                                    "env": env,
                                    "resources": {
                                        "requests": {
                                            "cpu": "250m",
                                            "memory": "512Mi",
                                            "ephemeral-storage": "1Gi",
                                        },
                                        "limits": {
                                            "cpu": "2",
                                            "memory": "4Gi",
                                            "ephemeral-storage": "10Gi",
                                        },
                                    },
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "capabilities": {"drop": ["ALL"]},
                                        "readOnlyRootFilesystem": True,
                                    },
                                    "volumeMounts": [
                                        {"name": "workspace", "mountPath": "/workspace"},
                                        {"name": "agent-secrets", "mountPath": "/agent-secrets"},
                                    ],
                                }
                            ],
                            "volumes": [
                                {"name": "workspace", "emptyDir": {"sizeLimit": "10Gi"}},
                                {"name": "agent-secrets", "emptyDir": {"medium": "Memory", "sizeLimit": "1Mi"}},
                            ],
                        },
                    },
                },
            }
        try:
            created = await _k8s_request(
                "POST",
                f"/apis/batch/v1/namespaces/{namespace}/jobs",
                body=job_body,
            )
        except Exception:
            await self.cleanup_secret(job.id)
            raise

        job_uid = created.get("metadata", {}).get("uid")
        if job_uid:
            try:
                await _k8s_request(
                    "PATCH",
                    f"/api/v1/namespaces/{namespace}/secrets/{secret_name}",
                    body={
                        "metadata": {
                            "ownerReferences": [
                                {
                                    "apiVersion": "batch/v1",
                                    "kind": "Job",
                                    "name": job_name,
                                    "uid": job_uid,
                                    "controller": False,
                                    "blockOwnerDeletion": False,
                                }
                            ]
                        }
                    },
                    content_type="application/merge-patch+json",
                )
            except (RuntimeError, httpx.HTTPError):
                pass
        return AgentLaunchResult(job_name=job_name, namespace=namespace)

    async def cleanup_secret(self, job_id: str) -> None:
        namespace = sandbox_namespace()
        secret_name = f"{_resource_name(job_id)}-spec"
        try:
            await _k8s_request("DELETE", f"/api/v1/namespaces/{namespace}/secrets/{secret_name}")
        except (RuntimeError, httpx.HTTPError):
            pass

    async def delete_job(self, job_id: str) -> None:
        namespace = sandbox_namespace()
        job_name = _resource_name(job_id)
        try:
            await _k8s_request(
                "DELETE",
                f"/apis/batch/v1/namespaces/{namespace}/jobs/{job_name}",
                body={"propagationPolicy": "Background"},
            )
        except (RuntimeError, httpx.HTTPError):
            pass


kubernetes_agent_executor = KubernetesAgentExecutor()
