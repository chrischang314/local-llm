import os
import pathlib
import sys
import unittest
from datetime import datetime, timedelta, timezone


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
os.environ.setdefault("JWT_SECRET", "test-suite-secret")

import main  # noqa: E402


class HealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_includes_worker_summary(self):
        async def fake_list_models():
            return {
                "models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:7b"}],
                "backends": [
                    {
                        "name": "mac-mini",
                        "enabled": True,
                        "available": True,
                        "in_flight": 2,
                        "loaded_models": [{"name": "llama3.2:3b"}],
                    },
                    {
                        "name": "chris-pc-1",
                        "enabled": True,
                        "available": False,
                        "in_flight": 0,
                        "loaded_models": [],
                    },
                    {
                        "name": "disabled-worker",
                        "enabled": False,
                        "available": False,
                        "in_flight": 1,
                        "loaded_models": [{"name": "old:model"}],
                    },
                ],
            }

        original = main.ollama_router.list_models
        main.ollama_router.list_models = fake_list_models
        try:
            payload = await main.health()
        finally:
            main.ollama_router.list_models = original

        self.assertEqual(payload["backend"], "ok")
        self.assertEqual(payload["ollama"], "ok")
        self.assertEqual(payload["model_count"], 2)
        self.assertEqual(payload["workers"]["total"], 3)
        self.assertEqual(payload["workers"]["enabled"], 2)
        self.assertEqual(payload["workers"]["available"], 1)
        self.assertEqual(payload["workers"]["unavailable"], 1)
        self.assertEqual(payload["workers"]["busy"], 3)
        self.assertEqual(payload["workers"]["loaded_model_count"], 2)
        self.assertEqual(payload["workers"]["readiness"]["state"], "degraded")
        self.assertEqual(payload["workers"]["readiness"]["severity"], "warning")
        self.assertEqual(
            payload["workers"]["readiness"]["summary"],
            "1/2 enabled workers available; 3 active requests; 2 resident models",
        )
        self.assertEqual(payload["workers"]["readiness"]["issue_count"], 1)
        self.assertEqual(payload["workers"]["readiness"]["issues"], [])

    async def test_health_returns_empty_worker_summary_when_router_fails(self):
        async def failing_list_models():
            raise RuntimeError("router unavailable")

        original = main.ollama_router.list_models
        main.ollama_router.list_models = failing_list_models
        try:
            payload = await main.health()
        finally:
            main.ollama_router.list_models = original

        self.assertEqual(payload["backend"], "ok")
        self.assertEqual(payload["ollama"], "down")
        self.assertEqual(payload["model_count"], 0)
        self.assertEqual(payload["workers"]["total"], 0)
        self.assertEqual(payload["workers"]["enabled"], 0)
        self.assertEqual(payload["workers"]["available"], 0)
        self.assertEqual(payload["workers"]["unavailable"], 0)
        self.assertEqual(payload["workers"]["busy"], 0)
        self.assertEqual(payload["workers"]["loaded_model_count"], 0)
        self.assertEqual(payload["workers"]["readiness"]["state"], "no_workers")
        self.assertEqual(payload["workers"]["readiness"]["issue_count"], 0)

    async def test_workers_endpoint_includes_readiness_issues(self):
        now = datetime.now(timezone.utc)

        async def fake_status():
            return {
                "backends": [
                    {
                        "name": "mac-mini",
                        "enabled": True,
                        "available": True,
                        "in_flight": 1,
                        "loaded_models": [{"name": "llama3.2:3b"}],
                    },
                    {
                        "name": "chris-pc-1",
                        "enabled": True,
                        "available": False,
                        "in_flight": 0,
                        "loaded_models": [],
                    },
                ],
            }

        async def fake_switches():
            return [
                {
                    "metadata": {
                        "name": "chris-pc-1-ollama-switch",
                        "namespace": "local-llm",
                        "labels": {"local-llm.io/worker": "chris-pc-1"},
                        "annotations": {
                            "local-llm.io/desired-state": "on",
                            "local-llm.io/actual-state": "off",
                            "local-llm.io/last-observed-at": (
                                now - timedelta(minutes=10)
                            ).isoformat(),
                        },
                    },
                    "spec": {"replicas": 1},
                    "status": {"readyReplicas": 0},
                }
            ]

        original_status = main.ollama_router.status
        original_switches = main._list_worker_switches
        original_stale_seconds = main.WORKER_SYNC_STALE_SECONDS
        main.ollama_router.status = fake_status
        main._list_worker_switches = fake_switches
        main.WORKER_SYNC_STALE_SECONDS = 300
        try:
            payload = await main.list_workers({})
        finally:
            main.ollama_router.status = original_status
            main._list_worker_switches = original_switches
            main.WORKER_SYNC_STALE_SECONDS = original_stale_seconds

        self.assertEqual(payload["readiness"]["state"], "degraded")
        self.assertEqual(payload["readiness"]["available"], 1)
        self.assertEqual(payload["readiness"]["unavailable"], 1)
        self.assertEqual(payload["readiness"]["pending_sync"], 1)
        self.assertEqual(payload["readiness"]["stale_sync"], 1)
        self.assertEqual(payload["readiness"]["issue_count"], 3)
        issue_types = {issue["type"] for issue in payload["readiness"]["issues"]}
        self.assertIn("sync_pending", issue_types)
        self.assertIn("sync_stale", issue_types)
        self.assertIn("worker_unavailable", issue_types)


if __name__ == "__main__":
    unittest.main()
