import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

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
        self.assertEqual(
            payload["workers"],
            {
                "total": 3,
                "enabled": 2,
                "available": 1,
                "busy": 3,
                "loaded_model_count": 2,
            },
        )

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
        self.assertEqual(
            payload["workers"],
            {
                "total": 0,
                "enabled": 0,
                "available": 0,
                "busy": 0,
                "loaded_model_count": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
