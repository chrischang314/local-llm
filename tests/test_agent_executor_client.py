import json
import os
import pathlib
import sys
import unittest

import httpx


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
os.environ.setdefault("JWT_SECRET", "test-suite-secret")

from agent_executor_client import (  # noqa: E402
    AgentExecutorClient,
    AgentExecutorNotConfigured,
    AgentExecutorRequestError,
)


class AgentExecutorClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_runner_request_with_bearer_token(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "http://runner.local/runs")
            self.assertEqual(request.headers["authorization"], "Bearer test-token")
            payload = json.loads(request.content.decode("utf-8"))
            self.assertEqual(payload["argv"], ["python", "-c", "print('ok')"])
            self.assertEqual(payload["timeout_seconds"], 5)
            self.assertEqual(payload["env"], {"PYTHONUNBUFFERED": "1"})
            return httpx.Response(
                200,
                json={
                    "id": "run-1",
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                    "timed_out": False,
                },
            )

        client = AgentExecutorClient(
            base_url="http://runner.local",
            token="test-token",
            transport=httpx.MockTransport(handler),
        )
        result = await client.run(
            ["python", "-c", "print('ok')"],
            timeout_seconds=5,
            env={"PYTHONUNBUFFERED": "1"},
        )

        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"], "ok\n")

    async def test_raises_when_runner_is_not_configured(self):
        client = AgentExecutorClient(base_url="")
        with self.assertRaises(AgentExecutorNotConfigured):
            await client.run(["python", "-c", "print('ok')"])

    async def test_wraps_runner_http_errors(self):
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"detail": "command is not allowed"})

        client = AgentExecutorClient(
            base_url="http://runner.local",
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaises(AgentExecutorRequestError) as ctx:
            await client.run(["sh", "-c", "echo nope"])

        self.assertIn("400", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
