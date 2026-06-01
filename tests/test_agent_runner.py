import asyncio
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import contextmanager

from fastapi import HTTPException


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-runner"))

from agent_runner import main as runner  # noqa: E402


@contextmanager
def patched_env(**values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class AgentRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_allowed_python_command_inside_workspace(self):
        python_cmd = pathlib.Path(sys.executable).name
        command_path = os.pathsep.join([str(pathlib.Path(sys.executable).parent), os.environ.get("PATH", "")])
        with tempfile.TemporaryDirectory() as workspace, patched_env(
            RUNNER_WORKSPACE=workspace,
            RUNNER_ALLOWED_COMMANDS=python_cmd,
            RUNNER_COMMAND_PATH=command_path,
            RUNNER_MAX_TIMEOUT_SECONDS="5",
        ):
            result = await runner.run_sandbox_command(
                runner.RunRequest(
                    argv=[python_cmd, "-c", "print('runner-ok')"],
                    timeout_seconds=5,
                )
            )

        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["stdout"].strip(), "runner-ok")

    async def test_times_out_long_running_command(self):
        python_cmd = pathlib.Path(sys.executable).name
        command_path = os.pathsep.join([str(pathlib.Path(sys.executable).parent), os.environ.get("PATH", "")])
        with tempfile.TemporaryDirectory() as workspace, patched_env(
            RUNNER_WORKSPACE=workspace,
            RUNNER_ALLOWED_COMMANDS=python_cmd,
            RUNNER_COMMAND_PATH=command_path,
            RUNNER_MAX_TIMEOUT_SECONDS="5",
        ):
            result = await runner.run_sandbox_command(
                runner.RunRequest(
                    argv=[python_cmd, "-c", "import time; time.sleep(3)"],
                    timeout_seconds=0.2,
                )
            )

        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["exit_code"], 0)

    def test_rejects_command_outside_allowlist(self):
        with tempfile.TemporaryDirectory() as workspace, patched_env(
            RUNNER_WORKSPACE=workspace,
            RUNNER_ALLOWED_COMMANDS="python",
        ):
            with self.assertRaises(HTTPException) as ctx:
                runner.prepare_run(runner.RunRequest(argv=["powershell", "-NoProfile"]))

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("not allowed", ctx.exception.detail)

    def test_rejects_workspace_escape(self):
        with tempfile.TemporaryDirectory() as workspace, patched_env(
            RUNNER_WORKSPACE=workspace,
            RUNNER_ALLOWED_COMMANDS="python",
        ):
            with self.assertRaises(HTTPException) as ctx:
                runner.prepare_run(runner.RunRequest(argv=["python", "-c", "print(1)"], cwd=".."))

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("workspace", ctx.exception.detail)

    def test_extracts_bearer_or_explicit_runner_token(self):
        self.assertEqual(
            runner.supplied_token({"Authorization": "Bearer secret-token"}),
            "secret-token",
        )
        self.assertEqual(
            runner.supplied_token({"X-Runner-Token": "explicit-token"}),
            "explicit-token",
        )
        self.assertIsNone(runner.supplied_token({"Authorization": "Basic nope"}))


if __name__ == "__main__":
    unittest.main()
