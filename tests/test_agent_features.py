import importlib.util
import hmac
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
os.environ.setdefault("JWT_SECRET", "test-suite-secret")

import agent_executor  # noqa: E402
import agent_routes  # noqa: E402
import agent_services  # noqa: E402
import github_client  # noqa: E402
from models import AgentJob  # noqa: E402


class AgentServiceTests(unittest.TestCase):
    def test_agent_jobs_enabled_is_feature_gated(self):
        with patch.dict(os.environ, {"AGENT_JOBS_ENABLED": "false"}, clear=False):
            self.assertFalse(agent_services.agent_jobs_enabled())
        with patch.dict(os.environ, {"AGENT_JOBS_ENABLED": "true"}, clear=False):
            self.assertTrue(agent_services.agent_jobs_enabled())

    def test_logs_are_redacted_and_diff_is_clamped(self):
        redacted = agent_services.redact_log(
            "token ghp_abcdefghijklmnopqrstuvwxyz123456 and github_pat_abcdefghijklmnopqrstuvwxyz123456"
        )
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz123456", redacted)
        self.assertNotIn("github_pat_abcdefghijklmnopqrstuvwxyz123456", redacted)
        self.assertIn("[REDACTED]", redacted)

        with patch.object(agent_services, "MAX_DIFF_CHARS", 12):
            self.assertEqual(agent_services.clamp_diff("short"), "short")
            self.assertTrue(agent_services.clamp_diff("x" * 20).endswith("[diff truncated]\n"))

    def test_callback_token_is_bound_to_job_id(self):
        with patch.dict(os.environ, {"AGENT_SECRET_KEY": "runner-secret"}, clear=False):
            first = agent_services.callback_token("job-a")
            second = agent_services.callback_token("job-b")
            repeat = agent_services.callback_token("job-a")
        self.assertNotEqual(first, second)
        self.assertTrue(hmac.compare_digest(first, repeat))


class KubernetesExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_launch_uses_restricted_job_manifest_and_secret_refs(self):
        calls = []

        async def fake_k8s_request(method, path, *, body=None, content_type="application/json"):
            calls.append({"method": method, "path": path, "body": body, "content_type": content_type})
            if path.endswith("/jobs") and method == "POST":
                return {"metadata": {"uid": "job-uid-123"}}
            return {}

        job = AgentJob(
            id="abcdef1234567890",
            repo_full_name="owner/repo",
            base_branch="main",
            work_branch="agent/abcdef123456",
            model="llama3.2:3b",
            task="make a small change",
            test_command="python -m unittest discover -s tests",
        )

        original = agent_executor._k8s_request
        agent_executor._k8s_request = fake_k8s_request
        try:
            with patch.dict(os.environ, {"AGENT_SECRET_KEY": "runner-secret"}, clear=False):
                result = await agent_executor.kubernetes_agent_executor.launch(job, "ghs_secret_token")
        finally:
            agent_executor._k8s_request = original

        self.assertEqual(result.namespace, "local-llm-sandbox")
        self.assertEqual(len(calls), 3)
        secret_body = calls[0]["body"]
        self.assertEqual(secret_body["kind"], "Secret")
        self.assertIn("github-token", secret_body["data"])
        self.assertIn("agent-callback-token", secret_body["data"])
        self.assertNotIn("ghs_secret_token", str(calls[1]["body"]))

        job_body = calls[1]["body"]
        pod_spec = job_body["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]
        init_container = pod_spec["initContainers"][0]
        self.assertFalse(pod_spec["automountServiceAccountToken"])
        self.assertFalse(container["securityContext"]["allowPrivilegeEscalation"])
        self.assertTrue(container["securityContext"]["readOnlyRootFilesystem"])
        self.assertEqual(container["securityContext"]["capabilities"], {"drop": ["ALL"]})
        self.assertFalse(init_container["securityContext"]["allowPrivilegeEscalation"])
        self.assertEqual(init_container["securityContext"]["capabilities"], {"drop": ["ALL"]})
        runner_env = {item["name"]: item for item in container["env"]}
        self.assertIn("GITHUB_TOKEN_FILE", runner_env)
        self.assertNotIn("GITHUB_TOKEN", runner_env)
        self.assertNotIn("AGENT_CALLBACK_TOKEN", runner_env)
        self.assertIn("agent-secrets", [volume["name"] for volume in pod_spec["volumes"]])
        self.assertEqual(job_body["spec"]["backoffLimit"], 0)
        self.assertEqual(job_body["spec"]["activeDeadlineSeconds"], 1800)
        self.assertFalse(any("hostPath" in volume for volume in pod_spec["volumes"]))
        self.assertIn("ownerReferences", calls[2]["body"]["metadata"])


def load_runner_module():
    runner_path = REPO_ROOT / "agent-runner" / "runner.py"
    spec = importlib.util.spec_from_file_location("agent_runner_test_module", runner_path)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    return runner


class AgentWorkflowTests(unittest.TestCase):
    def test_job_steps_show_multi_agent_quality_loop(self):
        steps = agent_routes._initial_steps("job-123")
        self.assertEqual(
            [step.name for step in steps],
            ["clone", "implement", "review", "revise", "test", "push"],
        )

    def test_runner_parses_reviewer_and_tester_decisions(self):
        runner = load_runner_module()

        approved = runner.parse_decision(
            '{"status":"approved","summary":"looks good","issues":[]}'
        )
        self.assertTrue(approved.approved)
        self.assertEqual(approved.summary, "looks good")

        requested = runner.parse_decision(
            '{"status":"changes_requested","summary":"needs tests","issues":["missing test"]}'
        )
        self.assertFalse(requested.approved)
        self.assertEqual(requested.issues, ["missing test"])

        passing = runner.TestResult(supplied=True, passed=True, exit_code=0, output="ok")
        failing = runner.TestResult(supplied=True, passed=False, exit_code=1, output="failed")
        missing = runner.TestResult(supplied=False, passed=False, exit_code=None, output="")
        self.assertTrue(runner.tester_agent(passing).approved)
        self.assertFalse(runner.tester_agent(failing).approved)
        self.assertFalse(runner.tester_agent(missing).approved)

    def test_runner_quality_loop_revises_until_review_and_tests_pass(self):
        runner = load_runner_module()
        steps = []
        review_calls = []
        revision_calls = []

        def fake_step(name, status, exit_code=None):
            steps.append((name, status, exit_code))

        def fake_reviewer(cycle):
            review_calls.append(cycle)
            if cycle == 1:
                return runner.AgentDecision(False, "needs cleanup", ["simplify the change"])
            return runner.AgentDecision(True, "ready for tests", [])

        def fake_tool_loop(role, assignment, *, max_iterations):
            revision_calls.append((role, assignment, max_iterations))
            return "revised"

        runner.step = fake_step
        runner.log = lambda *args, **kwargs: None
        runner.reviewer_agent = fake_reviewer
        runner.run_tool_loop = fake_tool_loop
        runner.run_tests = lambda: runner.TestResult(supplied=True, passed=True, exit_code=0, output="ok")

        result = runner.run_quality_loop()

        self.assertTrue(result.satisfactory)
        self.assertTrue(result.revised)
        self.assertEqual(review_calls, [1, 2])
        self.assertEqual(revision_calls[0][0], "revision")
        self.assertIn(("test", "succeeded", 0), steps)


class GitHubBypassTests(unittest.IsolatedAsyncioTestCase):
    async def test_bypass_token_satisfies_agent_config_without_app_credentials(self):
        with patch.dict(
            os.environ,
            {
                "AGENT_JOBS_ENABLED": "true",
                "AGENT_SECRET_KEY": "runner-secret",
                "GITHUB_ALLOWED_INSTALLATION_IDS": "bypass",
                "GITHUB_BYPASS_TOKEN": "test-bypass-token-value",
            },
            clear=False,
        ):
            for key in ("GITHUB_APP_ID", "GITHUB_APP_SLUG", "GITHUB_APP_PRIVATE_KEY", "GITHUB_APP_PRIVATE_KEY_FILE"):
                os.environ.pop(key, None)
            self.assertEqual(agent_routes._missing_agent_config(), [])
            token = await github_client.github_app_client.create_installation_token("bypass")
            self.assertEqual(token["token"], "test-bypass-token-value")

    async def test_bypass_repo_listing_uses_user_repos_api(self):
        seen = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return [
                    {
                        "id": 1,
                        "name": "demo",
                        "full_name": "owner/demo",
                        "private": True,
                        "default_branch": "main",
                    }
                ]

        async def fake_get(url, **kwargs):
            seen["url"] = url
            seen["headers"] = kwargs.get("headers", {})
            return FakeResponse()

        with patch.dict(os.environ, {"GITHUB_BYPASS_TOKEN": "test-bypass-token-value"}, clear=False):
            with patch("github_client.httpx.AsyncClient") as client_class:
                client = AsyncMock()
                client.__aenter__.return_value.get.side_effect = fake_get
                client_class.return_value = client

                repos = await github_client.github_app_client.repositories("bypass")

        self.assertTrue(seen["url"].endswith("/user/repos"))
        self.assertIn("Bearer test-bypass-token-value", seen["headers"]["Authorization"])
        self.assertEqual(repos["repositories"][0]["full_name"], "owner/demo")


class AgentRunnerPathTests(unittest.TestCase):
    def test_runner_refuses_paths_outside_workspace_and_git_dir_writes(self):
        runner = load_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            runner.REPO_DIR = pathlib.Path(tmp)
            self.assertEqual(runner.safe_path("src/app.py"), pathlib.Path(tmp) / "src" / "app.py")
            with self.assertRaises(ValueError):
                runner.safe_path("../outside.txt")
            with self.assertRaises(ValueError):
                runner.safe_path(".git/config")


if __name__ == "__main__":
    unittest.main()
