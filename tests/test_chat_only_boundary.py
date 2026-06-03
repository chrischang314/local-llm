import pathlib
import sys
import unittest

import httpx


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import main  # noqa: E402


class ChatOnlyBoundaryTests(unittest.TestCase):
    def test_old_agent_and_github_routes_are_not_registered(self):
        paths = {route.path for route in main.app.routes if hasattr(route, "path")}

        self.assertFalse(any(path.startswith("/agent") for path in paths))
        self.assertFalse(any(path.startswith("/github") for path in paths))

    def test_agent_runner_and_github_modules_are_removed(self):
        removed_paths = [
            "agent-runner",
            "backend/agent_executor.py",
            "backend/agent_routes.py",
            "backend/agent_services.py",
            "backend/github_client.py",
            "backend/github_routes.py",
            "backend/secret_store.py",
            "k8s/local-llm/agent-sandbox.yaml",
        ]

        for path in removed_paths:
            self.assertFalse((REPO_ROOT / path).exists(), path)

    def test_frontend_has_no_removed_github_or_agent_hooks(self):
        app_js = (REPO_ROOT / "frontend/app.js").read_text(encoding="utf-8")
        index_html = (REPO_ROOT / "frontend/index.html").read_text(encoding="utf-8")

        for needle in [
            "agent-job-list",
            "agent-status-pill",
            "code-job-form",
            "code-jobs-view",
            "github-connect-btn",
            "refreshGithubStatus",
            "repo-select",
            "settings-github",
            "Code Jobs",
            "/agent/jobs",
            "/github/",
            "github_oauth",
        ]:
            self.assertNotIn(needle, app_js)
            self.assertNotIn(needle, index_html)


class RemovedRouteSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_removed_code_feature_endpoints_return_404(self):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            agent_status = await client.get("/agent/status")
            github_status = await client.get("/github/status")
            agent_jobs = await client.post("/agent/jobs", json={})

        self.assertEqual(agent_status.status_code, 404)
        self.assertEqual(github_status.status_code, 404)
        self.assertEqual(agent_jobs.status_code, 404)


if __name__ == "__main__":
    unittest.main()
