import asyncio
import hashlib
import hmac
import json
import os
import pathlib
import sys
import unittest
from datetime import timedelta
from unittest.mock import patch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
os.environ.setdefault("JWT_SECRET", "test-suite-secret")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import main  # noqa: E402
from database import Base  # noqa: E402
from models import GitHubInstallation, GitHubInstallState, GitHubOAuthServiceConfig, User, utcnow  # noqa: E402
from secret_store import encrypt_secret  # noqa: E402


class AgentJobApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.SessionLocal = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        asyncio.run(self._create_schema())

        async def override_get_db():
            async with self.SessionLocal() as session:
                yield session

        async def override_current_user_id():
            return 42

        async def override_current_user():
            return {"id": 42, "username": "tester"}

        main.app.dependency_overrides[main.get_db] = override_get_db
        main.app.dependency_overrides[main.current_user_id] = override_current_user_id
        main.app.dependency_overrides[main.current_user] = override_current_user
        self.client = TestClient(main.app)

    def tearDown(self):
        main.app.dependency_overrides.clear()
        asyncio.run(self.engine.dispose())

    async def _create_schema(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with self.SessionLocal() as db:
            db.add(User(id=42, username="tester", password_hash="hash"))
            await db.commit()

    def _create_job(self) -> str:
        with patch.dict(os.environ, {"AGENT_JOBS_ENABLED": "true"}, clear=False):
            response = self.client.post(
                "/agent/jobs",
                json={
                    "title": "Implement API",
                    "prompt": "Add the backend API",
                    "repository_owner": "octo",
                    "repository_name": "repo",
                    "base_branch": "main",
                },
            )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def test_routes_are_disabled_without_feature_flag(self):
        with patch.dict(os.environ, {"AGENT_JOBS_ENABLED": "0"}, clear=False):
            jobs_response = self.client.get("/agent/jobs")
            github_response = self.client.get("/github/status")

        self.assertEqual(jobs_response.status_code, 404)
        self.assertEqual(github_response.status_code, 200)
        self.assertFalse(github_response.json()["configured"])

    def test_github_oauth_config_and_start_are_service_level(self):
        with patch.dict(os.environ, {"AGENT_JOBS_ENABLED": "0"}, clear=False):
            save_response = self.client.post(
                "/github/oauth/config",
                json={"client_id": "client-123", "client_secret": "secret-123"},
            )
            status_response = self.client.get("/github/status")
            start_response = self.client.post("/github/oauth/start")

        self.assertEqual(save_response.status_code, 200, save_response.text)
        self.assertEqual(status_response.status_code, 200, status_response.text)
        self.assertTrue(status_response.json()["configured"])
        self.assertTrue(status_response.json()["oauth"]["configured"])
        self.assertEqual(start_response.status_code, 200, start_response.text)
        self.assertTrue(start_response.json()["configured"])
        self.assertTrue(start_response.json()["auth_url"].startswith("https://github.com/login/oauth/authorize?"))
        self.assertIn("client_id=client-123", start_response.json()["auth_url"])
        self.assertIn("scope=repo", start_response.json()["auth_url"])

    def test_github_oauth_callback_connects_state_owner(self):
        async def seed_state():
            async with self.SessionLocal() as db:
                db.add(
                    GitHubOAuthServiceConfig(
                        id=1,
                        client_id="client-123",
                        client_secret_encrypted=encrypt_secret("secret-123"),
                        created_by_user_id=42,
                        updated_by_user_id=42,
                    )
                )
                db.add(
                    GitHubInstallState(
                        user_id=42,
                        state="state-123",
                        expires_at=utcnow() + timedelta(minutes=5),
                    )
                )
                await db.commit()

        asyncio.run(seed_state())

        async def fake_exchange_oauth_code(self, **_):
            return {"access_token": "gho_test", "scope": "repo", "token_type": "bearer"}

        async def fake_oauth_user(self, _token):
            return {"login": "octocat", "id": 1, "type": "User"}

        with patch("github_routes.GitHubAppClient.exchange_oauth_code", fake_exchange_oauth_code):
            with patch("github_routes.GitHubAppClient.oauth_user", fake_oauth_user):
                response = self.client.get(
                    "/github/oauth/callback?code=code-123&state=state-123",
                    follow_redirects=False,
                )

        self.assertEqual(response.status_code, 303)
        self.assertIn("github_oauth=connected", response.headers["location"])

        async def load_connection():
            async with self.SessionLocal() as db:
                result = await db.execute(select(GitHubInstallation))
                return result.scalars().first()

        connection = asyncio.run(load_connection())
        self.assertIsNotNone(connection)
        self.assertEqual(connection.user_id, 42)
        self.assertEqual(connection.auth_type, "oauth")
        self.assertEqual(connection.account_login, "octocat")

    def test_create_list_get_and_transition_job(self):
        with patch.dict(os.environ, {"AGENT_JOBS_ENABLED": "true"}, clear=False):
            create_response = self.client.post(
                "/agent/jobs",
                json={
                    "title": "Implement API",
                    "prompt": "Add the backend API",
                    "repository_owner": "octo",
                    "repository_name": "repo",
                    "base_branch": "main",
                    "metadata": {"source": "unit-test"},
                },
            )
            self.assertEqual(create_response.status_code, 201, create_response.text)
            created = create_response.json()

            self.assertEqual(created["status"], "queued")
            self.assertEqual(created["repository"], {"owner": "octo", "name": "repo"})
            self.assertEqual(created["metadata"], {"source": "unit-test"})
            self.assertEqual(created["events"][0]["status"], "queued")

            list_response = self.client.get("/agent/jobs")
            self.assertEqual(list_response.status_code, 200, list_response.text)
            self.assertEqual([job["id"] for job in list_response.json()], [created["id"]])

            running_response = self.client.post(
                f"/agent/jobs/{created['id']}/state",
                json={"status": "running", "message": "Started"},
            )
            self.assertEqual(running_response.status_code, 200, running_response.text)
            self.assertEqual(running_response.json()["status"], "running")
            self.assertIsNotNone(running_response.json()["started_at"])

            done_response = self.client.post(
                f"/agent/jobs/{created['id']}/state",
                json={
                    "status": "succeeded",
                    "message": "Done",
                    "result": {"branch": "codex/example"},
                },
            )
            self.assertEqual(done_response.status_code, 200, done_response.text)
            done = done_response.json()
            self.assertEqual(done["status"], "succeeded")
            self.assertEqual(done["result"], {"branch": "codex/example"})
            self.assertIsNotNone(done["completed_at"])

            get_response = self.client.get(f"/agent/jobs/{created['id']}")
            self.assertEqual(get_response.status_code, 200, get_response.text)
            self.assertGreaterEqual(len(get_response.json()["events"]), 3)

    def test_invalid_state_transition_returns_conflict(self):
        job_id = self._create_job()
        with patch.dict(os.environ, {"AGENT_JOBS_ENABLED": "true"}, clear=False):
            succeeded = self.client.post(
                f"/agent/jobs/{job_id}/state",
                json={"status": "succeeded", "message": "Done"},
            )
            self.assertEqual(succeeded.status_code, 200, succeeded.text)

            response = self.client.post(
                f"/agent/jobs/{job_id}/state",
                json={"status": "running", "message": "Too late"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("Cannot transition", response.json()["detail"])

    def test_create_job_can_dispatch_to_github_app(self):
        calls = []

        class FakeGitHubClient:
            config = type(
                "Config",
                (),
                {"default_owner": None, "default_repo": None},
            )()

            async def repository_dispatch(self, *, owner, repo, event_type, client_payload):
                calls.append(
                    {
                        "owner": owner,
                        "repo": repo,
                        "event_type": event_type,
                        "client_payload": client_payload,
                    }
                )

        with patch.dict(os.environ, {"AGENT_JOBS_ENABLED": "true"}, clear=False):
            with patch("agent_job_routes.GitHubAppClient", FakeGitHubClient):
                response = self.client.post(
                    "/agent/jobs",
                    json={
                        "title": "Dispatch me",
                        "prompt": "Run the job",
                        "repository_owner": "octo",
                        "repository_name": "repo",
                        "dispatch": True,
                        "dispatch_event": "local_llm.test",
                    },
                )

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["owner"], "octo")
        self.assertEqual(calls[0]["repo"], "repo")
        self.assertEqual(calls[0]["event_type"], "local_llm.test")
        self.assertEqual(calls[0]["client_payload"]["agent_job_id"], response.json()["id"])
        self.assertEqual(response.json()["events"][-1]["event_type"], "github.dispatch")

    def test_github_webhook_signature_updates_job_state(self):
        job_id = self._create_job()
        payload = {
            "action": "completed",
            "agent_job_id": job_id,
            "workflow_run": {
                "status": "completed",
                "conclusion": "success",
            },
        }
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        signature = "sha256=" + hmac.new(
            b"webhook-secret",
            body,
            hashlib.sha256,
        ).hexdigest()

        with patch.dict(
            os.environ,
            {
                "AGENT_JOBS_ENABLED": "true",
                "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            },
            clear=False,
        ):
            response = self.client.post(
                "/github/webhook",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "workflow_run",
                    "X-GitHub-Delivery": "delivery-1",
                    "X-Hub-Signature-256": signature,
                },
            )
            get_response = self.client.get(f"/agent/jobs/{job_id}")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["job_updated"])
        self.assertEqual(get_response.status_code, 200, get_response.text)
        self.assertEqual(get_response.json()["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
