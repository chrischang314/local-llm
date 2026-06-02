import asyncio
import os
import pathlib
import sys
import tempfile
import unittest
import uuid

import bcrypt
from fastapi.testclient import TestClient
from starlette.responses import Response


_TEST_PATH = pathlib.Path(tempfile.mkdtemp(prefix="local-llm-auth-test-"))
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(_TEST_PATH / 'chat.db').as_posix()}"
os.environ["SHARED_AUTH_DB"] = str(_TEST_PATH / "auth.db")
os.environ.setdefault("SECRET_STORE_KEY", "test-suite-secret")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

import auth  # noqa: E402
import main  # noqa: E402
import secret_store  # noqa: E402
from models import User  # noqa: E402


def unique_username(prefix: str = "user") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


class SharedAuthStoreTests(unittest.TestCase):
    def test_shared_user_session_lifecycle(self):
        username = unique_username("shared")

        registered = auth.register_user(username, "correct-horse")
        self.assertEqual(registered["username"], username)

        logged_in = auth.authenticate_user(username.upper(), "correct-horse")
        self.assertEqual(logged_in["id"], registered["id"])

        token = auth.create_session(registered["id"])
        self.assertEqual(
            auth.get_user_by_session_token(token)["username"],
            username,
        )

        auth.revoke_session(token)
        self.assertIsNone(auth.get_user_by_session_token(token))

    def test_session_cookie_domain_is_configurable(self):
        original = os.environ.get("AUTH_COOKIE_DOMAIN")
        try:
            os.environ["AUTH_COOKIE_DOMAIN"] = ".projects.lan"
            response = Response()
            auth.set_session_cookie(response, "test-token")
        finally:
            if original is None:
                os.environ.pop("AUTH_COOKIE_DOMAIN", None)
            else:
                os.environ["AUTH_COOKIE_DOMAIN"] = original

        cookie = response.headers["set-cookie"]
        self.assertIn("Domain=.projects.lan", cookie)
        self.assertIn("Path=/", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=lax", cookie)

    def test_register_reuses_legacy_username_only_shared_user(self):
        username = unique_username("legacy-shared")
        auth.init_shared_auth()
        now = auth._now()
        import sqlite3

        with sqlite3.connect(auth.auth_db_path()) as conn:
            conn.execute(
                """
                INSERT INTO users (username, username_key, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, username.casefold(), now, now),
            )

        registered = auth.register_user(username, "correct-horse")
        self.assertEqual(registered["username"], username)
        self.assertEqual(
            auth.authenticate_user(username, "correct-horse")["id"],
            registered["id"],
        )


class AuthEndpointTests(unittest.TestCase):
    def test_register_me_and_logout_use_http_only_cookie(self):
        username = unique_username("endpoint")

        with TestClient(main.app) as client:
            self.assertEqual(client.get("/auth/me").status_code, 401)

            registered = client.post(
                "/auth/register",
                json={"username": username, "password": "correct-horse"},
            )
            self.assertEqual(registered.status_code, 200, registered.text)
            self.assertEqual(registered.json()["username"], username)
            self.assertNotIn("token", registered.json())

            cookie = registered.headers["set-cookie"]
            self.assertIn("projects_lan_session=", cookie)
            self.assertIn("HttpOnly", cookie)
            self.assertIn("SameSite=lax", cookie)
            self.assertIn("Path=/", cookie)

            me = client.get("/auth/me")
            self.assertEqual(me.status_code, 200)
            self.assertEqual(me.json()["username"], username)
            self.assertEqual(client.get("/conversations").status_code, 200)

            logout = client.post("/auth/logout")
            self.assertEqual(logout.status_code, 200)
            self.assertEqual(client.get("/auth/me").status_code, 401)

    def test_login_migrates_legacy_local_password_to_shared_auth(self):
        username = unique_username("legacy-local")
        password = "abcd"

        async def seed_legacy_user():
            async with main.AsyncSessionLocal() as db:
                db.add(
                    User(
                        username=username,
                        password_hash=bcrypt.hashpw(
                            password.encode("utf-8"),
                            bcrypt.gensalt(),
                        ).decode("utf-8"),
                    )
                )
                await db.commit()

        with TestClient(main.app) as client:
            asyncio.run(seed_legacy_user())
            logged_in = client.post(
                "/auth/login",
                json={"username": username, "password": password},
            )

            self.assertEqual(logged_in.status_code, 200, logged_in.text)
            self.assertEqual(logged_in.json()["username"], username)
            self.assertNotIn("token", logged_in.json())
            self.assertEqual(client.get("/auth/me").json()["username"], username)
            with self.assertRaises((auth.InvalidCredentialsError, ValueError)):
                auth.authenticate_user(username, password)

            client.post("/auth/logout")
            logged_in_again = client.post(
                "/auth/login",
                json={"username": username, "password": password},
            )
            self.assertEqual(logged_in_again.status_code, 200, logged_in_again.text)
            self.assertEqual(logged_in_again.json()["username"], username)

    def test_empty_secret_store_key_file_is_repaired(self):
        key_file = _TEST_PATH / f"empty-secret-{uuid.uuid4().hex}"
        key_file.write_text("", encoding="utf-8")
        original_key = os.environ.pop("SECRET_STORE_KEY", None)
        original_file = os.environ.get("SECRET_STORE_KEY_FILE")
        os.environ["SECRET_STORE_KEY_FILE"] = str(key_file)
        try:
            loaded = secret_store._load_secret_store_key()
            self.assertTrue(loaded)
            self.assertEqual(key_file.read_text(encoding="utf-8").strip(), loaded)
            self.assertEqual(secret_store._load_secret_store_key(), loaded)
        finally:
            if original_key is not None:
                os.environ["SECRET_STORE_KEY"] = original_key
            else:
                os.environ.pop("SECRET_STORE_KEY", None)
            if original_file is not None:
                os.environ["SECRET_STORE_KEY_FILE"] = original_file
            else:
                os.environ.pop("SECRET_STORE_KEY_FILE", None)


if __name__ == "__main__":
    unittest.main()
