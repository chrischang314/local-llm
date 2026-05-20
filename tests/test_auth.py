import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
os.environ.setdefault("JWT_SECRET", "test-suite-secret")

import auth  # noqa: E402


class AuthSecretTests(unittest.TestCase):
    def test_jwt_secret_file_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret_file = pathlib.Path(tmp) / "jwt_secret"

            with patch.dict(os.environ, {"JWT_SECRET_FILE": str(secret_file)}, clear=False):
                os.environ.pop("JWT_SECRET", None)
                first = auth._load_jwt_secret()
                second = auth._load_jwt_secret()

            self.assertEqual(first, second)
            self.assertTrue(len(first) > 40)
            self.assertEqual(secret_file.read_text(encoding="utf-8").strip(), first)

    def test_jwt_secret_env_takes_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret_file = pathlib.Path(tmp) / "jwt_secret"
            with patch.dict(
                os.environ,
                {"JWT_SECRET": "configured-secret", "JWT_SECRET_FILE": str(secret_file)},
                clear=False,
            ):
                self.assertEqual(auth._load_jwt_secret(), "configured-secret")

            self.assertFalse(secret_file.exists())


if __name__ == "__main__":
    unittest.main()
