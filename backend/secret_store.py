"""Local encrypted storage helpers for user-supplied integration secrets."""

import base64
import hashlib
import os
import pathlib
import secrets

from cryptography.fernet import Fernet, InvalidToken


def _secret_file_path() -> pathlib.Path:
    return pathlib.Path(
        os.getenv(
            "SECRET_STORE_KEY_FILE",
            str(pathlib.Path(os.getenv("LOCAL_LLM_DATA_DIR", "data")) / "secret_store_key"),
        )
    )


def _legacy_jwt_secret_file_path() -> pathlib.Path:
    return pathlib.Path(
        os.getenv(
            "JWT_SECRET_FILE",
            str(pathlib.Path(os.getenv("LOCAL_LLM_DATA_DIR", "data")) / "jwt_secret"),
        )
    )


def _persist_secret(secret_path: pathlib.Path, value: str) -> None:
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if secret_path.exists():
        try:
            if secret_path.read_text(encoding="utf-8").strip():
                return
        except OSError:
            return
    with secret_path.open("w", encoding="utf-8") as handle:
        handle.write(value + "\n")
    try:
        secret_path.chmod(0o600)
    except OSError:
        pass


def _load_secret_store_key() -> str:
    configured = os.getenv("SECRET_STORE_KEY")
    if configured:
        return configured

    secret_path = _secret_file_path()
    try:
        persisted = secret_path.read_text(encoding="utf-8").strip()
        if persisted:
            return persisted
    except FileNotFoundError:
        pass

    # Legacy compatibility only: older releases derived integration encryption
    # from JWT_SECRET. Login sessions no longer use JWTs.
    legacy_env = os.getenv("JWT_SECRET")
    if legacy_env:
        _persist_secret(secret_path, legacy_env)
        return legacy_env

    legacy_path = _legacy_jwt_secret_file_path()
    try:
        legacy_secret = legacy_path.read_text(encoding="utf-8").strip()
        if legacy_secret:
            _persist_secret(secret_path, legacy_secret)
            return legacy_secret
    except FileNotFoundError:
        pass

    generated = secrets.token_urlsafe(48)
    _persist_secret(secret_path, generated)
    return generated


def _fernet() -> Fernet:
    secret = _load_secret_store_key()
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    return Fernet(key)


def encrypt_secret(value: str) -> str:
    if not value:
        return ""
    return _fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_secret(value: str | None) -> str:
    if not value:
        return ""
    try:
        return _fernet().decrypt(value.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Stored secret could not be decrypted; reconnect the integration") from exc
