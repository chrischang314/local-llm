import asyncio
import json
import os
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(frozen=True)
class OllamaBackend:
    name: str
    url: str
    priority: int = 100
    weight: int = 1
    models: tuple[str, ...] = ()
    enabled: bool = True
    essential: bool = False
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OllamaRouterConfig:
    backends: tuple[OllamaBackend, ...]
    model_preferences: dict[str, tuple[str, ...]] = field(default_factory=dict)


class OllamaRouter:
    def __init__(self):
        self._health_ttl = float(os.getenv("OLLAMA_HEALTH_TTL_SECONDS", "15"))
        self._health_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._in_flight: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def load_config(self) -> OllamaRouterConfig:
        raw = os.getenv("OLLAMA_BACKENDS")
        config_path = os.getenv("OLLAMA_BACKENDS_FILE")

        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as config_file:
                raw = config_file.read()

        if raw:
            parsed = json.loads(raw)
        else:
            parsed = {
                "backends": [
                    {
                        "name": os.getenv("OLLAMA_BACKEND_NAME", "default"),
                        "url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
                        "priority": 100,
                        "weight": 1,
                        "enabled": True,
                        "essential": True,
                    }
                ]
            }

        if isinstance(parsed, list):
            parsed = {"backends": parsed}

        backends = tuple(self._parse_backend(item) for item in parsed.get("backends", []))
        preferences = {
            model: tuple(names)
            for model, names in parsed.get("model_preferences", {}).items()
        }
        return OllamaRouterConfig(backends=backends, model_preferences=preferences)

    def _parse_backend(self, item: dict[str, Any]) -> OllamaBackend:
        return OllamaBackend(
            name=item["name"],
            url=item["url"].rstrip("/"),
            priority=int(item.get("priority", 100)),
            weight=max(int(item.get("weight", 1)), 1),
            models=tuple(item.get("models", [])),
            enabled=bool(item.get("enabled", True)),
            essential=bool(item.get("essential", False)),
            labels={str(k): str(v) for k, v in item.get("labels", {}).items()},
        )

    async def list_models(self) -> dict[str, Any]:
        config = self.load_config()
        health = await self._collect_health(config.backends)
        models_by_name: dict[str, dict[str, Any]] = {}

        for backend in config.backends:
            state = health[backend.name]
            for model in state["models"]:
                name = model.get("name") or model.get("model")
                if not name:
                    continue
                current = models_by_name.get(name)
                if current is None or model.get("size", 0) < current.get("size", float("inf")):
                    models_by_name[name] = dict(model)

        return {
            "models": sorted(models_by_name.values(), key=lambda model: (model.get("size", 0), model.get("name", ""))),
            "backends": self._public_backend_status(config.backends, health),
        }

    async def status(self) -> dict[str, Any]:
        config = self.load_config()
        health = await self._collect_health(config.backends, force=True)
        return {
            "backends": self._public_backend_status(config.backends, health),
            "model_preferences": {
                model: list(names)
                for model, names in config.model_preferences.items()
            },
        }

    async def choose_backend(self, model: str) -> OllamaBackend:
        config = self.load_config()
        if not config.backends:
            raise RuntimeError("No Ollama backends are configured")

        health = await self._collect_health(config.backends)
        candidates = [
            backend for backend in config.backends
            if backend.enabled
            and health[backend.name]["available"]
            and self._backend_can_run_model(backend, health[backend.name], model)
        ]

        if not candidates:
            available = [
                backend.name for backend in config.backends
                if backend.enabled and health[backend.name]["available"]
            ]
            raise RuntimeError(
                f"No available Ollama backend can run '{model}'. Available backends: {', '.join(available) or 'none'}"
            )

        preference = config.model_preferences.get(model, ())
        preference_index = {name: idx for idx, name in enumerate(preference)}

        async with self._lock:
            scored = [
                (
                    preference_index.get(backend.name, 10_000),
                    backend.priority,
                    self._in_flight.get(backend.name, 0) / backend.weight,
                    random.random(),
                    backend,
                )
                for backend in candidates
            ]
            scored.sort(key=lambda item: item[:4])
            return scored[0][4]

    @asynccontextmanager
    async def track_request(self, backend: OllamaBackend):
        async with self._lock:
            self._in_flight[backend.name] = self._in_flight.get(backend.name, 0) + 1
        try:
            yield
        finally:
            async with self._lock:
                self._in_flight[backend.name] = max(self._in_flight.get(backend.name, 1) - 1, 0)

    async def _collect_health(
        self,
        backends: tuple[OllamaBackend, ...],
        force: bool = False,
    ) -> dict[str, dict[str, Any]]:
        results = await asyncio.gather(
            *(self._backend_health(backend, force=force) for backend in backends)
        )
        return {backend.name: result for backend, result in zip(backends, results)}

    async def _backend_health(self, backend: OllamaBackend, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        cached = self._health_cache.get(backend.name)
        if not force and cached and now - cached[0] < self._health_ttl:
            return cached[1]

        state = {
            "available": False,
            "models": [],
            "model_names": set(),
            "error": None,
        }

        if not backend.enabled:
            state["error"] = "disabled"
            return state

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{backend.url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                state["available"] = True
                state["models"] = models
                state["model_names"] = {
                    model.get("name") or model.get("model")
                    for model in models
                    if model.get("name") or model.get("model")
                }
        except httpx.HTTPError as exc:
            state["error"] = str(exc)

        self._health_cache[backend.name] = (now, state)
        return state

    def _backend_can_run_model(self, backend: OllamaBackend, state: dict[str, Any], model: str) -> bool:
        configured_models = set(backend.models)
        actual_models = set(state["model_names"])

        if configured_models and model not in configured_models:
            return False
        if actual_models and model not in actual_models:
            return False
        return True

    def _public_backend_status(
        self,
        backends: tuple[OllamaBackend, ...],
        health: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": backend.name,
                "url": backend.url,
                "available": health[backend.name]["available"],
                "enabled": backend.enabled,
                "essential": backend.essential,
                "priority": backend.priority,
                "weight": backend.weight,
                "configured_models": list(backend.models),
                "available_models": sorted(health[backend.name]["model_names"]),
                "in_flight": self._in_flight.get(backend.name, 0),
                "labels": backend.labels,
                "error": health[backend.name]["error"],
            }
            for backend in backends
        ]


ollama_router = OllamaRouter()
