import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const healthStatus = require("../frontend/health-status.js");

test("compact health label stays ready when Ollama and workers are ready", () => {
  const payload = {
    ollama: "ok",
    model_count: 4,
    workers: {
      enabled: 2,
      available: 2,
      busy: 0,
      readiness: {
        severity: "ok",
        summary: "2/2 enabled workers available",
      },
    },
  };

  assert.equal(healthStatus.tone(payload), "ok");
  assert.deepEqual(healthStatus.status(payload), {
    className: "ok",
    label: "Ollama ready - 4 models - 2/2 enabled workers available",
  });
});

test("compact health label warns when workers are degraded but Ollama is ready", () => {
  const payload = {
    ollama: "ok",
    model_count: 4,
    workers: {
      enabled: 3,
      available: 2,
      busy: 1,
      readiness: {
        severity: "warning",
        summary: "2/3 enabled workers available; 1 active requests",
        issue_count: 1,
      },
    },
  };

  assert.equal(healthStatus.tone(payload), "warning");
  assert.deepEqual(healthStatus.status(payload), {
    className: "warning",
    label: "Worker capacity degraded - 4 models - 2/3 enabled workers available; 1 active requests",
  });
});

test("compact health label stays down when Ollama is unreachable", () => {
  const payload = {
    ollama: "down",
    model_count: 0,
    workers: {
      enabled: 2,
      available: 1,
      readiness: {
        severity: "warning",
        summary: "1/2 enabled workers available",
      },
    },
  };

  assert.equal(healthStatus.tone(payload), "down");
  assert.deepEqual(healthStatus.status(payload), {
    className: "down",
    label: "Ollama unreachable - 0 models - 1/2 enabled workers available",
  });
});
