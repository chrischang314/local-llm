import assert from "node:assert/strict";
import fs from "node:fs";
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

test("nginx API proxy does not capture health-status static asset", () => {
  const nginxConfig = fs.readFileSync(
    new URL("../frontend/nginx.conf", import.meta.url),
    "utf8",
  );
  const route = nginxConfig.match(/location ~ \^\/\(([^)]+)\)\(\/\|\$\)/);
  assert.ok(route, "API proxy route should require a slash or end after the prefix");

  const apiRoute = new RegExp(`^/(${route[1]})(/|$)`);
  assert.equal(apiRoute.test("/health"), true);
  assert.equal(apiRoute.test("/health-status.js"), false);
});

test("frontend pages use same-origin vendored assets", () => {
  const pageNames = ["index.html", "docs.html"];
  const blockedPatterns = [
    /https:\/\/cdn\.jsdelivr\.net/i,
    /https:\/\/unpkg\.com/i,
    /https:\/\/cdnjs\.cloudflare\.com/i,
    /lucide@latest/i,
  ];

  for (const pageName of pageNames) {
    const page = fs.readFileSync(
      new URL(`../frontend/${pageName}`, import.meta.url),
      "utf8",
    );
    for (const pattern of blockedPatterns) {
      assert.equal(pattern.test(page), false, `${pageName} should not reference ${pattern}`);
    }
  }

  const requiredAssets = [
    "../frontend/vendor/marked/4.3.0/marked.min.js",
    "../frontend/vendor/dompurify/3.4.7/purify.min.js",
    "../frontend/vendor/highlight.js/11.10.0/highlight.min.js",
    "../frontend/vendor/highlight.js/11.10.0/styles/github-dark.min.css",
    "../frontend/vendor/lucide/0.468.0/lucide.min.js",
  ];

  for (const assetPath of requiredAssets) {
    assert.equal(fs.existsSync(new URL(assetPath, import.meta.url)), true, `${assetPath} should exist`);
  }
});

test("API docs show the LAN base URL before the Compose fallback", () => {
  const docs = fs.readFileSync(new URL("../frontend/docs.html", import.meta.url), "utf8");
  const lanBaseUrlIndex = docs.indexOf("http://localllm.lan/v1");
  const composeBaseUrlIndex = docs.indexOf("http://localhost:8001/v1");

  assert.notEqual(lanBaseUrlIndex, -1);
  assert.notEqual(composeBaseUrlIndex, -1);
  assert.ok(lanBaseUrlIndex < composeBaseUrlIndex);
});
