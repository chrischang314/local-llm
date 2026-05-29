import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const healthStatus = require("../frontend/health-status.js");
const frontendRoot = new URL("../frontend/", import.meta.url);

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

test("runtime HTML uses same-origin pinned frontend assets", () => {
  const pages = ["index.html", "docs.html"];
  const requiredAssets = [
    "vendor/lucide-1.17.0.min.js",
    "vendor/marked-4.3.0.min.js",
    "vendor/dompurify-3.4.7.min.js",
    "vendor/highlight-github-dark-11.10.0.min.css",
    "vendor/highlight-11.10.0.min.js",
  ];

  for (const page of pages) {
    const html = fs.readFileSync(new URL(page, frontendRoot), "utf8");
    assert.doesNotMatch(html, /https?:\/\/(?:cdn\.jsdelivr\.net|unpkg\.com)/);
    assert.doesNotMatch(html, /@latest/);
    assert.match(html, /vendor\/lucide-1\.17\.0\.min\.js/);
  }

  const indexHtml = fs.readFileSync(new URL("index.html", frontendRoot), "utf8");
  for (const asset of requiredAssets) {
    assert.equal(
      fs.existsSync(new URL(asset, frontendRoot)),
      true,
      `${asset} should be vendored with the frontend`,
    );
    assert.match(indexHtml, new RegExp(asset.replaceAll(".", "\\.")));
  }
});

test("docs present the LAN base URL before the Compose development URL", () => {
  const docsHtml = fs.readFileSync(new URL("docs.html", frontendRoot), "utf8");
  const lanIndex = docsHtml.indexOf("http://localllm.lan/v1");
  const devIndex = docsHtml.indexOf("http://localhost:8001/v1");

  assert.notEqual(lanIndex, -1);
  assert.notEqual(devIndex, -1);
  assert.equal(lanIndex < devIndex, true);
});
