import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const healthStatus = require("../frontend/health-status.js");
const vendorManifest = JSON.parse(
  fs.readFileSync(new URL("../scripts/frontend-vendor-assets.json", import.meta.url), "utf8"),
);

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

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
  const blockedPatterns = vendorManifest.blockedRuntimeAssetPatterns.map(
    (pattern) => new RegExp(escapeRegExp(pattern), "i"),
  );

  for (const entrypoint of vendorManifest.htmlEntrypoints) {
    const page = fs.readFileSync(
      new URL(`../${entrypoint}`, import.meta.url),
      "utf8",
    );
    for (const pattern of blockedPatterns) {
      assert.equal(pattern.test(page), false, `${entrypoint} should not reference ${pattern}`);
    }
  }

  for (const asset of vendorManifest.assets) {
    assert.equal(
      fs.existsSync(new URL(`../${asset.destination}`, import.meta.url)),
      true,
      `${asset.destination} should exist`,
    );

    for (const entrypoint of asset.usedBy) {
      const page = fs.readFileSync(new URL(`../${entrypoint}`, import.meta.url), "utf8");
      assert.match(
        page,
        new RegExp(escapeRegExp(asset.runtimePath)),
        `${entrypoint} should load ${asset.runtimePath}`,
      );
    }
  }
});

test("frontend vendor manifest matches package pins", () => {
  const packageJson = JSON.parse(
    fs.readFileSync(new URL("../package.json", import.meta.url), "utf8"),
  );
  const packageLock = JSON.parse(
    fs.readFileSync(new URL("../package-lock.json", import.meta.url), "utf8"),
  );

  for (const asset of vendorManifest.assets) {
    assert.equal(packageJson.devDependencies[asset.packageName], asset.version);
    assert.equal(packageLock.packages[`node_modules/${asset.packageName}`].version, asset.version);
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
