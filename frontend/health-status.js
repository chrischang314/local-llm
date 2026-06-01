(function initHealthStatus(root) {
  function numberOrZero(value) {
    return Number.isFinite(value) ? value : 0;
  }

  function tone(payload) {
    if (!payload || payload.ollama !== "ok") return "down";
    const severity = payload.workers?.readiness?.severity;
    if (severity === "error") return "down";
    if (severity === "warning") return "warning";
    return "ok";
  }

  function workerSummary(workers) {
    if (!workers || !Number.isFinite(workers.enabled)) return "";
    const readiness = workers.readiness || {};
    if (readiness.summary) return readiness.summary;
    if (workers.enabled < 1) return "No workers enabled";

    const available = numberOrZero(workers.available);
    const workerWord = workers.enabled === 1 ? "worker" : "workers";
    const parts = [`${available}/${workers.enabled} ${workerWord}`];
    if (Number.isFinite(workers.busy) && workers.busy > 0) {
      parts.push(`${workers.busy} active`);
    }
    return parts.join("; ");
  }

  function status(payload) {
    if (!payload) {
      return { className: "down", label: "Backend unreachable" };
    }

    const className = tone(payload);
    const modelCount = numberOrZero(payload.model_count);
    const modelWord = modelCount === 1 ? "model" : "models";
    const summary = workerSummary(payload.workers);
    let prefix = "Ollama ready";

    if (className === "warning") {
      prefix = "Worker capacity degraded";
    } else if (className === "down") {
      prefix = payload.ollama === "ok" ? "Worker capacity offline" : "Ollama unreachable";
    }

    const parts = [`${prefix} - ${modelCount} ${modelWord}`];
    if (summary) parts.push(summary);
    return { className, label: parts.join(" - ") };
  }

  const api = { status, tone, workerSummary };
  root.LocalLlmHealthStatus = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
