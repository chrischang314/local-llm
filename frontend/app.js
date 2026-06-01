/* eslint-env browser */
/*
 * Local LLM Chat — frontend
 *
 * Talks to the FastAPI backend. JWT bearer auth, attached to every
 * authenticated request. Conversation settings (model, system prompt,
 * sampling params) live on the conversation row; the settings modal edits
 * them with PATCH /conversations/{id}.
 *
 * Streaming chat uses fetch + a ReadableStream reader. Generation is
 * cancellable via an AbortController bound to the visible "Stop" button.
 */

/* ---------- Configuration ---------- */

// API base resolution: same-origin by default (works behind nginx). For
// dev-mode where the frontend is served separately, set
// <meta name="api-base" content="http://localhost:8000"> in index.html
// or `localStorage.setItem('apiBase', '...')` in DevTools.
const API =
  localStorage.getItem("apiBase") ||
  document.querySelector('meta[name="api-base"]')?.content ||
  "";

// Default model context window if Ollama doesn't report one. Used purely
// for the "X / Y tokens" indicator; doesn't affect what's sent to Ollama.
const DEFAULT_CONTEXT_WINDOW = 8192;

marked.setOptions({ breaks: true, gfm: true });

/* ---------- App state ---------- */

let authToken = null;
let currentUser = null;
let currentConversation = null; // full row from backend
let currentConversationId = null;
let messages = []; // [{id?, role, content}]
let conversations = [];
let isStreaming = false;
let streamAbortController = null;
let loginMode = "login"; // or "register"
let currentWorkspace = "chat";
let codeWorkspaceLoaded = false;
let githubRepositories = [];
let codeJobs = [];
let selectedCodeJobId = null;
let githubStatus = null;
let githubOauthNotice = null;
let webResearchEnabled = false;
let webResearchAvailable = true;

/* ---------- DOM refs ---------- */

const $ = (id) => document.getElementById(id);
const loginScreen = $("login-screen");
const appEl = $("app");
const loginForm = $("login-form");
const usernameInput = $("username-input");
const passwordInput = $("password-input");
const loginError = $("login-error");
const loginSubmit = $("login-submit");
const loginSubtitle = $("login-subtitle");
const toggleMode = $("toggle-mode");
const sidebarUsername = $("sidebar-username");
const logoutBtn = $("logout-btn");
const newChatBtn = $("new-chat-btn");
const settingsBtn = $("settings-btn");
const conversationsList = $("conversations-list");
const messagesEl = $("messages");
const inputEl = $("input");
const sendBtn = $("send-btn");
const stopBtn = $("stop-btn");
const modelSelect = $("model-select");
const chatTitle = $("chat-title");
const tokenCounter = $("token-counter");
const healthIndicator = $("health-indicator");
const researchToggle = $("research-toggle");
const chatWorkspaceBtn = $("chat-workspace-btn");
const codeWorkspaceBtn = $("code-workspace-btn");
const chatWorkspace = document.querySelector(".main");
const codeWorkspace = $("code-workspace");
const githubRefreshBtn = $("github-refresh-btn");
const jobsRefreshBtn = $("jobs-refresh-btn");
const githubStatusPill = $("github-status-pill");
const githubSummary = $("github-summary");
const githubRepoSelect = $("github-repo-select");
const githubConnectBtn = $("github-connect-btn");
const githubDisconnectBtn = $("github-disconnect-btn");
const githubOauthSetup = $("github-oauth-setup");
const githubOauthCallbackUrl = $("github-oauth-callback-url");
const githubOauthClientId = $("github-oauth-client-id");
const githubOauthClientSecret = $("github-oauth-client-secret");
const githubOauthSaveBtn = $("github-oauth-save-btn");
const githubOauthConfigStatus = $("github-oauth-config-status");
const codeJobForm = $("code-job-form");
const jobTitleInput = $("job-title-input");
const jobRepoSelect = $("job-repo-select");
const jobRepoUrlInput = $("job-repo-url-input");
const jobBaseBranchInput = $("job-base-branch-input");
const jobWorkBranchInput = $("job-work-branch-input");
const jobModeSelect = $("job-mode-select");
const jobPromptInput = $("job-prompt-input");
const jobDispatch = $("job-dispatch");
const jobRunTests = $("job-run-tests");
const jobOpenPr = $("job-open-pr");
const jobSubmitBtn = $("job-submit-btn");
const jobFormStatus = $("job-form-status");
const jobsCountPill = $("jobs-count-pill");
const jobsList = $("jobs-list");
const jobDetail = $("job-detail");

// Settings modal
const settingsModal = $("settings-modal");
const settingsClose = $("settings-close");
const settingsSysPrompt = $("settings-system-prompt");
const settingsTemp = $("settings-temperature");
const settingsTempVal = $("settings-temperature-val");
const settingsTopP = $("settings-top-p");
const settingsTopPVal = $("settings-top-p-val");
const settingsTopK = $("settings-top-k");
const modelListEl = $("model-list");
const workerListEl = $("worker-list");
const pullModelInput = $("pull-model-name");
const pullModelBtn = $("pull-model-btn");
const pullProgress = $("pull-progress");

/* ---------- HTTP helpers ---------- */

function authHeaders(extra = {}) {
  return authToken ? { Authorization: `Bearer ${authToken}`, ...extra } : { ...extra };
}

async function apiFetch(path, opts = {}) {
  const headers = { ...(opts.headers || {}), ...authHeaders() };
  if (opts.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const res = await fetch(`${API}${path}`, { ...opts, headers });
  if (res.status === 401 && authToken) {
    // Token expired or invalid — drop creds and bounce to login.
    handleLogout();
    throw new Error("Session expired, please sign in again");
  }
  return res;
}

async function apiJson(path, opts = {}) {
  const res = await apiFetch(path, opts);
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

async function apiJsonAny(paths, opts = {}) {
  let lastError = null;
  for (const path of paths) {
    try {
      return await apiJson(path, opts);
    } catch (err) {
      lastError = err;
      if (!isMissingEndpointError(err)) break;
    }
  }
  throw lastError || new Error("No API route configured");
}

function isMissingEndpointError(err) {
  const message = String(err?.message || "");
  return message.includes("(404)") || /not found/i.test(message);
}

/* ---------- Init ---------- */

async function init() {
  const stored = localStorage.getItem("auth");
  if (stored) {
    try {
      const parsed = JSON.parse(stored);
      authToken = parsed.token;
      currentUser = { id: parsed.id, username: parsed.username };
      await loadApp();
      handleGithubOAuthReturn();
      return;
    } catch {}
  }
  showLogin();
}

function handleGithubOAuthReturn() {
  const params = new URLSearchParams(window.location.search);
  const result = params.get("github_oauth");
  if (!result) return;
  const message = params.get("message");
  githubOauthNotice =
    result === "connected"
      ? { tone: "success", text: "GitHub sign-in completed." }
      : { tone: "error", text: `GitHub sign-in failed: ${message || "unknown error"}` };
  params.delete("github_oauth");
  params.delete("message");
  const next = `${window.location.pathname}${params.toString() ? `?${params}` : ""}${window.location.hash}`;
  window.history.replaceState({}, document.title, next);
  switchWorkspace("code");
  refreshCodeWorkspace();
}

function showLogin() {
  loginScreen.classList.remove("hidden");
  appEl.classList.add("hidden");
}

async function loadApp() {
  loginScreen.classList.add("hidden");
  appEl.classList.remove("hidden");
  sidebarUsername.textContent = currentUser.username;
  switchWorkspace("chat", { skipLoad: true });
  updateResearchToggle();
  await Promise.all([loadModels(), loadConversations(), refreshHealth(), refreshResearchStatus()]);
  showEmptyState();
  // Refresh health every 15s so the indicator catches Ollama coming back online.
  setInterval(refreshHealth, 15000);
}

/* ---------- Auth UI ---------- */

toggleMode.addEventListener("click", (e) => {
  e.preventDefault();
  loginMode = loginMode === "login" ? "register" : "login";
  if (loginMode === "register") {
    loginSubtitle.textContent = "Create an account";
    loginSubmit.textContent = "Register";
    toggleMode.textContent = "Already have an account? Sign in";
    passwordInput.autocomplete = "new-password";
  } else {
    loginSubtitle.textContent = "Sign in to continue";
    loginSubmit.textContent = "Sign in";
    toggleMode.textContent = "Need an account? Register";
    passwordInput.autocomplete = "current-password";
  }
  loginError.classList.add("hidden");
});

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = usernameInput.value.trim();
  const password = passwordInput.value;
  if (!username || !password) return;

  try {
    const path = loginMode === "register" ? "/auth/register" : "/auth/login";
    const res = await fetch(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Login failed");
    const data = await res.json();
    authToken = data.token;
    currentUser = { id: data.id, username: data.username };
    localStorage.setItem("auth", JSON.stringify(data));
    loginError.classList.add("hidden");
    passwordInput.value = "";
    await loadApp();
  } catch (err) {
    loginError.textContent = err.message;
    loginError.classList.remove("hidden");
  }
});

function handleLogout() {
  authToken = null;
  currentUser = null;
  currentConversation = null;
  currentConversationId = null;
  messages = [];
  conversations = [];
  localStorage.removeItem("auth");
  // Reset to login mode (user is more likely returning than registering).
  if (loginMode !== "login") toggleMode.click();
  passwordInput.value = "";
  showLogin();
}

logoutBtn.addEventListener("click", handleLogout);

/* ---------- Workspace navigation ---------- */

chatWorkspaceBtn?.addEventListener("click", () => switchWorkspace("chat"));
codeWorkspaceBtn?.addEventListener("click", () => switchWorkspace("code"));
githubRefreshBtn?.addEventListener("click", refreshGitHubIntegration);
jobsRefreshBtn?.addEventListener("click", refreshCodeJobs);
codeJobForm?.addEventListener("submit", createCodeJob);
githubConnectBtn?.addEventListener("click", startGithubSignIn);
githubDisconnectBtn?.addEventListener("click", disconnectGithub);
githubOauthSaveBtn?.addEventListener("click", saveGithubOAuthConfig);

githubRepoSelect?.addEventListener("change", () => {
  if (githubRepoSelect.value && jobRepoSelect) {
    jobRepoSelect.value = githubRepoSelect.value;
    applyRepositoryDefaults(githubRepoSelect.value);
  }
});

jobRepoSelect?.addEventListener("change", () => {
  if (jobRepoSelect.value) {
    jobRepoUrlInput.value = "";
    applyRepositoryDefaults(jobRepoSelect.value);
  }
});

function switchWorkspace(target, { skipLoad = false } = {}) {
  currentWorkspace = target;
  const isCode = target === "code";

  chatWorkspace?.classList.toggle("hidden", isCode);
  codeWorkspace?.classList.toggle("hidden", !isCode);
  chatWorkspaceBtn?.classList.toggle("active", !isCode);
  codeWorkspaceBtn?.classList.toggle("active", isCode);
  chatWorkspaceBtn?.setAttribute("aria-selected", String(!isCode));
  codeWorkspaceBtn?.setAttribute("aria-selected", String(isCode));

  if (isCode && !skipLoad && !codeWorkspaceLoaded) {
    codeWorkspaceLoaded = true;
    refreshCodeWorkspace();
  }
}

async function refreshCodeWorkspace() {
  await Promise.allSettled([refreshGitHubIntegration(), refreshCodeJobs()]);
}

/* ---------- Health ---------- */

async function refreshHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    const ok = data.ollama === "ok";
    const modelCount = data.model_count || 0;
    const modelWord = modelCount === 1 ? "model" : "models";
    const workerText = formatHealthWorkers(data.workers);
    const label = ok
      ? `Ollama ready - ${modelCount} ${modelWord}${workerText}`
      : `Ollama unreachable${workerText}`;
    healthIndicator.classList.toggle("ok", ok);
    healthIndicator.classList.toggle("down", !ok);
    healthIndicator.querySelector(".label").textContent = label;
    healthIndicator.title = label;
  } catch {
    healthIndicator.classList.remove("ok");
    healthIndicator.classList.add("down");
    healthIndicator.querySelector(".label").textContent = "Backend unreachable";
    healthIndicator.title = "Backend unreachable";
  }
}

function formatHealthWorkers(workers) {
  if (!workers || !Number.isFinite(workers.enabled)) return "";
  if (workers.enabled < 1) return " - no workers enabled";

  const available = Number.isFinite(workers.available) ? workers.available : 0;
  const workerWord = workers.enabled === 1 ? "worker" : "workers";
  const busy =
    Number.isFinite(workers.busy) && workers.busy > 0
      ? ` - ${workers.busy} active`
      : "";
  return ` - ${available}/${workers.enabled} ${workerWord}${busy}`;
}

async function refreshResearchStatus() {
  try {
    const data = await apiJson("/research/status");
    webResearchAvailable = data.enabled !== false;
    if (!webResearchAvailable) webResearchEnabled = false;
  } catch {
    webResearchAvailable = false;
    webResearchEnabled = false;
  }
  updateResearchToggle();
}

function updateResearchToggle() {
  if (!researchToggle) return;
  researchToggle.classList.toggle("active", webResearchEnabled && webResearchAvailable);
  researchToggle.disabled = !webResearchAvailable || isStreaming;
  researchToggle.setAttribute("aria-pressed", String(webResearchEnabled && webResearchAvailable));
  researchToggle.title = webResearchAvailable
    ? "Use web research"
    : "Web research unavailable";
}

researchToggle?.addEventListener("click", () => {
  if (!webResearchAvailable || isStreaming) return;
  webResearchEnabled = !webResearchEnabled;
  updateResearchToggle();
});

/* ---------- Models ---------- */

async function loadModels() {
  try {
    const data = await apiJson("/models");
    const models = [...(data.models ?? [])].sort(
      (a, b) => (a.size || 0) - (b.size || 0) || String(a.name).localeCompare(String(b.name))
    );
    modelSelect.replaceChildren();
    if (models.length) {
      for (const model of models) {
        const option = document.createElement("option");
        option.value = model.name;
        option.textContent = model.name;
        modelSelect.appendChild(option);
      }
    } else {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No models found";
      modelSelect.appendChild(option);
    }
    // Restore the conversation's saved model if applicable.
    if (currentConversation?.model) modelSelect.value = currentConversation.model;
    return models;
  } catch {
    modelSelect.innerHTML = `<option value="">Ollama unavailable</option>`;
    return [];
  }
}

modelSelect.addEventListener("change", async () => {
  // Persist model selection to the active conversation if one exists.
  if (!currentConversationId) return;
  try {
    await apiJson(`/conversations/${currentConversationId}`, {
      method: "PATCH",
      body: JSON.stringify({ model: modelSelect.value }),
    });
    currentConversation.model = modelSelect.value;
  } catch (err) {
    console.error("Failed to update conversation model:", err);
  }
});

/* ---------- Conversations ---------- */

async function loadConversations() {
  try {
    conversations = await apiJson("/conversations");
    renderConversations();
  } catch {}
}

function renderConversations() {
  conversationsList.innerHTML = "";
  for (const conv of conversations) {
    const item = document.createElement("div");
    item.className = "conv-item" + (conv.id === currentConversationId ? " active" : "");

    const title = document.createElement("span");
    title.className = "conv-title";
    title.textContent = conv.title;
    title.title = "Double-click to rename";

    title.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      startRename(conv.id, title);
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "conv-delete";
    deleteBtn.textContent = "×";
    deleteBtn.title = "Delete";
    deleteBtn.onclick = (e) => {
      e.stopPropagation();
      deleteConversation(conv.id);
    };

    item.appendChild(title);
    item.appendChild(deleteBtn);
    item.onclick = () => selectConversation(conv.id);
    conversationsList.appendChild(item);
  }
}

function startRename(id, titleEl) {
  const original = titleEl.textContent;
  titleEl.contentEditable = "true";
  titleEl.focus();
  // Select all text inside the element.
  const range = document.createRange();
  range.selectNodeContents(titleEl);
  const sel = window.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);

  let committed = false;
  const commit = async () => {
    if (committed) return;
    committed = true;
    titleEl.contentEditable = "false";
    const newTitle = titleEl.textContent.trim();
    if (!newTitle || newTitle === original) {
      titleEl.textContent = original;
      return;
    }
    try {
      const updated = await apiJson(`/conversations/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ title: newTitle }),
      });
      const conv = conversations.find((c) => c.id === id);
      if (conv) conv.title = updated.title;
      if (id === currentConversationId) chatTitle.textContent = updated.title;
    } catch (err) {
      titleEl.textContent = original;
      alert(`Rename failed: ${err.message}`);
    }
  };

  titleEl.addEventListener("blur", commit, { once: true });
  titleEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      titleEl.blur();
    } else if (e.key === "Escape") {
      e.preventDefault();
      titleEl.textContent = original;
      committed = true;
      titleEl.contentEditable = "false";
      titleEl.blur();
    }
  });
}

async function selectConversation(id) {
  if (isStreaming) return;
  currentConversationId = id;
  currentConversation = conversations.find((c) => c.id === id) || null;
  if (currentConversation) {
    chatTitle.textContent = currentConversation.title;
    if (currentConversation.model) modelSelect.value = currentConversation.model;
  }
  renderConversations();
  try {
    messages = await apiJson(`/conversations/${id}/messages`);
    renderMessages();
    updateTokenCounter();
  } catch {}
}

async function deleteConversation(id) {
  try {
    await apiFetch(`/conversations/${id}`, { method: "DELETE" });
    conversations = conversations.filter((c) => c.id !== id);
    if (currentConversationId === id) {
      currentConversationId = null;
      currentConversation = null;
      messages = [];
      chatTitle.textContent = "Local LLM Chat";
      showEmptyState();
      updateTokenCounter();
    }
    renderConversations();
  } catch {}
}

newChatBtn.addEventListener("click", () => {
  if (isStreaming) return;
  switchWorkspace("chat", { skipLoad: true });
  currentConversationId = null;
  currentConversation = null;
  messages = [];
  chatTitle.textContent = "New Chat";
  showEmptyState();
  renderConversations();
  updateTokenCounter();
  inputEl.focus();
});

/* ---------- Messages rendering ---------- */

function showEmptyState() {
  messagesEl.innerHTML = "";
  const el = document.createElement("div");
  el.className = "empty-state";
  el.textContent = "Start a new conversation";
  messagesEl.appendChild(el);
}

function renderMessages() {
  messagesEl.innerHTML = "";
  messages.forEach((msg, idx) => {
    appendMessage(msg.role, msg.content, {
      id: msg.id,
      index: idx,
      isLast: idx === messages.length - 1,
    });
  });
  if (messages.length === 0) showEmptyState();
}

function renderMarkdown(content) {
  return DOMPurify.sanitize(marked.parse(content));
}

// Enhance a freshly-rendered assistant bubble: syntax-highlight code,
// and inject a copy button into each <pre>.
function enhanceCodeBlocks(bubble) {
  bubble.querySelectorAll("pre code").forEach((code) => {
    try {
      if (window.hljs) hljs.highlightElement(code);
    } catch {}
    const pre = code.parentElement;
    if (!pre || pre.querySelector(".code-copy-btn")) return;
    const btn = document.createElement("button");
    btn.className = "code-copy-btn";
    btn.textContent = "Copy";
    btn.onclick = async () => {
      try {
        await navigator.clipboard.writeText(code.innerText);
        btn.textContent = "Copied";
        btn.classList.add("copied");
        setTimeout(() => {
          btn.textContent = "Copy";
          btn.classList.remove("copied");
        }, 1200);
      } catch {
        btn.textContent = "Failed";
      }
    };
    pre.appendChild(btn);
  });
}

function appendMessage(role, content = "", { id, index, skipActions, isLast } = {}) {
  // Clear empty-state placeholder if present.
  const emptyState = messagesEl.querySelector(".empty-state");
  if (emptyState) emptyState.remove();

  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  if (id !== undefined) wrapper.dataset.messageId = String(id);
  if (index !== undefined) wrapper.dataset.index = String(index);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant" && content) {
    bubble.innerHTML = renderMarkdown(content);
    enhanceCodeBlocks(bubble);
  } else {
    bubble.textContent = content;
  }
  wrapper.appendChild(bubble);

  // Per-message actions (edit on user, regenerate on last assistant).
  if (!skipActions) {
    const actions = makeMessageActions(role, wrapper, { isLast });
    if (actions) wrapper.appendChild(actions);
  }

  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function makeMessageActions(role, wrapper, { isLast } = {}) {
  if (isStreaming) return null;
  const actions = document.createElement("div");
  actions.className = "message-actions";

  if (role === "user") {
    const editBtn = document.createElement("button");
    editBtn.textContent = "Edit";
    editBtn.onclick = () => startEditMessage(wrapper);
    actions.appendChild(editBtn);
  }
  // Regenerate is shown only on the most recent assistant message and
  // only when not currently streaming.
  if (role === "assistant" && isLast) {
    const regenBtn = document.createElement("button");
    regenBtn.textContent = "↺ Regenerate";
    regenBtn.onclick = () => regenerateLastAssistant();
    actions.appendChild(regenBtn);
  }
  return actions.children.length ? actions : null;
}

/* ---------- Token estimation ---------- */

// Rough heuristic: ~4 characters per token. Good enough for the UI
// indicator — we don't make routing decisions from it.
function estimateTokens(text) {
  return Math.ceil((text || "").length / 4);
}

function updateTokenCounter() {
  if (!messages.length) {
    tokenCounter.classList.add("hidden");
    return;
  }
  const total =
    estimateTokens(currentConversation?.system_prompt || "") +
    messages.reduce((sum, m) => sum + estimateTokens(m.content), 0);
  const window_ = DEFAULT_CONTEXT_WINDOW;
  tokenCounter.textContent = `~${total.toLocaleString()} / ${window_.toLocaleString()} tok`;
  tokenCounter.classList.remove("hidden", "warn", "danger");
  const ratio = total / window_;
  if (ratio > 0.95) tokenCounter.classList.add("danger");
  else if (ratio > 0.75) tokenCounter.classList.add("warn");
}

/* ---------- Edit message ---------- */

function startEditMessage(wrapper) {
  if (isStreaming) return;
  const index = parseInt(wrapper.dataset.index, 10);
  const original = messages[index]?.content || "";
  const messageId = wrapper.dataset.messageId;
  if (!messageId) {
    alert("This message can't be edited yet (still saving).");
    return;
  }

  wrapper.innerHTML = "";
  const textarea = document.createElement("textarea");
  textarea.className = "edit-textarea";
  textarea.value = original;
  textarea.rows = Math.min(8, Math.max(2, original.split("\n").length));
  wrapper.appendChild(textarea);

  const actions = document.createElement("div");
  actions.className = "edit-actions";
  const saveBtn = document.createElement("button");
  saveBtn.textContent = "Save & resend";
  const cancelBtn = document.createElement("button");
  cancelBtn.className = "cancel-edit";
  cancelBtn.textContent = "Cancel";
  actions.append(saveBtn, cancelBtn);
  wrapper.appendChild(actions);
  textarea.focus();

  cancelBtn.onclick = () => renderMessages();
  saveBtn.onclick = async () => {
    const newContent = textarea.value.trim();
    if (!newContent) return;
    saveBtn.disabled = true;
    cancelBtn.disabled = true;
    try {
      // Truncate the conversation at (and including) the edited message,
      // then submit the new content as a fresh user turn.
      await apiFetch(
        `/conversations/${currentConversationId}/messages/from/${messageId}`,
        { method: "DELETE" }
      );
      messages = messages.slice(0, index);
      renderMessages();
      await sendMessageContent(newContent);
    } catch (err) {
      alert(`Edit failed: ${err.message}`);
      renderMessages();
    }
  };
}

/* ---------- Regenerate ---------- */

async function regenerateLastAssistant() {
  if (isStreaming || !currentConversationId) return;
  // Remove the last assistant message from the local state immediately;
  // the backend will delete it from the DB.
  const lastIdx = [...messages].map((m) => m.role).lastIndexOf("assistant");
  if (lastIdx === -1) return;
  messages = messages.slice(0, lastIdx);
  renderMessages();
  await runChatStream({ regenerate: true });
}

/* ---------- Sending messages ---------- */

async function sendMessage() {
  const content = inputEl.value.trim();
  if (!content) return;
  inputEl.value = "";
  autoResize();
  await sendMessageContent(content);
}

async function sendMessageContent(content) {
  if (isStreaming) return;
  const model = modelSelect.value;
  if (!model) {
    alert("No model selected. Pull a model in Settings or start Ollama.");
    return;
  }
  messages.push({ role: "user", content });
  appendMessage("user", content, { skipActions: true });
  await runChatStream({ regenerate: false });
}

function setStreamStatus(bubble, cursor, currentStatusEl, text) {
  if (!text) {
    currentStatusEl?.remove();
    return null;
  }
  const el = currentStatusEl || document.createElement("span");
  el.className = "stream-status";
  el.textContent = text;
  if (!el.parentElement) {
    bubble.insertBefore(el, cursor);
  }
  return el;
}

function researchStatusText(status, sourceCount) {
  if (status === "ok" && sourceCount > 0) {
    const sourceWord = sourceCount === 1 ? "source" : "sources";
    return `Using ${sourceCount} web ${sourceWord}...`;
  }
  if (status === "disabled") return "Web research disabled...";
  if (status === "empty") return "No web sources found...";
  if (status === "error") return "Research unavailable...";
  return "Searching web...";
}

async function runChatStream({ regenerate }) {
  const model = modelSelect.value;
  const bubble = appendMessage("assistant", "", { skipActions: true });
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  bubble.appendChild(cursor);

  setStreaming(true);
  streamAbortController = new AbortController();

  let assistantContent = "";
  let aborted = false;
  let statusEl = null;
  const useResearch = webResearchEnabled && webResearchAvailable;

  try {
    const body = {
      messages: messages.map(({ role, content }) => ({ role, content })),
      conversation_id: currentConversationId,
      regenerate,
    };
    if (useResearch) {
      body.web_research = true;
      statusEl = setStreamStatus(bubble, cursor, statusEl, "Searching web...");
    }
    // Settings only ship to the backend for the first message of a new
    // conversation; thereafter the backend reads them from the DB row.
    if (!currentConversationId) {
      body.model = model;
      body.system_prompt = currentConversation?.system_prompt || "";
      body.temperature = currentConversation?.temperature ?? 0.7;
      body.top_p = currentConversation?.top_p ?? 0.9;
      body.top_k = currentConversation?.top_k ?? 40;
    }

    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
      signal: streamAbortController.signal,
    });

    if (!res.ok) {
      let detail = `Server error: ${res.status}`;
      try { detail = (await res.json()).detail || detail; } catch {}
      throw new Error(detail);
    }

    const returnedId = res.headers.get("X-Conversation-Id");
    if (returnedId) currentConversationId = parseInt(returnedId, 10);

    const backendName = res.headers.get("X-LLM-Backend");
    const modelStatus = res.headers.get("X-LLM-Model-Status");
    if (useResearch) {
      const researchStatus = res.headers.get("X-Research-Status");
      const sourceCount = parseInt(res.headers.get("X-Research-Source-Count") || "0", 10);
      statusEl = setStreamStatus(
        bubble,
        cursor,
        statusEl,
        researchStatusText(researchStatus, sourceCount)
      );
    }
    if (modelStatus === "loading") {
      const loadingText = `Loading ${model} on ${backendName || "worker"}...`;
      statusEl = setStreamStatus(
        bubble,
        cursor,
        statusEl,
        statusEl ? `${statusEl.textContent} ${loadingText}` : loadingText
      );
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (statusEl) {
        statusEl.remove();
        statusEl = null;
      }
      assistantContent += decoder.decode(value, { stream: true });
      bubble.innerHTML = renderMarkdown(assistantContent);
      bubble.appendChild(cursor);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    const tail = decoder.decode();
    if (tail) {
      assistantContent += tail;
      bubble.innerHTML = renderMarkdown(assistantContent);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      aborted = true;
    } else {
      bubble.textContent = `Error: ${err.message}`;
    }
  } finally {
    if (statusEl) statusEl.remove();
    cursor.remove();
    if (useResearch) {
      webResearchEnabled = false;
    }
    if (assistantContent) {
      enhanceCodeBlocks(bubble);
      messages.push({ role: "assistant", content: assistantContent });
    } else if (aborted) {
      bubble.textContent = "[stopped]";
    }
    setStreaming(false);
    streamAbortController = null;
    inputEl.focus();
    await loadConversations();
    // The backend deletes orphan conversations (e.g. aborted before any
    // content was streamed). Reset our dangling id so the next send
    // creates a fresh conversation instead of 404-ing.
    if (
      currentConversationId &&
      !conversations.find((c) => c.id === currentConversationId)
    ) {
      currentConversationId = null;
      currentConversation = null;
      messages = [];
      chatTitle.textContent = "Local LLM Chat";
      showEmptyState();
    } else if (currentConversationId) {
      // After a new conversation, reload its row for accurate settings + title.
      currentConversation =
        conversations.find((c) => c.id === currentConversationId) || currentConversation;
      if (currentConversation) chatTitle.textContent = currentConversation.title;
      // Re-render to pick up message ids and per-message actions.
      try {
        messages = await apiJson(`/conversations/${currentConversationId}/messages`);
        renderMessages();
      } catch {}
    }
    updateTokenCounter();
  }
}

function setStreaming(active) {
  isStreaming = active;
  sendBtn.disabled = active;
  inputEl.disabled = active;
  updateResearchToggle();
  stopBtn.classList.toggle("hidden", !active);
  sendBtn.classList.toggle("hidden", active);
}

stopBtn.addEventListener("click", () => {
  if (streamAbortController) streamAbortController.abort();
});

function autoResize() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${inputEl.scrollHeight}px`;
}

inputEl.addEventListener("input", autoResize);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
sendBtn.addEventListener("click", sendMessage);

/* ---------- Code Jobs workspace ---------- */

async function refreshGitHubIntegration() {
  if (!githubStatusPill || !githubSummary) return;

  setGithubStatus("checking", "warning", "Checking GitHub integration...");
  let statusRepositories = [];
  try {
    githubStatus = await apiJson("/github/status");
    renderGithubStatus(githubStatus);
    statusRepositories = repositoriesFromGithubStatus(githubStatus);
  } catch (err) {
    githubStatus = null;
    setGithubStatus(
      "setup needed",
      "warning",
      `GitHub status unavailable: ${err.message}. Backend integration may not be installed yet.`
    );
  }

  try {
    const payload = await apiJsonAny(["/github/repositories", "/github/repos"]);
    githubRepositories = normalizeRepositories(payload);
    renderRepositoryOptions();
    if (githubRepositories.length) {
      const repoWord = githubRepositories.length === 1 ? "repository" : "repositories";
      githubSummary.textContent = `${githubRepositories.length} ${repoWord} available for code jobs.`;
    }
  } catch (err) {
    githubRepositories = statusRepositories;
    if (githubRepositories.length) {
      renderRepositoryOptions();
      const repo = githubRepositories[0].full_name;
      githubSummary.textContent = `Using configured default repository ${repo}. Repository browsing is unavailable: ${err.message}.`;
    } else {
      renderRepositoryOptions(`Repositories unavailable: ${err.message}`);
    }
  }
}

function setGithubStatus(label, tone, summary) {
  githubStatusPill.textContent = label;
  githubStatusPill.dataset.tone = tone;
  githubSummary.textContent = summary;
}

function renderGithubStatus(status) {
  const connected = Boolean(status.connected ?? status.authenticated ?? status.ok ?? false);
  const configured = Boolean(status.configured ?? status.oauth?.configured ?? false);
  const account =
    status.connection?.account_login ||
    status.installation?.account_login ||
    status.account ||
    status.username ||
    status.login ||
    status.user?.login ||
    status.user?.name ||
    repositoriesFromGithubStatus(status)[0]?.full_name ||
    "GitHub";
  const detail = status.error || status.detail || status.message || "";
  const rateLimit = status.rate_limit?.remaining ?? status.rateLimit?.remaining;

  if (githubOauthCallbackUrl && status.oauth?.callback_url) {
    githubOauthCallbackUrl.value = status.oauth.callback_url;
  }
  if (githubOauthSetup) githubOauthSetup.classList.toggle("hidden", configured);
  if (githubConnectBtn) {
    githubConnectBtn.disabled = !configured;
    githubConnectBtn.querySelector("span").textContent = connected ? "Reconnect GitHub" : "Sign in with GitHub";
  }
  githubDisconnectBtn?.classList.toggle("hidden", !connected);

  if (githubOauthNotice && githubOauthConfigStatus) {
    githubOauthConfigStatus.textContent = githubOauthNotice.text;
    githubOauthConfigStatus.classList.remove("error", "success");
    githubOauthConfigStatus.classList.add(githubOauthNotice.tone);
    githubOauthNotice = null;
  }

  if (connected) {
    const limitText = Number.isFinite(rateLimit) ? ` API remaining: ${rateLimit}.` : "";
    setGithubStatus("connected", "success", `Signed in as ${account}.${limitText}`);
  } else if (configured) {
    setGithubStatus(
      "ready",
      "warning",
      "GitHub OAuth is ready. Sign in with GitHub to authorize this browser account."
    );
  } else {
    setGithubStatus(
      "setup needed",
      "warning",
      detail || "Set up the Local LLM GitHub OAuth App once, then every user can sign in with GitHub."
    );
  }
}

async function saveGithubOAuthConfig() {
  const clientId = githubOauthClientId?.value.trim() || "";
  const clientSecret = githubOauthClientSecret?.value.trim() || "";
  if (!clientId || !clientSecret) {
    if (githubOauthConfigStatus) {
      githubOauthConfigStatus.textContent = "Client ID and Client Secret are required for the one-time service setup.";
      githubOauthConfigStatus.classList.add("error");
    }
    return;
  }

  if (githubOauthConfigStatus) {
    githubOauthConfigStatus.textContent = "Saving GitHub OAuth App...";
    githubOauthConfigStatus.classList.remove("error", "success");
  }

  try {
    await apiJson("/github/oauth/config", {
      method: "POST",
      body: JSON.stringify({ client_id: clientId, client_secret: clientSecret }),
    });
    if (githubOauthClientSecret) githubOauthClientSecret.value = "";
    if (githubOauthConfigStatus) {
      githubOauthConfigStatus.textContent = "GitHub OAuth App saved. Users can now sign in with GitHub.";
      githubOauthConfigStatus.classList.add("success");
    }
    await refreshGitHubIntegration();
  } catch (err) {
    if (githubOauthConfigStatus) {
      githubOauthConfigStatus.textContent = err.message;
      githubOauthConfigStatus.classList.add("error");
    }
  }
}

async function startGithubSignIn() {
  try {
    const data = await apiJson("/github/oauth/start", { method: "POST" });
    if (!data.configured || !data.auth_url) {
      if (githubOauthConfigStatus) {
        githubOauthConfigStatus.textContent = `GitHub OAuth is not configured: ${(data.missing || []).join(", ")}`;
        githubOauthConfigStatus.classList.add("error");
      }
      return;
    }
    window.location.href = data.auth_url;
  } catch (err) {
    if (githubOauthConfigStatus) {
      githubOauthConfigStatus.textContent = `GitHub sign-in failed to start: ${err.message}`;
      githubOauthConfigStatus.classList.add("error");
    } else {
      alert(`GitHub sign-in failed to start: ${err.message}`);
    }
  }
}

async function disconnectGithub() {
  try {
    await apiJson("/github/install", { method: "DELETE" });
    await refreshGitHubIntegration();
  } catch (err) {
    if (githubOauthConfigStatus) {
      githubOauthConfigStatus.textContent = `GitHub disconnect failed: ${err.message}`;
      githubOauthConfigStatus.classList.add("error");
    }
  }
}

function repositoriesFromGithubStatus(status) {
  const repo = status.default_repository || status.defaultRepository;
  const owner = repo?.owner;
  const name = repo?.name;
  if (!owner || !name) return [];
  return [{
    id: `${owner}/${name}`,
    full_name: `${owner}/${name}`,
    name,
    default_branch: repo.default_branch || repo.defaultBranch || "main",
    html_url: `https://github.com/${owner}/${name}`,
  }];
}

function normalizeRepositories(payload) {
  const items = Array.isArray(payload)
    ? payload
    : payload.repositories || payload.repos || payload.data || payload.items || [];

  return items
    .map((repo) => {
      if (typeof repo === "string") {
        return { id: repo, full_name: repo, name: repo, default_branch: "main" };
      }
      const fullName =
        repo.full_name ||
        repo.fullName ||
        repo.slug ||
        repo.repository ||
        repo.name_with_owner ||
        repo.name;
      if (!fullName) return null;
      return {
        id: String(repo.id ?? fullName),
        full_name: fullName,
        name: repo.name || fullName,
        default_branch: repo.default_branch || repo.defaultBranch || "main",
        html_url: repo.html_url || repo.web_url || repo.url || "",
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.full_name.localeCompare(b.full_name));
}

function renderRepositoryOptions(errorText) {
  replaceRepoOptions(githubRepoSelect, errorText || "Select repository", githubRepositories);
  replaceRepoOptions(jobRepoSelect, "Select from GitHub or enter below", githubRepositories);
}

function replaceRepoOptions(select, placeholder, repositories) {
  if (!select) return;
  const previous = select.value;
  select.replaceChildren();
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = placeholder;
  select.appendChild(empty);

  for (const repo of repositories) {
    const option = document.createElement("option");
    option.value = repo.full_name;
    option.textContent = repo.full_name;
    select.appendChild(option);
  }

  if (previous && repositories.some((repo) => repo.full_name === previous)) {
    select.value = previous;
  }
}

function applyRepositoryDefaults(fullName) {
  const repo = githubRepositories.find((item) => item.full_name === fullName);
  if (repo?.default_branch && !jobBaseBranchInput.value.trim()) {
    jobBaseBranchInput.value = repo.default_branch;
  }
}

async function refreshCodeJobs() {
  if (!jobsList) return;
  jobsList.textContent = "Loading jobs...";
  try {
    const payload = await apiJsonAny(["/agent/jobs", "/code-jobs"]);
    codeJobs = normalizeCodeJobs(payload);
    if (selectedCodeJobId && !codeJobs.some((job) => job.id === selectedCodeJobId)) {
      selectedCodeJobId = null;
    }
    renderCodeJobs();
  } catch (err) {
    codeJobs = [];
    selectedCodeJobId = null;
    renderCodeJobs(`Code Jobs unavailable: ${err.message}`);
  }
}

function normalizeCodeJobs(payload) {
  const items = Array.isArray(payload)
    ? payload
    : payload.jobs || payload.items || payload.data || payload.results || [];
  return items.map(normalizeCodeJob).filter((job) => job.id);
}

function normalizeCodeJob(job) {
  const id = String(job.id ?? job.job_id ?? job.jobId ?? job.name ?? "");
  const instructions = job.instructions || job.prompt || job.description || "";
  const repository = repositoryLabel(
    job.repository ||
      job.repo ||
      job.repo_full_name ||
      job.github_repository ||
      job.githubRepository ||
      ""
  );
  const result = job.result || {};
  return {
    id,
    title: job.title || job.name || firstLine(instructions) || `Job ${id}`,
    status: String(job.status || job.state || job.phase || "unknown").toLowerCase(),
    repository,
    base_branch: job.base_branch || job.baseBranch || "",
    work_branch:
      job.work_branch ||
      job.target_branch ||
      job.targetBranch ||
      job.branch ||
      job.head_branch ||
      job.headBranch ||
      "",
    created_at: job.created_at || job.createdAt || job.created || "",
    updated_at: job.updated_at || job.updatedAt || job.finished_at || job.finishedAt || "",
    pull_request_url:
      job.pull_request_url ||
      job.pr_url ||
      job.pullRequestUrl ||
      job.pull_request?.html_url ||
      result.pull_request_url ||
      result.pr_url ||
      "",
    branch_url: job.branch_url || job.branchUrl || "",
    repo_url: job.repo_url || job.repository_url || job.repositoryUrl || repositoryUrl(repository),
    summary: job.summary || job.status_detail || job.message || result.summary || "",
    logs: job.logs || job.log || job.output || job.events || "",
    raw: job,
  };
}

function repositoryLabel(value) {
  if (!value) return "";
  if (typeof value === "string") return value;
  if (value.full_name) return value.full_name;
  if (value.fullName) return value.fullName;
  if (value.owner && value.name) return `${value.owner}/${value.name}`;
  if (value.repository_owner && value.repository_name) {
    return `${value.repository_owner}/${value.repository_name}`;
  }
  return "";
}

function repositoryUrl(fullName) {
  if (!fullName || !/^[^/\s]+\/[^/\s]+$/.test(fullName)) return "";
  return `https://github.com/${fullName}`;
}

function renderCodeJobs(errorText) {
  jobsList.replaceChildren();
  const jobWord = codeJobs.length === 1 ? "job" : "jobs";
  jobsCountPill.textContent = `${codeJobs.length} ${jobWord}`;

  if (errorText) {
    jobsList.textContent = errorText;
    renderJobDetail(null);
    return;
  }

  if (!codeJobs.length) {
    jobsList.textContent = "No code jobs yet.";
    renderJobDetail(null);
    return;
  }

  for (const job of codeJobs) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "job-row";
    row.classList.toggle("active", job.id === selectedCodeJobId);
    row.onclick = () => selectCodeJob(job.id);

    const main = document.createElement("span");
    main.className = "job-row-main";

    const title = document.createElement("span");
    title.className = "job-row-title";
    title.textContent = job.title;

    const subtitle = document.createElement("span");
    subtitle.className = "job-row-subtitle";
    subtitle.textContent = [job.repository, job.work_branch, formatDateTime(job.updated_at || job.created_at)]
      .filter(Boolean)
      .join(" - ");

    const status = document.createElement("span");
    status.className = `job-status ${job.status}`;
    status.textContent = job.status;

    main.append(title, subtitle);
    row.append(main, status);
    jobsList.appendChild(row);
  }

  if (selectedCodeJobId) {
    renderJobDetail(codeJobs.find((job) => job.id === selectedCodeJobId) || null);
  } else {
    renderJobDetail(null);
  }
}

async function selectCodeJob(id) {
  selectedCodeJobId = id;
  const localJob = codeJobs.find((job) => job.id === id) || null;
  renderCodeJobs();
  renderJobDetail(localJob, { loading: true });

  try {
    const detail = normalizeCodeJob(
      await apiJsonAny([
        `/agent/jobs/${encodeURIComponent(id)}`,
        `/code-jobs/${encodeURIComponent(id)}`,
      ])
    );
    codeJobs = codeJobs.map((job) => (job.id === id ? detail : job));
    renderCodeJobs();
    renderJobDetail(detail);
  } catch (err) {
    renderJobDetail(localJob, { error: `Detail refresh failed: ${err.message}` });
  }
}

function renderJobDetail(job, { loading = false, error = "" } = {}) {
  jobDetail.replaceChildren();
  if (!job) {
    jobDetail.textContent = "Select a job to inspect status, logs, and links.";
    return;
  }

  const title = document.createElement("h3");
  title.textContent = job.title;
  jobDetail.appendChild(title);

  if (loading || error) {
    const note = document.createElement("p");
    note.className = error ? "form-status error" : "form-status";
    note.textContent = error || "Loading latest detail...";
    jobDetail.appendChild(note);
  }

  const grid = document.createElement("div");
  grid.className = "job-detail-grid";
  appendJobMeta(grid, "Status", job.status);
  appendJobMeta(grid, "Repository", job.repository || "unknown");
  appendJobMeta(grid, "Work branch", job.work_branch || "not set");
  appendJobMeta(grid, "Updated", formatDateTime(job.updated_at || job.created_at) || "unknown");
  jobDetail.appendChild(grid);

  if (job.summary) {
    const summary = document.createElement("p");
    summary.textContent = job.summary;
    jobDetail.appendChild(summary);
  }

  const links = renderJobLinks(job);
  if (links.children.length) jobDetail.appendChild(links);

  const log = document.createElement("pre");
  log.className = "job-log";
  log.textContent = jobLogText(job);
  jobDetail.appendChild(log);
}

function appendJobMeta(parent, label, value) {
  const item = document.createElement("div");
  item.className = "job-meta";
  const strong = document.createElement("strong");
  strong.textContent = label;
  const span = document.createElement("span");
  span.textContent = value || "unknown";
  item.append(strong, span);
  parent.appendChild(item);
}

function renderJobLinks(job) {
  const links = document.createElement("div");
  links.className = "job-links";
  appendJobLink(links, "Repository", job.repo_url);
  appendJobLink(links, "Branch", job.branch_url);
  appendJobLink(links, "Pull request", job.pull_request_url);
  return links;
}

function appendJobLink(parent, label, href) {
  if (!href) return;
  const link = document.createElement("a");
  link.href = href;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = label;
  parent.appendChild(link);
}

function jobLogText(job) {
  const logs = job.logs;
  if (!logs) return "No logs available yet.";
  if (Array.isArray(logs)) return logs.map(formatLogEntry).join("\n");
  if (typeof logs === "object") return JSON.stringify(logs, null, 2);
  return String(logs);
}

function formatLogEntry(entry) {
  if (typeof entry === "string") return entry;
  const time = entry.time || entry.timestamp || entry.created_at || "";
  const level = entry.level || entry.status || "";
  const message = entry.message || entry.text || entry.detail || JSON.stringify(entry);
  return [time, level, message].filter(Boolean).join(" ");
}

async function createCodeJob(e) {
  e.preventDefault();
  const title = jobTitleInput.value.trim();
  const selectedRepo = jobRepoSelect.value.trim();
  const manualRepo = jobRepoUrlInput.value.trim();
  const repository = selectedRepo || manualRepo;
  const instructions = jobPromptInput.value.trim();
  const repoParts = parseRepository(repository);

  if (!repository) {
    showJobFormStatus("Choose a repository or enter one manually.", "error");
    return;
  }
  if (!repoParts) {
    showJobFormStatus("Use a GitHub repository in owner/repo or GitHub URL format.", "error");
    return;
  }
  if (!instructions) {
    showJobFormStatus("Add instructions before starting a job.", "error");
    return;
  }
  if (jobDispatch.checked && !githubStatus?.connected) {
    showJobFormStatus("Sign in with GitHub before dispatching a code job.", "error");
    return;
  }

  const body = {
    title,
    prompt: instructions,
    repository_owner: repoParts.owner,
    repository_name: repoParts.name,
    base_branch: jobBaseBranchInput.value.trim() || "main",
    target_branch: jobWorkBranchInput.value.trim() || null,
    dispatch: jobDispatch.checked,
    metadata: {
      mode: jobModeSelect.value,
      run_tests: jobRunTests.checked,
      open_pull_request: jobOpenPr.checked,
      repository: repoParts.fullName,
    },
  };

  jobSubmitBtn.disabled = true;
  showJobFormStatus("Starting code job...", "");
  try {
    const created = await apiJsonAny(["/agent/jobs", "/code-jobs"], {
      method: "POST",
      body: JSON.stringify(body),
    });
    const job = normalizeCodeJob(created.job || created);
    selectedCodeJobId = job.id || selectedCodeJobId;
    showJobFormStatus("Code job started.", "success");
    codeJobForm.reset();
    jobBaseBranchInput.value = "main";
    jobDispatch.checked = true;
    jobRunTests.checked = true;
    jobOpenPr.checked = true;
    await refreshCodeJobs();
    if (job.id) selectCodeJob(job.id);
  } catch (err) {
    showJobFormStatus(`Failed to start job: ${err.message}`, "error");
  } finally {
    jobSubmitBtn.disabled = false;
  }
}

function showJobFormStatus(message, tone) {
  jobFormStatus.textContent = message;
  jobFormStatus.classList.remove("hidden", "error", "success");
  if (tone) jobFormStatus.classList.add(tone);
}

function parseRepository(value) {
  const clean = String(value || "")
    .trim()
    .replace(/^https?:\/\/github\.com\//i, "")
    .replace(/^git@github\.com:/i, "")
    .replace(/\.git$/i, "")
    .replace(/\/+$/g, "");
  const [owner, name] = clean.split("/");
  if (!owner || !name) return null;
  return { owner, name, fullName: `${owner}/${name}` };
}

function firstLine(text) {
  return String(text || "").split("\n").find((line) => line.trim())?.trim();
}

function formatDateTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/* ---------- Settings modal ---------- */

settingsBtn.addEventListener("click", openSettings);
settingsClose.addEventListener("click", closeSettings);
settingsModal.addEventListener("click", (e) => {
  if (e.target === settingsModal) closeSettings();
});

async function openSettings() {
  // Populate from current conversation (or sensible defaults).
  const c = currentConversation || {
    system_prompt: "",
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
  };
  settingsSysPrompt.value = c.system_prompt || "";
  settingsTemp.value = c.temperature ?? 0.7;
  settingsTopP.value = c.top_p ?? 0.9;
  settingsTopK.value = c.top_k ?? 40;
  updateSliderLabels();
  // If no conversation yet, settings live only in-memory until first message.
  await Promise.all([refreshModelList(), refreshWorkerList()]);
  settingsModal.classList.remove("hidden");
}

function closeSettings() {
  settingsModal.classList.add("hidden");
}

function updateSliderLabels() {
  settingsTempVal.textContent = parseFloat(settingsTemp.value).toFixed(2);
  settingsTopPVal.textContent = parseFloat(settingsTopP.value).toFixed(2);
}

settingsTemp.addEventListener("input", updateSliderLabels);
settingsTopP.addEventListener("input", updateSliderLabels);

// Debounced auto-save for setting fields.
let saveSettingsTimer = null;
function scheduleSettingsSave() {
  clearTimeout(saveSettingsTimer);
  saveSettingsTimer = setTimeout(saveSettings, 350);
}

[settingsSysPrompt, settingsTemp, settingsTopP, settingsTopK].forEach((el) => {
  el.addEventListener("change", scheduleSettingsSave);
  el.addEventListener("input", scheduleSettingsSave);
});

async function saveSettings() {
  const patch = {
    system_prompt: settingsSysPrompt.value,
    temperature: parseFloat(settingsTemp.value),
    top_p: parseFloat(settingsTopP.value),
    top_k: parseInt(settingsTopK.value, 10) || 40,
  };
  if (currentConversationId) {
    try {
      const updated = await apiJson(`/conversations/${currentConversationId}`, {
        method: "PATCH",
        body: JSON.stringify(patch),
      });
      currentConversation = { ...currentConversation, ...updated };
      updateTokenCounter();
    } catch (err) {
      console.error("Failed to save settings:", err);
    }
  } else {
    // Buffer settings for the eventual first-message create.
    currentConversation = { ...(currentConversation || {}), ...patch };
  }
}

/* ---------- Model management ---------- */

async function refreshModelList() {
  modelListEl.textContent = "Loading…";
  try {
    const data = await apiJson("/models");
    const models = data.models ?? [];
    if (!models.length) {
      modelListEl.textContent = "No models installed yet — pull one below.";
      return;
    }
    modelListEl.innerHTML = "";
    for (const m of models) {
      const row = document.createElement("div");
      row.className = "row";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = m.name;
      const size = document.createElement("span");
      size.className = "size";
      size.textContent = formatBytes(m.size || 0);
      const del = document.createElement("button");
      del.textContent = "Delete";
      del.onclick = () => deleteModel(m.name);
      row.append(name, size, del);
      modelListEl.appendChild(row);
    }
  } catch (err) {
    modelListEl.textContent = `Failed to load models: ${err.message}`;
  }
}

async function refreshWorkerList() {
  workerListEl.textContent = "Loading...";
  try {
    const data = await apiJson("/workers");
    const workers = data.workers ?? [];
    if (!workers.length) {
      workerListEl.textContent = "No workers configured.";
      return;
    }

    workerListEl.innerHTML = "";
    if (!data.control_available && data.control_error) {
      const note = document.createElement("div");
      note.className = "worker-note";
      note.textContent = `Control unavailable: ${data.control_error}`;
      workerListEl.appendChild(note);
    }

    for (const worker of workers) {
      const row = document.createElement("div");
      row.className = "row worker-row";

      const info = document.createElement("div");
      info.className = "worker-info";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = worker.name;
      const details = document.createElement("span");
      details.className = "worker-details";
      details.textContent = workerSummary(worker);
      info.append(name, details);

      const state = document.createElement("span");
      state.className = `worker-state ${worker.available ? "available" : "unavailable"}`;
      state.textContent = worker.available ? "available" : "unavailable";

      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.textContent = worker.enabled ? "Turn off" : "Turn on";
      toggle.disabled = !worker.controllable || worker.essential;
      toggle.onclick = () => setWorkerState(worker.name, !worker.enabled, toggle);

      row.append(info, state, toggle);
      workerListEl.appendChild(row);
    }
  } catch (err) {
    workerListEl.textContent = `Failed to load workers: ${err.message}`;
  }
}

function workerSummary(worker) {
  const actual = worker.control?.actual_state || (worker.enabled ? "on" : "off");
  const desired = worker.control?.desired_state || (worker.enabled ? "on" : "off");
  const models = worker.available_models?.length
    ? worker.available_models.join(", ")
    : worker.configured_models?.join(", ") || "no models";
  const resident = worker.loaded_models?.length
    ? `resident: ${worker.loaded_models.map(loadedModelSummary).join(", ")}`
    : "resident: none; loads on demand";
  const active = worker.in_flight ? `${worker.in_flight} active` : "idle";
  return `${actual}/${desired} - ${active} - ${resident} - installed: ${models}`;
}

function loadedModelSummary(model) {
  const name = model.name || model.model || "model";
  if (!model.expires_at) return name;
  const expiresAt = new Date(model.expires_at).getTime();
  if (Number.isNaN(expiresAt)) return name;
  const ms = expiresAt - Date.now();
  if (ms <= 0) return `${name} evicting`;
  const minutes = Math.max(1, Math.round(ms / 60000));
  return `${name} evicts in ${minutes}m`;
}

async function setWorkerState(name, enabled, button) {
  button.disabled = true;
  button.textContent = enabled ? "Turning on..." : "Turning off...";
  try {
    await apiJson(`/workers/${encodeURIComponent(name)}`, {
      method: "PATCH",
      body: JSON.stringify({ enabled }),
    });
    await Promise.all([refreshWorkerList(), refreshHealth(), loadModels()]);
  } catch (err) {
    alert(`Worker update failed: ${err.message}`);
    await refreshWorkerList();
  }
}

function formatBytes(bytes) {
  if (!bytes) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let n = bytes;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n.toFixed(1)} ${units[i]}`;
}

async function deleteModel(name) {
  if (!confirm(`Delete model "${name}"? This frees disk space but you'll need to pull it again to use it.`)) return;
  try {
    const res = await apiFetch(`/models/${encodeURIComponent(name)}`, { method: "DELETE" });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
    await Promise.all([refreshModelList(), loadModels(), refreshHealth()]);
  } catch (err) {
    alert(`Delete failed: ${err.message}`);
  }
}

pullModelBtn.addEventListener("click", pullModel);
pullModelInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    pullModel();
  }
});

async function pullModel() {
  const name = pullModelInput.value.trim();
  if (!name) return;
  pullProgress.classList.remove("hidden");
  pullProgress.textContent = `Pulling ${name}…\n`;
  pullModelBtn.disabled = true;
  try {
    const res = await apiFetch("/models/pull", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error(`Pull failed (${res.status})`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Each Ollama response is a JSON line. Show the most recent status.
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const obj = JSON.parse(line);
          appendPullProgress(obj);
        } catch {
          appendPullProgress({ status: line });
        }
      }
    }
    appendPullProgress({ status: "✓ done" });
    await Promise.all([refreshModelList(), loadModels(), refreshHealth()]);
    pullModelInput.value = "";
  } catch (err) {
    appendPullProgress({ error: err.message });
  } finally {
    pullModelBtn.disabled = false;
  }
}

function appendPullProgress(obj) {
  if (obj.error) {
    pullProgress.textContent += `Error: ${obj.error}\n`;
  } else if (obj.completed && obj.total) {
    const pct = ((obj.completed / obj.total) * 100).toFixed(1);
    pullProgress.textContent += `${obj.status || ""}: ${pct}%\n`;
  } else if (obj.status) {
    pullProgress.textContent += `${obj.status}\n`;
  }
  pullProgress.scrollTop = pullProgress.scrollHeight;
}

/* ---------- Boot ---------- */

init();
