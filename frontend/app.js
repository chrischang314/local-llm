/* eslint-env browser */
/*
 * Local LLM Chat — frontend
 *
 * Talks to the FastAPI backend. The shared HttpOnly session cookie is
 * attached by the browser. Conversation settings (model, system prompt,
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
const AUTH_DISPLAY_KEY = "authDisplay";

marked.setOptions({ breaks: true, gfm: true });

/* ---------- App state ---------- */

let currentUser = null;
let currentConversation = null; // full row from backend
let currentConversationId = null;
let messages = []; // [{id?, role, content}]
let conversations = [];
let isStreaming = false;
let streamAbortController = null;
let loginMode = "login"; // or "register"
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
const exportConversationBtn = $("export-conversation-btn");
const healthIndicator = $("health-indicator");
const researchToggle = $("research-toggle");

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
const workerReadinessPanel = $("worker-readiness-panel");
const pullModelInput = $("pull-model-name");
const pullModelBtn = $("pull-model-btn");
const pullProgress = $("pull-progress");

function refreshIcons() {
  if (window.lucide) lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

/* ---------- HTTP helpers ---------- */

function requestHeaders(extra = {}) {
  return { ...extra };
}

async function apiFetch(path, opts = {}) {
  const headers = { ...(opts.headers || {}), ...requestHeaders() };
  if (opts.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const res = await fetch(`${API}${path}`, { ...opts, headers, credentials: "include" });
  if (res.status === 401 && currentUser) {
    // Session expired or invalid; drop display state and bounce to login.
    clearLocalSession();
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

function cacheDisplayUser(user) {
  localStorage.setItem(AUTH_DISPLAY_KEY, JSON.stringify({
    id: user.id,
    username: user.username,
  }));
}

function readCachedDisplayUser() {
  try {
    const parsed = JSON.parse(localStorage.getItem(AUTH_DISPLAY_KEY) || "null");
    return parsed?.username ? { id: parsed.id, username: parsed.username } : null;
  } catch {
    return null;
  }
}

/* ---------- Init ---------- */

async function init() {
  localStorage.removeItem("auth");
  currentUser = readCachedDisplayUser();
  try {
    const data = await apiJson("/auth/me");
    currentUser = { id: data.id, username: data.username };
    cacheDisplayUser(currentUser);
    await loadApp();
    return;
  } catch {
    currentUser = null;
  }
  showLogin();
}

function showLogin() {
  loginScreen.classList.remove("hidden");
  appEl.classList.add("hidden");
}

async function loadApp() {
  loginScreen.classList.add("hidden");
  appEl.classList.remove("hidden");
  sidebarUsername.textContent = currentUser.username;
  await Promise.all([
    loadModels(),
    loadConversations(),
    refreshHealth(),
    refreshResearchStatus(),
  ]);
  showEmptyState();
  updateExportButtonState();
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
      credentials: "include",
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Login failed");
    const data = await res.json();
    currentUser = { id: data.id, username: data.username };
    localStorage.removeItem("auth");
    cacheDisplayUser(currentUser);
    loginError.classList.add("hidden");
    passwordInput.value = "";
    await loadApp();
  } catch (err) {
    loginError.textContent = err.message;
    loginError.classList.remove("hidden");
  }
});

function clearLocalSession() {
  stopAgentEventStream();
  currentUser = null;
  currentConversation = null;
  currentConversationId = null;
  messages = [];
  conversations = [];
  webResearchEnabled = false;
  localStorage.removeItem("auth");
  localStorage.removeItem(AUTH_DISPLAY_KEY);
  // Reset to login mode (user is more likely returning than registering).
  if (loginMode !== "login") toggleMode.click();
  passwordInput.value = "";
  showLogin();
}

async function handleLogout() {
  try {
    await fetch(`${API}/auth/logout`, {
      method: "POST",
      credentials: "include",
    });
  } catch {}
  clearLocalSession();
}

logoutBtn.addEventListener("click", () => handleLogout());

/* ---------- Health ---------- */

async function refreshHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    const status = LocalLlmHealthStatus.status(data);
    healthIndicator.classList.remove("ok", "warning", "down");
    healthIndicator.classList.add(status.className);
    healthIndicator.querySelector(".label").textContent = status.label;
    healthIndicator.title = status.label;
  } catch {
    healthIndicator.classList.remove("ok", "warning");
    healthIndicator.classList.add("down");
    healthIndicator.querySelector(".label").textContent = "Backend unreachable";
    healthIndicator.title = "Backend unreachable";
  }
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
  const active = webResearchEnabled && webResearchAvailable;
  researchToggle.classList.toggle("active", active);
  researchToggle.disabled = !webResearchAvailable || isStreaming;
  researchToggle.setAttribute("aria-pressed", String(active));
  researchToggle.title = webResearchAvailable
    ? "Use web research for the next message"
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
    populateModelSelect(modelSelect, models);
    // Restore the conversation's saved model if applicable.
    if (currentConversation?.model) modelSelect.value = currentConversation.model;
    return models;
  } catch {
    modelSelect.innerHTML = `<option value="">Ollama unavailable</option>`;
    return [];
  }
}

function populateModelSelect(selectEl, models) {
  if (!selectEl) return;
  const previous = selectEl.value;
  selectEl.replaceChildren();
  if (models.length) {
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.name;
      selectEl.appendChild(option);
    }
    if (previous && models.some((m) => m.name === previous)) selectEl.value = previous;
  } else {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No models found";
    selectEl.appendChild(option);
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

function updateExportButtonState() {
  if (!exportConversationBtn) return;
  const canExport =
    !isStreaming &&
    !!currentConversationId &&
    messages.length > 0;
  exportConversationBtn.disabled = !canExport;
  exportConversationBtn.title = canExport
    ? "Export active conversation as Markdown"
    : "Open a saved conversation to export it";
}

async function exportCurrentConversation(format = "markdown") {
  if (!currentConversationId || isStreaming) return;
  exportConversationBtn.disabled = true;
  try {
    const res = await apiFetch(
      `/conversations/${currentConversationId}/export?format=${encodeURIComponent(format)}`
    );
    if (!res.ok) {
      let detail = `Export failed (${res.status})`;
      try { detail = (await res.json()).detail || detail; } catch {}
      throw new Error(detail);
    }
    const blob = await res.blob();
    const fallback = format === "json" ? "local-llm-conversation.json" : "local-llm-conversation.md";
    downloadBlob(blob, filenameFromContentDisposition(res.headers.get("Content-Disposition")) || fallback);
  } catch (err) {
    alert(`Export failed: ${err.message}`);
  } finally {
    updateExportButtonState();
  }
}

function filenameFromContentDisposition(value) {
  if (!value) return null;
  const match = value.match(/filename="?([^"]+)"?/i);
  return match?.[1] || null;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

exportConversationBtn?.addEventListener("click", () => exportCurrentConversation("markdown"));

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
    updateExportButtonState();
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
      updateExportButtonState();
    }
    renderConversations();
  } catch {}
}

newChatBtn.addEventListener("click", () => {
  if (isStreaming) return;
  currentConversationId = null;
  currentConversation = null;
  messages = [];
  chatTitle.textContent = "New Chat";
  showEmptyState();
  renderConversations();
  updateTokenCounter();
  updateExportButtonState();
  inputEl.focus();
});

/* ---------- Messages rendering ---------- */

function showEmptyState() {
  messagesEl.innerHTML = "";
  const el = document.createElement("div");
  el.className = "empty-state";
  el.textContent = "Start a new conversation";
  messagesEl.appendChild(el);
  updateExportButtonState();
}

function renderMessages() {
  messagesEl.innerHTML = "";
  messages.forEach((msg, idx) => {
    appendMessage(msg.role, msg.content, {
      id: msg.id,
      index: idx,
      isLast: idx === messages.length - 1,
      route: msg,
    });
  });
  if (messages.length === 0) showEmptyState();
  updateExportButtonState();
}

function renderMarkdown(content) {
  return DOMPurify.sanitize(marked.parse(content));
}

// Enhance a freshly-rendered assistant bubble so rich Markdown stays usable
// inside the chat column on narrow and wide screens.
function enhanceRenderedContent(bubble, { highlightCode = true } = {}) {
  wrapMarkdownTables(bubble);
  enhanceRenderedMedia(bubble);
  enhanceRenderedLinks(bubble);
  if (highlightCode) enhanceCodeBlocks(bubble);
}

function wrapMarkdownTables(bubble) {
  bubble.querySelectorAll("table").forEach((table) => {
    if (table.parentElement?.classList.contains("markdown-table-scroll")) return;
    const wrapper = document.createElement("div");
    wrapper.className = "markdown-table-scroll";
    wrapper.tabIndex = 0;
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-label", "Scrollable table");
    table.before(wrapper);
    wrapper.appendChild(table);
  });
}

function enhanceRenderedMedia(bubble) {
  bubble.querySelectorAll("img").forEach((img) => {
    img.loading = "lazy";
    img.decoding = "async";
  });
  bubble.querySelectorAll("video").forEach((video) => {
    if (!video.hasAttribute("controls")) video.setAttribute("controls", "");
  });
}

function enhanceRenderedLinks(bubble) {
  bubble.querySelectorAll("a[href]").forEach((link) => {
    link.target = "_blank";
    link.rel = "noreferrer";
  });
}

// Syntax-highlight code and inject a copy button into each <pre>.
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

function appendMessage(role, content = "", { id, index, skipActions, isLast, route } = {}) {
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
    enhanceRenderedContent(bubble);
  } else {
    bubble.textContent = content;
  }
  if (role === "assistant") appendMessageRoute(wrapper, route);
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

function appendMessageRoute(wrapper, route) {
  const label = formatMessageRoute(route);
  if (!label) return;
  const el = document.createElement("div");
  el.className = "message-route";
  el.textContent = label;
  el.title = `Response route: ${label}`;
  wrapper.appendChild(el);
}

function setMessageRoute(bubble, route) {
  const wrapper = bubble?.parentElement;
  if (!wrapper) return;
  const label = formatMessageRoute(route);
  if (!label) return;
  let el = Array.from(wrapper.children).find((child) => child.classList.contains("message-route"));
  if (!el) {
    el = document.createElement("div");
    el.className = "message-route";
    wrapper.insertBefore(el, bubble);
  }
  el.textContent = label;
  el.title = `Response route: ${label}`;
}

function formatMessageRoute(route = {}) {
  const backendName = route.backend_name || route.backendName;
  const modelName = route.model;
  const status = readableModelStatus(route.model_status || route.modelStatus);
  if (!backendName && !modelName && !status) return "";

  const parts = [];
  if (backendName) parts.push(`via ${backendName}`);
  if (modelName) parts.push(modelName);
  if (status) parts.push(status);
  return parts.join(" - ");
}

function readableModelStatus(status) {
  if (!status) return "";
  if (status === "resident") return "resident";
  if (status === "loading") return "loaded on demand";
  return String(status).replaceAll("_", " ");
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
  let assistantRoute = null;

  try {
    const body = {
      messages: messages.map(({ role, content }) => ({ role, content })),
      conversation_id: currentConversationId,
      regenerate,
    };
    // Settings only ship to the backend for the first message of a new
    // conversation; thereafter the backend reads them from the DB row.
    if (!currentConversationId) {
      body.model = model;
      body.system_prompt = currentConversation?.system_prompt || "";
      body.temperature = currentConversation?.temperature ?? 0.7;
      body.top_p = currentConversation?.top_p ?? 0.9;
      body.top_k = currentConversation?.top_k ?? 40;
    }
    if (webResearchEnabled && webResearchAvailable) {
      body.web_research = true;
      webResearchEnabled = false;
      updateResearchToggle();
    }

    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: requestHeaders({ "Content-Type": "application/json" }),
      credentials: "include",
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
    const researchStatus = res.headers.get("X-Research-Status");
    const researchSourceCount = Number.parseInt(res.headers.get("X-Research-Source-Count") || "0", 10);
    assistantRoute = {
      model,
      backend_name: backendName,
      model_status: modelStatus || (res.headers.get("X-LLM-Model-Loaded") === "true" ? "resident" : ""),
    };
    setMessageRoute(bubble, assistantRoute);
    if (researchStatus && researchStatus !== "not_requested") {
      statusEl = setStreamStatus(bubble, cursor, statusEl, researchStatusText(researchStatus, researchSourceCount));
    } else if (modelStatus === "loading") {
      statusEl = setStreamStatus(bubble, cursor, statusEl, `Loading ${model} on ${backendName || "worker"}...`);
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
      enhanceRenderedContent(bubble, { highlightCode: false });
      bubble.appendChild(cursor);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    const tail = decoder.decode();
    if (tail) {
      assistantContent += tail;
      bubble.innerHTML = renderMarkdown(assistantContent);
      enhanceRenderedContent(bubble, { highlightCode: false });
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
    if (assistantContent) {
      enhanceRenderedContent(bubble);
      messages.push({ role: "assistant", content: assistantContent, ...(assistantRoute || {}) });
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
  stopBtn.classList.toggle("hidden", !active);
  sendBtn.classList.toggle("hidden", active);
  updateResearchToggle();
  updateExportButtonState();
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
  renderWorkerReadiness({
    severity: "warning",
    state: "checking",
    summary: "Checking worker readiness...",
    issues: [],
  });
  try {
    const data = await apiJson("/workers");
    const workers = data.workers ?? [];
    renderWorkerReadiness(data.readiness, data.control_error);
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
    renderWorkerReadiness({
      severity: "error",
      state: "unavailable",
      summary: `Failed to load worker readiness: ${err.message}`,
      issues: [],
    });
  }
}

function renderWorkerReadiness(readiness, fallbackError = "") {
  if (!workerReadinessPanel) return;

  const allowedSeverities = new Set(["ok", "warning", "error"]);
  const severity = allowedSeverities.has(readiness?.severity)
    ? readiness.severity
    : "warning";
  const state = readiness?.state
    ? readiness.state.replace(/_/g, " ")
    : severity;
  const summary = readiness?.summary
    || (fallbackError ? `Worker control unavailable: ${fallbackError}` : "Worker readiness unavailable.");
  const issues = Array.isArray(readiness?.issues) ? readiness.issues : [];

  workerReadinessPanel.className = `worker-readiness-panel ${severity}`;
  workerReadinessPanel.innerHTML = "";

  const summaryRow = document.createElement("div");
  summaryRow.className = "worker-readiness-summary";
  const badge = document.createElement("span");
  badge.className = "worker-readiness-badge";
  badge.textContent = state;
  const summaryText = document.createElement("span");
  summaryText.textContent = summary;
  summaryRow.append(badge, summaryText);
  workerReadinessPanel.appendChild(summaryRow);

  if (!issues.length) return;

  const issueList = document.createElement("div");
  issueList.className = "worker-readiness-issues";
  for (const issue of issues.slice(0, 4)) {
    const issueEl = document.createElement("div");
    issueEl.className = `worker-readiness-issue ${issue.severity || "warning"}`;

    const message = document.createElement("strong");
    message.textContent = issue.message || issue.type || "Worker readiness issue";
    issueEl.appendChild(message);

    if (issue.next_check) {
      const nextCheck = document.createElement("span");
      nextCheck.textContent = issue.next_check;
      issueEl.appendChild(nextCheck);
    }

    issueList.appendChild(issueEl);
  }
  if (issues.length > 4) {
    const more = document.createElement("div");
    more.className = "worker-readiness-more";
    more.textContent = `${issues.length - 4} more readiness issues`;
    issueList.appendChild(more);
  }
  workerReadinessPanel.appendChild(issueList);
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
