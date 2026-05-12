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

/* ---------- Init ---------- */

async function init() {
  const stored = localStorage.getItem("auth");
  if (stored) {
    try {
      const parsed = JSON.parse(stored);
      authToken = parsed.token;
      currentUser = { id: parsed.id, username: parsed.username };
      await loadApp();
      return;
    } catch {}
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
  await Promise.all([loadModels(), loadConversations(), refreshHealth()]);
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

/* ---------- Health ---------- */

async function refreshHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    const ok = data.ollama === "ok";
    healthIndicator.classList.toggle("ok", ok);
    healthIndicator.classList.toggle("down", !ok);
    healthIndicator.querySelector(".label").textContent = ok
      ? `Ollama ready · ${data.model_count} model${data.model_count === 1 ? "" : "s"}`
      : "Ollama unreachable";
  } catch {
    healthIndicator.classList.remove("ok");
    healthIndicator.classList.add("down");
    healthIndicator.querySelector(".label").textContent = "Backend unreachable";
  }
}

/* ---------- Models ---------- */

async function loadModels() {
  try {
    const data = await apiJson("/models");
    const models = data.models ?? [];
    modelSelect.innerHTML = models.length
      ? models.map((m) => `<option value="${m.name}">${m.name}</option>`).join("")
      : `<option value="">No models found</option>`;
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

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
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
    cursor.remove();
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
  await refreshModelList();
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
