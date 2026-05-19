const API = getApiBaseUrl();

if (window.marked) {
  marked.setOptions({ breaks: true, gfm: true });
}

function getApiBaseUrl() {
  const saved = localStorage.getItem("apiBaseUrl");
  if (saved) return saved.replace(/\/$/, "");

  if (window.location.protocol === "file:") {
    return "http://localhost:8001";
  }
  return "";
}

function escapeHtml(content) {
  return content
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(content) {
  if (!window.marked || !window.DOMPurify) {
    return escapeHtml(content).replaceAll("\n", "<br>");
  }
  return DOMPurify.sanitize(marked.parse(content));
}

function refreshIcons() {
  if (window.lucide) {
    lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
  }
}

function createIcon(name) {
  const icon = document.createElement("i");
  icon.setAttribute("data-lucide", name);
  icon.setAttribute("aria-hidden", "true");
  return icon;
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

let currentUser = null;
let currentConversationId = null;
let messages = [];
let conversations = [];
let isStreaming = false;

const loginScreen = document.getElementById("login-screen");
const appEl = document.getElementById("app");
const loginForm = document.getElementById("login-form");
const usernameInput = document.getElementById("username-input");
const loginError = document.getElementById("login-error");
const sidebarUsername = document.getElementById("sidebar-username");
const logoutBtn = document.getElementById("logout-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const conversationsList = document.getElementById("conversations-list");
const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const sendBtnLabel = sendBtn.querySelector(".btn-label");
const modelSelect = document.getElementById("model-select");
const modelStatus = document.getElementById("model-status");

async function init() {
  refreshIcons();

  const stored = localStorage.getItem("sharedLocalUser") || localStorage.getItem("user");
  if (stored) {
    try {
      currentUser = JSON.parse(stored);
      await loadApp();
    } catch {
      localStorage.removeItem("user");
      localStorage.removeItem("sharedLocalUser");
      showLogin();
    }
  } else {
    showLogin();
  }
}

function showLogin() {
  loginScreen.classList.remove("hidden");
  appEl.classList.add("hidden");
  usernameInput.focus();
}

async function loadApp() {
  loginScreen.classList.add("hidden");
  appEl.classList.remove("hidden");
  sidebarUsername.textContent = currentUser.username;
  await Promise.all([loadModels(), loadConversations()]);
  showEmptyState();
  refreshIcons();
  inputEl.focus();
}

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = usernameInput.value.trim();
  if (!username) return;

  loginError.classList.add("hidden");

  try {
    const res = await fetch(`${API}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username }),
    });
    if (!res.ok) throw new Error((await res.json()).detail);
    currentUser = await res.json();
    localStorage.setItem("user", JSON.stringify(currentUser));
    localStorage.setItem("sharedLocalUser", JSON.stringify(currentUser));
    await loadApp();
  } catch (err) {
    loginError.textContent = err.message || "Unable to sign in";
    loginError.classList.remove("hidden");
  }
});

logoutBtn.addEventListener("click", () => {
  currentUser = null;
  currentConversationId = null;
  messages = [];
  conversations = [];
  localStorage.removeItem("user");
  localStorage.removeItem("sharedLocalUser");
  showLogin();
});

async function loadModels() {
  setModelStatus("Checking Ollama", "neutral");
  modelSelect.innerHTML = "";

  try {
    const res = await fetch(`${API}/models`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    const models = data.models ?? [];

    if (!models.length) {
      modelSelect.append(new Option("No models found", ""));
      setModelStatus("No models", "warning");
      return;
    }

    const sortedModels = [...models].sort((a, b) => {
      const sizeA = a.size ?? Number.MAX_SAFE_INTEGER;
      const sizeB = b.size ?? Number.MAX_SAFE_INTEGER;
      return sizeA - sizeB || a.name.localeCompare(b.name);
    });
    const savedModel = localStorage.getItem("selectedModel");

    for (const model of sortedModels) {
      modelSelect.append(new Option(model.name, model.name));
    }
    const defaultModel = sortedModels.find((model) => model.name === savedModel) ?? sortedModels[0];
    modelSelect.value = defaultModel.name;
    setModelStatus(`${models.length} ready`, "success");
  } catch {
    modelSelect.append(new Option("Ollama unavailable", ""));
    setModelStatus("Offline", "error");
  }
}

function setModelStatus(text, tone) {
  modelStatus.textContent = text;
  modelStatus.dataset.tone = tone;
}

async function loadConversations() {
  try {
    const res = await fetch(`${API}/conversations?user_id=${currentUser.id}`);
    if (!res.ok) throw new Error();
    conversations = await res.json();
  } catch {
    conversations = [];
  }
  renderConversations();
}

function renderConversations() {
  conversationsList.innerHTML = "";

  if (!conversations.length) {
    const empty = document.createElement("p");
    empty.className = "conversation-empty";
    empty.textContent = "No saved chats";
    conversationsList.appendChild(empty);
    return;
  }

  for (const conv of conversations) {
    const item = document.createElement("div");
    item.className = "conv-item" + (conv.id === currentConversationId ? " active" : "");

    const openBtn = document.createElement("button");
    openBtn.className = "conv-open";
    openBtn.type = "button";
    openBtn.onclick = () => selectConversation(conv.id);

    const iconWrap = document.createElement("span");
    iconWrap.className = "conv-icon";
    iconWrap.appendChild(createIcon("message-square"));

    const textWrap = document.createElement("span");
    textWrap.className = "conv-text";

    const title = document.createElement("span");
    title.className = "conv-title";
    title.textContent = conv.title || "Untitled chat";

    const date = document.createElement("span");
    date.className = "conv-date";
    date.textContent = formatDate(conv.updated_at);

    textWrap.append(title, date);
    openBtn.append(iconWrap, textWrap);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "conv-delete";
    deleteBtn.type = "button";
    deleteBtn.title = "Delete conversation";
    deleteBtn.setAttribute("aria-label", `Delete ${conv.title || "conversation"}`);
    deleteBtn.appendChild(createIcon("trash-2"));
    deleteBtn.onclick = (e) => {
      e.stopPropagation();
      deleteConversation(conv.id);
    };

    item.append(openBtn, deleteBtn);
    conversationsList.appendChild(item);
  }

  refreshIcons();
}

async function selectConversation(id) {
  if (isStreaming) return;
  currentConversationId = id;
  renderConversations();

  try {
    const res = await fetch(`${API}/conversations/${id}/messages?user_id=${currentUser.id}`);
    if (!res.ok) throw new Error();
    messages = await res.json();
    renderMessages();
  } catch {}
}

async function deleteConversation(id) {
  if (isStreaming) return;

  try {
    await fetch(`${API}/conversations/${id}?user_id=${currentUser.id}`, { method: "DELETE" });
    conversations = conversations.filter((c) => c.id !== id);
    if (currentConversationId === id) {
      currentConversationId = null;
      messages = [];
      showEmptyState();
    }
    renderConversations();
  } catch {}
}

newChatBtn.addEventListener("click", () => {
  if (isStreaming) return;
  currentConversationId = null;
  messages = [];
  showEmptyState();
  renderConversations();
  inputEl.focus();
});

function showEmptyState() {
  messagesEl.innerHTML = "";

  const el = document.createElement("section");
  el.className = "empty-state";

  const mark = document.createElement("span");
  mark.className = "empty-mark";
  mark.appendChild(createIcon("sparkles"));

  const title = document.createElement("h2");
  title.textContent = "New conversation";

  const prompts = document.createElement("div");
  prompts.className = "prompt-grid";

  for (const prompt of [
    "Explain vector databases",
    "Draft a launch checklist",
    "Write a Python helper",
  ]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "prompt-chip";
    button.textContent = prompt;
    button.onclick = () => {
      inputEl.value = prompt;
      autoResize();
      inputEl.focus();
    };
    prompts.appendChild(button);
  }

  el.append(mark, title, prompts);
  messagesEl.appendChild(el);
  refreshIcons();
}

function renderMessages() {
  messagesEl.innerHTML = "";
  for (const msg of messages) {
    appendMessage(msg.role, msg.content);
  }
  refreshIcons();
}

function appendMessage(role, content = "") {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.appendChild(createIcon(role === "user" ? "user-round" : "sparkles"));

  const body = document.createElement("div");
  body.className = "message-body";

  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? currentUser.username : "Local LLM";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "assistant" && content) {
    bubble.innerHTML = renderMarkdown(content);
  } else {
    bubble.textContent = content;
  }

  body.append(label, bubble);
  wrapper.append(avatar, body);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  refreshIcons();
  return bubble;
}

async function sendMessage() {
  const content = inputEl.value.trim();
  if (!content || isStreaming) return;

  const model = modelSelect.value;
  if (!model) {
    setModelStatus("Choose a model", "warning");
    return;
  }

  const emptyState = messagesEl.querySelector(".empty-state");
  if (emptyState) emptyState.remove();

  inputEl.value = "";
  autoResize();

  messages.push({ role: "user", content });
  appendMessage("user", content);

  const bubble = appendMessage("assistant");
  bubble.classList.add("streaming");
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  bubble.appendChild(cursor);

  setStreaming(true);

  let assistantContent = "";

  try {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages,
        user_id: currentUser.id,
        conversation_id: currentConversationId,
      }),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    if (!res.body) throw new Error("Streaming response unavailable");

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
    bubble.classList.add("error-bubble");
    bubble.textContent = err.message || "Unable to send message";
  } finally {
    cursor.remove();
    bubble.classList.remove("streaming");
    if (assistantContent) {
      messages.push({ role: "assistant", content: assistantContent });
    }
    setStreaming(false);
    inputEl.focus();
    await loadConversations();
  }
}

function setStreaming(active) {
  isStreaming = active;
  sendBtn.disabled = active;
  inputEl.disabled = active;
  sendBtnLabel.textContent = active ? "Sending" : "Send";
  appEl.classList.toggle("is-streaming", active);
}

function autoResize() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
  inputEl.style.overflowY = inputEl.scrollHeight > 180 ? "auto" : "hidden";
}

inputEl.addEventListener("input", autoResize);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
sendBtn.addEventListener("click", sendMessage);
modelSelect.addEventListener("change", () => {
  if (modelSelect.value) {
    localStorage.setItem("selectedModel", modelSelect.value);
  }
});

init();
