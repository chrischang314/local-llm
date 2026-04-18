const API = "";

marked.setOptions({ breaks: true, gfm: true });

function renderMarkdown(content) {
  return DOMPurify.sanitize(marked.parse(content));
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
const modelSelect = document.getElementById("model-select");

// --- Init ---

async function init() {
  const stored = localStorage.getItem("user");
  if (stored) {
    currentUser = JSON.parse(stored);
    await loadApp();
  } else {
    showLogin();
  }
}

function showLogin() {
  loginScreen.classList.remove("hidden");
  appEl.classList.add("hidden");
}

async function loadApp() {
  loginScreen.classList.add("hidden");
  appEl.classList.remove("hidden");
  sidebarUsername.textContent = currentUser.username;
  await Promise.all([loadModels(), loadConversations()]);
  showEmptyState();
}

// --- Auth ---

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = usernameInput.value.trim();
  if (!username) return;

  try {
    const res = await fetch(`${API}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username }),
    });
    if (!res.ok) throw new Error((await res.json()).detail);
    currentUser = await res.json();
    localStorage.setItem("user", JSON.stringify(currentUser));
    loginError.classList.add("hidden");
    await loadApp();
  } catch (err) {
    loginError.textContent = err.message;
    loginError.classList.remove("hidden");
  }
});

logoutBtn.addEventListener("click", () => {
  currentUser = null;
  currentConversationId = null;
  messages = [];
  conversations = [];
  localStorage.removeItem("user");
  showLogin();
});

// --- Models ---

async function loadModels() {
  try {
    const res = await fetch(`${API}/models`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    const models = data.models ?? [];
    modelSelect.innerHTML = models.length
      ? models.map((m) => `<option value="${m.name}">${m.name}</option>`).join("")
      : `<option value="">No models found</option>`;
  } catch {
    modelSelect.innerHTML = `<option value="">Ollama unavailable</option>`;
  }
}

// --- Conversations ---

async function loadConversations() {
  try {
    const res = await fetch(`${API}/conversations?user_id=${currentUser.id}`);
    conversations = await res.json();
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

async function selectConversation(id) {
  if (isStreaming) return;
  currentConversationId = id;
  renderConversations();

  try {
    const res = await fetch(`${API}/conversations/${id}/messages?user_id=${currentUser.id}`);
    messages = await res.json();
    renderMessages();
  } catch {}
}

async function deleteConversation(id) {
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

// --- Messages ---

function showEmptyState() {
  messagesEl.innerHTML = "";
  const el = document.createElement("div");
  el.className = "empty-state";
  el.textContent = "Start a new conversation";
  messagesEl.appendChild(el);
}

function renderMessages() {
  messagesEl.innerHTML = "";
  for (const msg of messages) {
    appendMessage(msg.role, msg.content);
  }
}

function appendMessage(role, content = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant" && content) {
    bubble.innerHTML = renderMarkdown(content);
  } else {
    bubble.textContent = content;
  }
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

// --- Chat ---

async function sendMessage() {
  const content = inputEl.value.trim();
  if (!content || isStreaming) return;

  const model = modelSelect.value;
  if (!model) {
    alert("No model selected. Make sure Ollama is running and has models pulled.");
    return;
  }

  // Clear empty state on first message
  const emptyState = messagesEl.querySelector(".empty-state");
  if (emptyState) emptyState.remove();

  inputEl.value = "";
  autoResize();

  messages.push({ role: "user", content });
  appendMessage("user", content);

  const bubble = appendMessage("assistant");
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

    const returnedId = res.headers.get("X-Conversation-Id");
    if (returnedId) currentConversationId = parseInt(returnedId);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      assistantContent += decoder.decode(value);
      bubble.innerHTML = renderMarkdown(assistantContent);
      bubble.appendChild(cursor);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  } catch (err) {
    bubble.textContent = `Error: ${err.message}`;
  } finally {
    cursor.remove();
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
}

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

init();
