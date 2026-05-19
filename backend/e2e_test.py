"""End-to-end test of the local-LLM chat app using Playwright.

Drives a real Chromium browser against the dev frontend (http://localhost:8080)
which talks to the FastAPI backend (http://localhost:8000) and a real
Ollama on 11434. The test exercises every feature implemented in this
change.

Run from the `backend/` directory with:
    .testvenv/Scripts/python.exe e2e_test.py
"""

from __future__ import annotations

import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    expect,
    sync_playwright,
)

FRONTEND = "http://localhost:8080/"
BACKEND = "http://localhost:8000"
MODEL = "llama3.2:latest"

# Use a fresh username on each run so we exercise register + login.
USERNAME = f"e2e_{int(time.time())}"
PASSWORD = "hunter22"

results: list[tuple[str, bool, str]] = []


@contextmanager
def step(name: str):
    print(f"\n=== {name}")
    try:
        yield
        results.append((name, True, ""))
        print(f"    [pass] {name}")
    except Exception as e:  # noqa: BLE001
        results.append((name, False, repr(e)))
        print(f"    [FAIL] {name}: {e!r}")
        raise


def wait_for(predicate, timeout_s: float = 30.0, interval_s: float = 0.25):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
            last = value
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(interval_s)
    raise AssertionError(f"wait_for timed out (last value: {last!r})")


def run(p: Playwright) -> None:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.on("console", lambda m: print(f"  [console:{m.type}] {m.text}") if m.type in {"error", "warning"} else None)

    try:
        execute_tests(page)
    finally:
        # Save a screenshot for debugging regardless of outcome.
        try:
            page.screenshot(path="/tmp/e2e_final.png")
            print("\nFinal screenshot saved to /tmp/e2e_final.png")
        except Exception:
            pass
        context.close()
        browser.close()


def execute_tests(page: Page) -> None:
    page.goto(FRONTEND)
    expect(page.locator("#login-screen")).to_be_visible()

    # --- 1. Register flow (toggles to register mode, creates a user) ---
    with step("Register a new user"):
        page.click("#toggle-mode")
        expect(page.locator("#login-submit")).to_have_text("Register")
        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)
        page.click("#login-submit")
        expect(page.locator("#app")).to_be_visible(timeout=10_000)
        expect(page.locator("#sidebar-username")).to_have_text(USERNAME)

    # --- 2. Health indicator reflects backend + Ollama state ---
    with step("Health indicator shows Ollama as ready"):
        # Wait for the indicator to flip from 'checking…' to OK.
        wait_for(lambda: page.locator("#health-indicator.ok").count() > 0)
        label = page.locator("#health-indicator .label").inner_text()
        assert "Ollama ready" in label, f"unexpected label: {label}"

    # --- 3. Model dropdown is populated ---
    with step("Model dropdown lists Ollama models"):
        wait_for(lambda: page.locator(f'#model-select option[value="{MODEL}"]').count() > 0)
        page.select_option("#model-select", MODEL)

    # --- 4. Settings modal opens, sliders work, persist ---
    with step("Open settings modal, tweak system prompt + temperature"):
        page.click("#settings-btn")
        expect(page.locator("#settings-modal")).to_be_visible()
        page.fill("#settings-system-prompt", "Always respond in haiku (5-7-5).")
        page.locator("#settings-temperature").evaluate(
            "(el) => { el.value = '0.30'; el.dispatchEvent(new Event('input')); el.dispatchEvent(new Event('change')); }"
        )
        # Slider label shows new value
        expect(page.locator("#settings-temperature-val")).to_have_text("0.30")
        page.click("#settings-close")
        expect(page.locator("#settings-modal")).to_be_hidden()

    # --- 5. Send first message, conversation gets created & titled ---
    with step("Send a chat message and stream a response"):
        page.fill("#input", "Say only one word: pong")
        page.click("#send-btn")
        # Stop button visible while streaming
        expect(page.locator("#stop-btn")).to_be_visible(timeout=10_000)
        # Wait for streaming to finish (stop hides, send shows)
        expect(page.locator("#stop-btn")).to_be_hidden(timeout=60_000)
        # At least 2 messages on screen
        assert page.locator(".message").count() >= 2

    # Snapshot conversation id from sidebar for later assertions.
    sidebar_titles = page.locator(".conv-item .conv-title")
    assert sidebar_titles.count() >= 1, "sidebar should list the new conversation"
    first_title = sidebar_titles.first.inner_text()
    print(f"    conversation titled: {first_title!r}")

    # --- 6. Syntax highlighting + copy button appear on a code block ---
    with step("Assistant message with code renders highlight.js + copy button"):
        page.fill("#input", "Reply with just this exact code block in Python:\n```python\nprint('hi')\n```")
        page.click("#send-btn")
        expect(page.locator("#stop-btn")).to_be_hidden(timeout=60_000)
        # Wait for code to appear and be enhanced
        wait_for(lambda: page.locator(".message.assistant .bubble pre code").count() > 0, timeout_s=10)
        # highlight.js sets class names like "hljs language-python"
        code_class = page.locator(".message.assistant .bubble pre code").last.get_attribute("class") or ""
        assert "hljs" in code_class, f"highlight.js not applied; class={code_class!r}"
        # Copy button injected
        assert page.locator(".message.assistant .bubble pre .code-copy-btn").count() > 0

    # --- 7. Stop generation button aborts mid-stream ---
    with step("Stop button aborts a long generation"):
        page.fill("#input", "Write a 500-word essay about the history of toast")
        page.click("#send-btn")
        expect(page.locator("#stop-btn")).to_be_visible(timeout=10_000)
        # Wait for some content to start flowing in
        time.sleep(2)
        page.click("#stop-btn")
        # Streaming should end (stop hides) within a couple seconds
        expect(page.locator("#stop-btn")).to_be_hidden(timeout=10_000)

    # --- 8. Regenerate replaces the last assistant message ---
    with step("Regenerate the last assistant response"):
        page.fill("#input", "Reply with one short sentence about ducks.")
        page.click("#send-btn")
        expect(page.locator("#stop-btn")).to_be_hidden(timeout=60_000)
        # Capture last assistant content
        last_assistant = page.locator(".message.assistant").last.inner_text()
        # Hover the last assistant to reveal action buttons
        page.locator(".message.assistant").last.hover()
        regen = page.locator(".message.assistant").last.locator(".message-actions button", has_text="Regenerate")
        expect(regen).to_be_visible(timeout=5000)
        regen.click()
        expect(page.locator("#stop-btn")).to_be_hidden(timeout=60_000)
        new_last = page.locator(".message.assistant").last.inner_text()
        assert new_last and new_last != "[stopped]", "regenerate produced no content"
        print(f"    pre-regen : {last_assistant[:80]!r}")
        print(f"    post-regen: {new_last[:80]!r}")

    # --- 9. Edit a user message + re-submit ---
    with step("Edit the last user message and resend"):
        # Wait for the last user message to have a message id (i.e. the
        # backend reload has completed after the previous test step).
        wait_for(
            lambda: page.locator(".message.user[data-message-id]").count() > 0,
            timeout_s=10,
        )
        last_user = page.locator(".message.user[data-message-id]").last
        last_user.hover()
        edit_btn = last_user.locator(".message-actions button", has_text="Edit")
        expect(edit_btn).to_be_visible(timeout=5000)
        edit_btn.click()
        textarea = page.locator(".message.user .edit-textarea").last
        expect(textarea).to_be_visible(timeout=5000)
        textarea.fill("Reply with the word OAK only.")
        page.locator(".edit-actions button", has_text="Save & resend").click()
        expect(page.locator("#stop-btn")).to_be_hidden(timeout=60_000)
        # Last user message should now reflect the edit
        wait_for(
            lambda: "OAK"
            in (page.locator(".message.user .bubble").last.inner_text() or "").upper(),
            timeout_s=10,
        )

    # --- 10. Rename a conversation via double-click ---
    with step("Rename conversation via double-click in sidebar"):
        title_el = page.locator(".conv-item.active .conv-title")
        title_el.dblclick()
        # contenteditable should be on
        expect(title_el).to_have_attribute("contenteditable", "true")
        page.keyboard.press("Control+A")
        page.keyboard.type("Renamed-by-e2e")
        page.keyboard.press("Enter")
        # Title should now show new name in sidebar AND header
        wait_for(lambda: "Renamed-by-e2e" in (page.locator(".conv-item.active .conv-title").inner_text() or ""))
        wait_for(lambda: "Renamed-by-e2e" in page.locator("#chat-title").inner_text())

    # --- 11. Token counter visible and reflects message volume ---
    with step("Token counter displays approximate context size"):
        text = page.locator("#token-counter").inner_text()
        assert re.search(r"~\d", text), f"unexpected counter text: {text!r}"
        print(f"    counter shows: {text}")

    # --- 12. Settings modal shows installed models, can pull/delete UI exists ---
    with step("Settings modal lists installed models"):
        page.click("#settings-btn")
        expect(page.locator("#settings-modal")).to_be_visible()
        wait_for(
            lambda: page.locator("#model-list .row").count() > 0
            or "No models installed" in page.locator("#model-list").inner_text()
        )
        # The current model should be listed
        names = page.locator("#model-list .row .name").all_inner_texts()
        assert any(MODEL in n for n in names), f"expected {MODEL} in {names}"
        page.click("#settings-close")

    # --- 13. Logout clears auth and bounces back to login ---
    with step("Logout drops back to the login screen"):
        page.click("#logout-btn")
        expect(page.locator("#login-screen")).to_be_visible()
        # localStorage cleared
        stored = page.evaluate("() => localStorage.getItem('auth')")
        assert stored is None

    # --- 14. Re-login with the same credentials ---
    with step("Login again with the registered user"):
        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)
        page.click("#login-submit")
        expect(page.locator("#app")).to_be_visible(timeout=10_000)
        # The previously renamed conversation should still be there
        wait_for(lambda: any(
            "Renamed-by-e2e" in t
            for t in page.locator(".conv-item .conv-title").all_inner_texts()
        ))


def main() -> int:
    print(f"Running e2e tests as user {USERNAME!r}")
    with sync_playwright() as p:
        try:
            run(p)
        except Exception as e:  # noqa: BLE001
            print(f"\nFATAL: {e}")
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = [name for name, ok, _ in results if not ok]
    print(f"Results: {passed}/{len(results)} passed")
    for name, ok, err in results:
        mark = "[pass]" if ok else "[FAIL]"
        print(f"  {mark} {name}{' -- ' + err if err else ''}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
