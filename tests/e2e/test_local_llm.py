"""
End-to-end Playwright tests for the local-llm chat app.
Covers: login, model load, new chat, streaming response,
        conversation persistence, conversation delete.
"""
import re
import time
from playwright.sync_api import sync_playwright, expect

BASE = "http://local-llm.lan:8080"
TIMEOUT = 120_000  # 2 min — LLM responses can be slow


def run():
    results = []

    def ok(name):
        results.append(("PASS", name))
        print(f"  PASS  {name}")

    def fail(name, reason):
        results.append(("FAIL", name, reason))
        print(f"  FAIL  {name}: {reason}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.set_default_timeout(TIMEOUT)

        # ── 1. Login screen loads ──────────────────────────────────────────
        try:
            page.goto(BASE)
            expect(page.locator("#login-screen")).to_be_visible()
            expect(page.locator("#app")).to_have_class(re.compile(r"hidden"))
            ok("Login screen renders")
        except Exception as e:
            fail("Login screen renders", e)

        # ── 2. Login with a username ───────────────────────────────────────
        try:
            page.locator("#username-input").fill("playwright-test")
            page.locator("#login-form button").click()
            expect(page.locator("#app")).not_to_have_class(re.compile(r"hidden"), timeout=10_000)
            expect(page.locator("#login-screen")).not_to_be_visible()
            ok("Login succeeds and app appears")
        except Exception as e:
            fail("Login succeeds and app appears", e)

        # ── 3. Username shown in sidebar ───────────────────────────────────
        try:
            expect(page.locator("#sidebar-username")).to_have_text("playwright-test")
            ok("Username shown in sidebar")
        except Exception as e:
            fail("Username shown in sidebar", e)

        # ── 4. Model dropdown populated ────────────────────────────────────
        try:
            model_select = page.locator("#model-select")
            # Wait until "Loading models…" is replaced
            expect(model_select).not_to_have_text("Loading models...", timeout=15_000)
            options = model_select.locator("option").all()
            assert len(options) > 0 and options[0].get_attribute("value") != ""
            model_name = options[0].inner_text()
            ok(f"Model dropdown populated ({model_name})")
        except Exception as e:
            fail("Model dropdown populated", e)

        # ── 5. New Chat button shows empty state ──────────────────────────
        try:
            page.locator("#new-chat-btn").click()
            # showEmptyState() adds a .empty-state div; #messages is never
            # truly empty — check for the placeholder div instead.
            expect(page.locator("#messages .empty-state")).to_be_visible(timeout=5_000)
            expect(page.locator(".message")).to_have_count(0)
            ok("New Chat shows empty state and clears messages")
        except Exception as e:
            fail("New Chat shows empty state and clears messages", e)

        # ── 6. Send a message and receive a streaming response ─────────────
        try:
            page.locator("#input").fill("Reply with exactly the word: HELLO")
            page.locator("#send-btn").click()

            # User bubble appears
            expect(page.locator(".message.user")).to_be_visible(timeout=10_000)
            ok("User message bubble appears")

            # Assistant streams a response (wait up to 2 min)
            assistant = page.locator(".message.assistant")
            expect(assistant).to_be_visible(timeout=TIMEOUT)

            # Poll until the text is non-empty and not a spinner/placeholder
            deadline = time.time() + 120
            response_text = ""
            while time.time() < deadline:
                response_text = assistant.inner_text().strip()
                if response_text and len(response_text) > 1:
                    break
                time.sleep(1)

            assert response_text, "Assistant response was empty"
            ok(f"Streaming response received (starts: {response_text[:60]!r})")
        except Exception as e:
            fail("Streaming response received", e)

        # ── 7. Conversation saved in sidebar ───────────────────────────────
        try:
            expect(page.locator("#conversations-list")).not_to_be_empty(timeout=10_000)
            conv_items = page.locator("#conversations-list *").all()
            assert len(conv_items) > 0
            ok("Conversation appears in sidebar history")
        except Exception as e:
            fail("Conversation appears in sidebar history", e)

        # ── 8. Conversation persists after re-login ────────────────────────
        try:
            page.locator("#logout-btn").click()
            expect(page.locator("#login-screen")).to_be_visible(timeout=10_000)

            page.locator("#username-input").fill("playwright-test")
            page.locator("#login-form button").click()
            expect(page.locator("#app")).not_to_have_class(re.compile(r"hidden"), timeout=10_000)

            expect(page.locator("#conversations-list")).not_to_be_empty(timeout=10_000)
            ok("Conversation persists after logout/re-login")
        except Exception as e:
            fail("Conversation persists after logout/re-login", e)

        # ── 9. Opening a past conversation loads messages ──────────────────
        try:
            # Use .conv-item to avoid matching child spans with :first-child
            first_conv = page.locator(".conv-item").first
            first_conv.click()
            expect(page.locator(".message.user")).to_be_visible(timeout=10_000)
            expect(page.locator(".message.assistant")).to_be_visible(timeout=10_000)
            ok("Opening past conversation loads messages")
        except Exception as e:
            fail("Opening past conversation loads messages", e)

        # ── 10. Delete conversation (hover-reveal button) ──────────────────
        try:
            item = page.locator(".conv-item").first
            before_count = page.locator(".conv-item").count()
            # Hover to reveal the CSS display:none delete button
            item.hover()
            del_btn = item.locator(".conv-delete")
            expect(del_btn).to_be_visible(timeout=3_000)
            del_btn.click()
            # Conversation should disappear from the list
            expect(page.locator(".conv-item")).to_have_count(before_count - 1, timeout=5_000)
            ok("Delete conversation (hover-reveal) removes it from the list")
        except Exception as e:
            fail("Delete conversation", e)

        browser.close()

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] == "FAIL")
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
    if failed:
        print()
        print("Failures:")
        for r in results:
            if r[0] == "FAIL":
                print(f"  • {r[1]}: {r[2]}")
    print("=" * 55)
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
