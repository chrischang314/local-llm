import json
import pathlib
import sys
import unittest

import httpx


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from research_client import (  # noqa: E402
    ResearchResult,
    ResearchSource,
    WebResearchConfig,
    build_research_context,
    build_sources_footer,
    research_status,
    research_web,
)
import main  # noqa: E402


class WebResearchClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_provider_extracts_sources_and_page_excerpt(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "www.mojeek.com":
                return httpx.Response(
                    200,
                    text="""
                    <html><body>
                      <ul class="results-standard">
                        <li>
                          <h2>
                            <a class="title" href="https://93.184.216.34/fresh-news">
                              Fresh result
                            </a>
                          </h2>
                          <p class="s">Published today with current details.</p>
                        </li>
                      </ul>
                    </body></html>
                    """,
                    headers={"content-type": "text/html"},
                )
            if request.url.host == "93.184.216.34":
                return httpx.Response(
                    200,
                    text="""
                    <html><body>
                      <script>ignore me</script>
                      <main>Fresh facts from the page body. Updated June 2026.</main>
                    </body></html>
                    """,
                    headers={"content-type": "text/html"},
                )
            return httpx.Response(404)

        result = await research_web(
            "fresh news",
            config=WebResearchConfig(max_results=1, timeout_seconds=2, fetch_pages=True),
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].title, "Fresh result")
        self.assertEqual(result.sources[0].url, "https://93.184.216.34/fresh-news")
        self.assertIn("current details", result.sources[0].snippet)
        self.assertIn("Updated June 2026", result.sources[0].excerpt)

    async def test_default_provider_does_not_fetch_result_pages(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "www.mojeek.com":
                return httpx.Response(
                    200,
                    text="""
                    <html><body>
                      <a class="title" href="https://93.184.216.34/article">Result</a>
                      <p class="s">Search snippet only.</p>
                    </body></html>
                    """,
                    headers={"content-type": "text/html"},
                )
            raise AssertionError(f"unexpected page fetch to {request.url}")

        result = await research_web(
            "snippet result",
            config=WebResearchConfig(max_results=1, timeout_seconds=2),
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.sources[0].snippet, "Search snippet only.")
        self.assertEqual(result.sources[0].excerpt, "")

    async def test_research_degrades_to_error_without_raising(self):
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="search unavailable")

        result = await research_web(
            "latest facts",
            config=WebResearchConfig(timeout_seconds=2),
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.sources, [])
        self.assertIsNotNone(result.error)

    async def test_page_fetch_does_not_follow_redirects_to_private_hosts(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "www.mojeek.com":
                return httpx.Response(
                    200,
                    text="""
                    <html><body>
                      <a class="title" href="https://93.184.216.34/redirect">Result</a>
                      <p class="s">Search snippet survives without page text.</p>
                    </body></html>
                    """,
                    headers={"content-type": "text/html"},
                )
            if request.url.host == "93.184.216.34":
                return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
            raise AssertionError(f"unexpected fetch to {request.url}")

        result = await research_web(
            "redirecting result",
            config=WebResearchConfig(max_results=1, timeout_seconds=2, fetch_pages=True),
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.sources[0].snippet, "Search snippet survives without page text.")
        self.assertEqual(result.sources[0].excerpt, "")

    async def test_private_literal_ip_results_are_skipped(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "www.mojeek.com")
            return httpx.Response(
                200,
                text="""
                <html><body>
                  <a class="title" href="http://127.0.0.1/private">Private result</a>
                  <p class="s">Should not be included.</p>
                </body></html>
                """,
                headers={"content-type": "text/html"},
            )

        result = await research_web(
            "private result",
            config=WebResearchConfig(max_results=1, timeout_seconds=2),
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.sources, [])

    async def test_research_can_be_disabled_by_config(self):
        result = await research_web(
            "anything",
            config=WebResearchConfig(enabled=False),
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )

        self.assertEqual(result.status, "disabled")
        self.assertEqual(result.sources, [])

    def test_context_includes_source_urls_and_current_date_instruction(self):
        result = ResearchResult(
            query="current model news",
            status="ok",
            sources=[
                ResearchSource(
                    title="Model release notes",
                    url="https://example.com/release",
                    snippet="A new model shipped.",
                    excerpt="The release notes say the model was updated this month.",
                )
            ],
        )
        context = build_research_context(
            result,
            config=WebResearchConfig(max_context_chars=4000),
        )

        self.assertIn("Current date:", context)
        self.assertIn("https://example.com/release", context)
        self.assertIn("full source URLs", context)

        footer = build_sources_footer(result)
        self.assertIn("Sources:", footer)
        self.assertIn("- Model release notes: https://example.com/release", footer)

    def test_status_omits_secret_fields(self):
        status = research_status(WebResearchConfig(enabled=True, max_results=2, timeout_seconds=3))

        self.assertEqual(status["provider"], "mojeek")
        self.assertEqual(status["max_results"], 2)
        self.assertFalse(status["fetch_pages"])
        self.assertNotIn("api_key", status)
        self.assertNotIn("token", status)


class WebResearchPromptTests(unittest.TestCase):
    def test_research_context_is_inserted_after_existing_system_prompt(self):
        messages = [
            {"role": "system", "content": "Follow the user's style."},
            {"role": "user", "content": "What changed today?"},
        ]

        augmented = main._insert_research_context(messages, "Research context")

        self.assertEqual(augmented[0]["content"], "Follow the user's style.")
        self.assertEqual(augmented[1], {"role": "system", "content": "Research context"})
        self.assertEqual(augmented[2]["role"], "user")

    def test_latest_user_message_uses_most_recent_user_turn(self):
        messages = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest question"},
        ]

        self.assertEqual(main._latest_user_message(messages), "latest question")


class WebResearchApiStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_v1_streaming_emits_sources_before_terminal_chunk(self):
        class FakeBackend:
            name = "fake-backend"
            url = "http://ollama.local"

        class FakeTrackRequest:
            async def __aenter__(self):
                return None

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeStreamResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                yield json.dumps({"message": {"content": "Fresh answer."}, "done": False})
                yield json.dumps({"done": True})

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, *args, **kwargs):
                return FakeStreamResponse()

        async def fake_choose_backend(model):
            return FakeBackend()

        async def fake_runtime_status(backend, model):
            return {"loaded": True}

        def fake_track_request(backend):
            return FakeTrackRequest()

        async def fake_research_web(query):
            return ResearchResult(
                query=query,
                status="ok",
                sources=[ResearchSource(title="Fresh source", url="https://example.org/fresh")],
            )

        original_choose_backend = main.ollama_router.choose_backend
        original_runtime_status = main.ollama_router.model_runtime_status
        original_track_request = main.ollama_router.track_request
        original_async_client = main.httpx.AsyncClient
        original_research_web = main.research_web
        main.ollama_router.choose_backend = fake_choose_backend
        main.ollama_router.model_runtime_status = fake_runtime_status
        main.ollama_router.track_request = fake_track_request
        main.httpx.AsyncClient = FakeAsyncClient
        main.research_web = fake_research_web
        try:
            response = await main.v1_chat_completions(
                main.ApiChatRequest(
                    model="llama3.2:1b",
                    messages=[main.ApiChatMessage(role="user", content="latest news")],
                    stream=True,
                    web_research=True,
                )
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
            payload = "".join(chunks)
        finally:
            main.ollama_router.choose_backend = original_choose_backend
            main.ollama_router.model_runtime_status = original_runtime_status
            main.ollama_router.track_request = original_track_request
            main.httpx.AsyncClient = original_async_client
            main.research_web = original_research_web

        self.assertIn("Sources:", payload)
        self.assertLess(payload.index("Sources:"), payload.index('"finish_reason": "stop"'))
        self.assertIn("data: [DONE]", payload)


if __name__ == "__main__":
    unittest.main()
