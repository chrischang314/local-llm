"""Small web-research client for request-time LLM augmentation.

The client intentionally has no database dependency and no API-key requirement.
It uses a simple public search-results page, fetches a few public result pages,
and returns compact excerpts that can be injected into the model context.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx


USER_AGENT = (
    "LocalLLMResearch/1.0 "
    "(compatible; request-time retrieval for a private local LLM app)"
)
DDG_SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"
MOJEEK_SEARCH_URL = "https://www.mojeek.com/search?q={query}"
MAX_PAGE_BYTES = 300_000


@dataclass(frozen=True)
class WebResearchConfig:
    enabled: bool = True
    provider: str = "mojeek"
    max_results: int = 3
    timeout_seconds: float = 8.0
    fetch_pages: bool = False
    max_excerpt_chars: int = 1800
    max_context_chars: int = 9000

    @classmethod
    def from_env(cls) -> "WebResearchConfig":
        enabled_raw = os.getenv("WEB_RESEARCH_ENABLED", os.getenv("RESEARCH_ENABLED", "true"))
        enabled = enabled_raw.strip().lower() not in {"0", "false", "no", "off"}
        return cls(
            enabled=enabled,
            provider=os.getenv("WEB_RESEARCH_PROVIDER", os.getenv("RESEARCH_PROVIDER", "mojeek")).strip().lower(),
            max_results=_env_int("WEB_RESEARCH_MAX_RESULTS", 3, minimum=1, maximum=8),
            timeout_seconds=_env_float("WEB_RESEARCH_TIMEOUT_SECONDS", 8.0, minimum=1.0, maximum=30.0),
            fetch_pages=_env_bool("WEB_RESEARCH_FETCH_PAGES", False),
            max_excerpt_chars=_env_int("WEB_RESEARCH_MAX_EXCERPT_CHARS", 1800, minimum=300, maximum=5000),
            max_context_chars=_env_int("WEB_RESEARCH_MAX_CONTEXT_CHARS", 9000, minimum=2000, maximum=20000),
        )


@dataclass(frozen=True)
class ResearchSource:
    title: str
    url: str
    snippet: str = ""
    excerpt: str = ""


@dataclass(frozen=True)
class ResearchResult:
    query: str
    status: str
    sources: list[ResearchSource]
    error: str | None = None


def research_status(config: WebResearchConfig | None = None) -> dict:
    config = config or WebResearchConfig.from_env()
    return {
        "enabled": config.enabled,
        "provider": config.provider,
        "max_results": config.max_results,
        "timeout_seconds": config.timeout_seconds,
        "fetch_pages": config.fetch_pages,
    }


async def research_web(
    query: str,
    *,
    config: WebResearchConfig | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ResearchResult:
    config = config or WebResearchConfig.from_env()
    clean_query = " ".join((query or "").split())
    if not config.enabled:
        return ResearchResult(query=clean_query, status="disabled", sources=[])
    if not clean_query:
        return ResearchResult(query=clean_query, status="empty", sources=[])
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    timeout = httpx.Timeout(config.timeout_seconds)
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers=headers,
            timeout=timeout,
            transport=transport,
        ) as client:
            if config.provider == "mojeek":
                search_sources = await _mojeek_search(client, clean_query, config.max_results)
            elif config.provider == "duckduckgo":
                search_sources = await _duckduckgo_search(client, clean_query, config.max_results)
            else:
                return ResearchResult(
                    query=clean_query,
                    status="error",
                    sources=[],
                    error=f"Unsupported research provider: {config.provider}",
                )
            sources: list[ResearchSource] = []
            for source in search_sources:
                excerpt = ""
                if config.fetch_pages:
                    excerpt = await _fetch_excerpt(client, source.url, config.max_excerpt_chars)
                sources.append(
                    ResearchSource(
                        title=source.title,
                        url=source.url,
                        snippet=source.snippet,
                        excerpt=excerpt,
                    )
                )
        return ResearchResult(query=clean_query, status="ok" if sources else "empty", sources=sources)
    except Exception as exc:
        return ResearchResult(query=clean_query, status="error", sources=[], error=str(exc))


def build_research_context(
    result: ResearchResult | None,
    *,
    config: WebResearchConfig | None = None,
) -> str:
    if result is None:
        return ""
    config = config or WebResearchConfig.from_env()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = [
        f"Current date: {now}.",
        f"Web research was requested for: {result.query or 'the latest user request'}.",
        "Use this research context when it is relevant to the answer. Prefer it over stale model memory for current facts. When you use a source, include a final Sources section with the full source URLs, not only bracket numbers. If the sources are insufficient, say so plainly.",
    ]
    if result.status == "disabled":
        header.append("Web research is disabled by server configuration, so no live sources were retrieved.")
        return "\n".join(header)
    if result.status == "error":
        header.append(f"Web research failed before sources could be retrieved: {result.error or 'unknown error'}.")
        return "\n".join(header)
    if not result.sources:
        header.append("No useful web sources were retrieved.")
        return "\n".join(header)

    parts = ["\n".join(header), "Retrieved sources:"]
    for index, source in enumerate(result.sources, start=1):
        source_parts = [
            f"[{index}] {source.title or 'Untitled source'}",
            f"URL: {source.url}",
        ]
        if source.snippet:
            source_parts.append(f"Search snippet: {source.snippet}")
        if source.excerpt:
            source_parts.append(f"Page excerpt: {source.excerpt}")
        parts.append("\n".join(source_parts))

    context = "\n\n".join(parts)
    if len(context) > config.max_context_chars:
        return context[: config.max_context_chars].rsplit(" ", 1)[0] + "\n\n[Research context truncated.]"
    return context


def build_sources_footer(result: ResearchResult | None) -> str:
    if result is None or not result.sources:
        return ""
    lines = ["", "", "Sources:"]
    for source in result.sources:
        title = _clean_text(source.title) or source.url
        lines.append(f"- {title}: {source.url}")
    return "\n".join(lines)


async def _duckduckgo_search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
) -> list[ResearchSource]:
    response = await client.get(DDG_SEARCH_URL.format(query=quote_plus(query)))
    response.raise_for_status()
    parser = DuckDuckGoHTMLParser()
    parser.feed(response.text)
    return _dedupe_safe_results(parser.results, max_results)


async def _mojeek_search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
) -> list[ResearchSource]:
    response = await client.get(MOJEEK_SEARCH_URL.format(query=quote_plus(query)))
    response.raise_for_status()
    parser = MojeekHTMLParser()
    parser.feed(response.text)
    return _dedupe_safe_results(parser.results, max_results)


def _dedupe_safe_results(results: list[_SearchHit], max_results: int) -> list[ResearchSource]:
    seen: set[str] = set()
    sources: list[ResearchSource] = []

    for result in results:
        url = _normalize_duckduckgo_url(result.url)
        if not url or url in seen or not _is_safe_public_url(url):
            continue
        seen.add(url)
        sources.append(
            ResearchSource(
                title=_clean_text(result.title) or url,
                url=url,
                snippet=_clean_text(result.snippet),
            )
        )
        if len(sources) >= max_results:
            break
    return sources


async def _fetch_excerpt(
    client: httpx.AsyncClient,
    url: str,
    max_chars: int,
) -> str:
    if not _is_safe_public_url(url):
        return ""
    if not _hostname_resolves_public(url):
        return ""
    try:
        response = await client.get(
            url,
            headers={"Range": f"bytes=0-{MAX_PAGE_BYTES - 1}"},
            follow_redirects=False,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return ""
    content_type = response.headers.get("content-type", "")
    if content_type and not any(kind in content_type.lower() for kind in ("html", "text", "xml")):
        return ""
    text = response.text[:MAX_PAGE_BYTES]
    parser = TextExcerptParser(max_chars=max_chars)
    parser.feed(text)
    return _clean_text(parser.text)


@dataclass
class _SearchHit:
    title: str
    url: str
    snippet: str = ""


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[_SearchHit] = []
        self._active_link: dict | None = None
        self._active_snippet = False
        self._snippet_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {name: value or "" for name, value in attrs}
        class_name = attrs_dict.get("class", "")
        if tag == "a" and "result__a" in class_name:
            self._active_link = {"href": attrs_dict.get("href", ""), "chunks": []}
        elif "result__snippet" in class_name:
            self._active_snippet = True
            self._snippet_chunks = []

    def handle_data(self, data: str) -> None:
        if self._active_link is not None:
            self._active_link["chunks"].append(data)
        if self._active_snippet:
            self._snippet_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_link is not None:
            title = _clean_text(" ".join(self._active_link["chunks"]))
            href = self._active_link["href"]
            if title and href:
                self.results.append(_SearchHit(title=title, url=href))
            self._active_link = None
        if self._active_snippet:
            snippet = _clean_text(" ".join(self._snippet_chunks))
            if snippet and self.results:
                last = self.results[-1]
                self.results[-1] = _SearchHit(title=last.title, url=last.url, snippet=snippet)
            self._active_snippet = False
            self._snippet_chunks = []


class MojeekHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[_SearchHit] = []
        self._active_link: dict | None = None
        self._active_snippet = False
        self._snippet_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {name: value or "" for name, value in attrs}
        class_name = attrs_dict.get("class", "")
        if tag == "a" and "title" in class_name.split():
            self._active_link = {"href": attrs_dict.get("href", ""), "chunks": []}
        elif tag == "p" and "s" in class_name.split():
            self._active_snippet = True
            self._snippet_chunks = []

    def handle_data(self, data: str) -> None:
        if self._active_link is not None:
            self._active_link["chunks"].append(data)
        if self._active_snippet:
            self._snippet_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_link is not None:
            title = _clean_text(" ".join(self._active_link["chunks"]))
            href = self._active_link["href"]
            if title and href:
                self.results.append(_SearchHit(title=title, url=href))
            self._active_link = None
        if tag == "p" and self._active_snippet:
            snippet = _clean_text(" ".join(self._snippet_chunks))
            if snippet and self.results:
                last = self.results[-1]
                self.results[-1] = _SearchHit(title=last.title, url=last.url, snippet=snippet)
            self._active_snippet = False
            self._snippet_chunks = []


class TextExcerptParser(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "nav", "footer", "header"}

    def __init__(self, max_chars: int) -> None:
        super().__init__()
        self.max_chars = max_chars
        self._skip_depth = 0
        self._chunks: list[str] = []
        self._length = 0

    @property
    def text(self) -> str:
        return " ".join(self._chunks)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._length >= self.max_chars:
            return
        clean = _clean_text(data)
        if not clean:
            return
        remaining = self.max_chars - self._length
        chunk = clean[:remaining]
        self._chunks.append(chunk)
        self._length += len(chunk) + 1


def _normalize_duckduckgo_url(url: str) -> str:
    url = unescape(url or "").strip()
    if url.startswith("//"):
        url = f"https:{url}"
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        url = unquote(target)
    return url


def _is_safe_public_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.strip().lower()
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return _is_public_address(address)


def _hostname_resolves_public(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        try:
            infos = socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)
        except socket.gaierror:
            return False
        addresses = {info[4][0] for info in infos if info and info[4]}
        if not addresses:
            return False
        try:
            return all(_is_public_address(ipaddress.ip_address(address)) for address in addresses)
        except ValueError:
            return False
    return _is_public_address(address)


def _is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
    )


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value or "").replace("\u2060", "")).strip()


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}
