"""Tests for empty page detection and filtering."""

from __future__ import annotations

import functools
import socket
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from claw.config import CrawlConfig
from claw.crawl.crawler import crawl
from claw.crawl.parser import parse_html
from claw.storage.writer import is_empty_page


def test_is_empty_page():
    assert is_empty_page(0, 150)
    assert is_empty_page(149, 150)
    assert not is_empty_page(150, 150)
    assert not is_empty_page(500, 150)


def test_parse_html_text_length():
    empty = parse_html("<html><body><main></main></body></html>", "https://example.com/empty")
    assert empty.text_length == 0

    rich = parse_html(
        "<html><body><main><h1>Title</h1><p>Hello world content here.</p></main></body></html>",
        "https://example.com/page",
    )
    assert rich.text_length > 20


@pytest.mark.asyncio
async def test_crawl_skips_empty_pages(tmp_path: Path):
    fixture_dir = Path(__file__).parent / "fixtures"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(fixture_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        config = CrawlConfig(
            max_depth=1,
            max_pages=10,
            request_delay_ms=0,
            ignore_robots=True,
            min_content_chars=150,
        )
        output = tmp_path / "out"
        url = f"http://127.0.0.1:{port}/root.html"
        result = await crawl(url, output, config)

        assert result.pages_saved >= 1
        assert result.pages_empty >= 1
        empty_records = [p for p in result.manifest.pages if p.status == "empty"]
        assert empty_records
        assert not list(output.glob("empty*.md"))
    finally:
        server.shutdown()
