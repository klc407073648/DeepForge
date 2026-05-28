"""Integration tests for crawler using local HTTP server."""

from __future__ import annotations

import functools
import socket
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from claw.config import CrawlConfig
from claw.crawl.crawler import crawl


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.mark.asyncio
async def test_crawl_local_site(tmp_path: Path, fixture_dir: Path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(fixture_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        config = CrawlConfig(max_depth=1, max_pages=10, request_delay_ms=0, ignore_robots=True)
        output = tmp_path / "out"
        url = f"http://127.0.0.1:{port}/root.html"
        result = await crawl(url, output, config)

        assert result.pages_fetched >= 1
        assert (output / "manifest.json").is_file()
        assert list(output.glob("*.md"))
    finally:
        server.shutdown()
