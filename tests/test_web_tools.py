"""Tests for tools/web.py — fetch, check, http_request, curl, web_search."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────

def _make_response(
    status_code: int = 200,
    headers: dict | None = None,
    text: str = "",
    json_data: dict | None = None,
    reason_phrase: str = "OK",
):
    """Build a mock httpx response with the common attributes."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.reason_phrase = reason_phrase
    resp.headers = headers or {}
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    return resp


# ── TestFetchUrl ─────────────────────────────────────────────

class TestFetchUrl:
    """Tests for tool_fetch_url: fetch content from a URL."""

    def test_fetch_json_response(self, monkeypatch, mock_confirm):
        """A JSON content-type response is pretty-printed inside a json fence."""
        import tools.web as web_mod

        payload = {"key": "value", "count": 42}
        resp = _make_response(
            headers={"content-type": "application/json"},
            text=json.dumps(payload),
            json_data=payload,
        )
        monkeypatch.setattr("httpx.get", lambda *a, **kw: resp)

        result = web_mod.tool_fetch_url("https://api.example.com/data")
        assert "Status: 200" in result
        assert "```json" in result
        assert '"key": "value"' in result

    def test_fetch_html_response(self, monkeypatch, mock_confirm):
        """HTML responses have scripts/styles stripped and tags removed."""
        import tools.web as web_mod

        html = (
            "<html><head><script>alert(1)</script>"
            "<style>body{color:red}</style></head>"
            "<body><p>Hello World</p></body></html>"
        )
        resp = _make_response(
            headers={"content-type": "text/html"},
            text=html,
        )
        monkeypatch.setattr("httpx.get", lambda *a, **kw: resp)

        result = web_mod.tool_fetch_url("https://example.com")
        assert "Status: 200" in result
        assert "Hello World" in result
        # Scripts and styles should be gone
        assert "alert(1)" not in result
        assert "color:red" not in result

    def test_fetch_plain_text_response(self, monkeypatch, mock_confirm):
        """Plain text (non-json, non-html) responses include the content type."""
        import tools.web as web_mod

        resp = _make_response(
            headers={"content-type": "text/plain"},
            text="Just plain text",
        )
        monkeypatch.setattr("httpx.get", lambda *a, **kw: resp)

        result = web_mod.tool_fetch_url("https://example.com/readme.txt")
        assert "Status: 200" in result
        assert "text/plain" in result
        assert "Just plain text" in result

    def test_fetch_auto_adds_https(self, monkeypatch, mock_confirm):
        """A bare domain gets https:// prepended automatically."""
        import tools.web as web_mod

        captured = {}

        def fake_get(url, **kw):
            captured["url"] = url
            return _make_response(
                headers={"content-type": "text/plain"},
                text="ok",
            )

        monkeypatch.setattr("httpx.get", fake_get)

        web_mod.tool_fetch_url("example.com")
        assert captured["url"] == "https://example.com"

    def test_fetch_empty_url_returns_error(self, mock_confirm):
        """An empty URL string should return an error without making a request."""
        import tools.web as web_mod

        result = web_mod.tool_fetch_url("")
        assert "Error" in result
        assert "Empty" in result

    def test_fetch_connection_error(self, monkeypatch, mock_confirm):
        """When httpx.get raises an exception, the error is reported."""
        import tools.web as web_mod

        monkeypatch.setattr(
            "httpx.get",
            MagicMock(side_effect=ConnectionError("refused")),
        )

        result = web_mod.tool_fetch_url("https://unreachable.example.com")
        assert "Error" in result
        assert "unreachable.example.com" in result

    def test_fetch_httpx_import_error_uses_urllib(self, monkeypatch, mock_confirm):
        """When httpx is unavailable, fall back to urllib."""
        import tools.web as web_mod

        # Make the `import httpx` inside tool_fetch_url raise ImportError
        import builtins
        _real_import = builtins.__import__

        def _import_no_httpx(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("no httpx")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _import_no_httpx)

        # Mock urllib.request.urlopen
        fake_body = b"Fallback body content"
        mock_resp_ctx = MagicMock()
        mock_resp_obj = MagicMock()
        mock_resp_obj.read.return_value = fake_body
        mock_resp_obj.status = 200
        mock_resp_ctx.__enter__ = MagicMock(return_value=mock_resp_obj)
        mock_resp_ctx.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **kw: mock_resp_ctx,
        )

        result = web_mod.tool_fetch_url("https://fallback.example.com")
        assert "Status: 200" in result
        assert "Fallback body content" in result


# ── TestCheckUrl ─────────────────────────────────────────────

class TestCheckUrl:
    """Tests for tool_check_url: HEAD request to check reachability."""

    def test_check_url_reachable(self, monkeypatch, mock_confirm):
        """A 200 HEAD response reports the URL as reachable."""
        import tools.web as web_mod

        resp = _make_response(
            status_code=200,
            reason_phrase="OK",
            headers={"content-type": "text/html", "server": "nginx"},
        )
        monkeypatch.setattr("httpx.head", lambda *a, **kw: resp)

        result = web_mod.tool_check_url("https://example.com")
        assert "Status: 200" in result
        assert "OK" in result

    def test_check_url_not_found(self, monkeypatch, mock_confirm):
        """A 404 HEAD response reports the correct status."""
        import tools.web as web_mod

        resp = _make_response(
            status_code=404,
            reason_phrase="Not Found",
            headers={"content-type": "text/html"},
        )
        monkeypatch.setattr("httpx.head", lambda *a, **kw: resp)

        result = web_mod.tool_check_url("https://example.com/missing")
        assert "404" in result
        assert "Not Found" in result

    def test_check_url_empty(self, mock_confirm):
        """An empty URL returns an error."""
        import tools.web as web_mod

        result = web_mod.tool_check_url("")
        assert "Error" in result
        assert "Empty" in result

    def test_check_url_auto_adds_https(self, monkeypatch, mock_confirm):
        """A bare domain gets https:// prepended for the HEAD request."""
        import tools.web as web_mod

        captured = {}

        def fake_head(url, **kw):
            captured["url"] = url
            return _make_response(reason_phrase="OK", headers={})

        monkeypatch.setattr("httpx.head", fake_head)

        web_mod.tool_check_url("example.com")
        assert captured["url"] == "https://example.com"


# ── TestHttpRequest ──────────────────────────────────────────

class TestHttpRequest:
    """Tests for tool_http_request: method|url|body_json format."""

    def test_get_request(self, monkeypatch, mock_confirm):
        """A simple GET request returns status and body."""
        import tools.web as web_mod

        resp = _make_response(
            status_code=200,
            reason_phrase="OK",
            headers={"content-type": "text/plain"},
            text="response body",
        )
        monkeypatch.setattr("httpx.request", lambda *a, **kw: resp)
        # Suppress console.print calls
        monkeypatch.setattr(web_mod, "console", MagicMock())

        result = web_mod.tool_http_request("GET|https://api.example.com/items")
        assert "Status: 200" in result
        assert "response body" in result

    def test_post_with_json_body(self, monkeypatch, mock_confirm):
        """POST with valid JSON body sends json= kwarg to httpx.request."""
        import tools.web as web_mod

        captured = {}

        def fake_request(method, url, **kw):
            captured["method"] = method
            captured["url"] = url
            captured["kwargs"] = kw
            return _make_response(
                status_code=201,
                reason_phrase="Created",
                headers={"content-type": "application/json"},
                text='{"id": 1}',
                json_data={"id": 1},
            )

        monkeypatch.setattr("httpx.request", fake_request)
        monkeypatch.setattr(web_mod, "console", MagicMock())

        result = web_mod.tool_http_request('POST|https://api.example.com/items|{"name":"test"}')
        assert captured["method"] == "POST"
        assert captured["kwargs"]["json"] == {"name": "test"}
        assert "201" in result

    def test_post_with_plain_body(self, monkeypatch, mock_confirm):
        """POST with non-JSON body sends content= kwarg with text/plain header."""
        import tools.web as web_mod

        captured = {}

        def fake_request(method, url, **kw):
            captured["kwargs"] = kw
            return _make_response(
                status_code=200,
                reason_phrase="OK",
                headers={"content-type": "text/plain"},
                text="ok",
            )

        monkeypatch.setattr("httpx.request", fake_request)
        monkeypatch.setattr(web_mod, "console", MagicMock())

        web_mod.tool_http_request("POST|https://api.example.com/submit|not valid json{{{")
        assert captured["kwargs"]["content"] == "not valid json{{{"
        assert captured["kwargs"]["headers"]["Content-Type"] == "text/plain"

    def test_empty_url_returns_error(self, monkeypatch, mock_confirm):
        """Missing URL in args returns a format error."""
        import tools.web as web_mod
        monkeypatch.setattr(web_mod, "console", MagicMock())

        result = web_mod.tool_http_request("GET")
        assert "Error" in result
        assert "format" in result.lower() or "url" in result.lower()

    def test_auto_adds_http(self, monkeypatch, mock_confirm):
        """A URL without a scheme gets http:// prepended (not https)."""
        import tools.web as web_mod

        captured = {}

        def fake_request(method, url, **kw):
            captured["url"] = url
            return _make_response(
                headers={"content-type": "text/plain"},
                text="ok",
            )

        monkeypatch.setattr("httpx.request", fake_request)
        monkeypatch.setattr(web_mod, "console", MagicMock())

        web_mod.tool_http_request("GET|localhost:8080/health")
        assert captured["url"] == "http://localhost:8080/health"


# ── TestCurl ─────────────────────────────────────────────────

class TestCurl:
    """Tests for tool_curl: simple curl-like fetch."""

    def test_curl_success(self, monkeypatch, mock_confirm):
        """Successful curl returns HTTP status code and body."""
        import tools.web as web_mod

        resp = _make_response(
            status_code=200,
            headers={"content-type": "text/html", "server": "nginx"},
            text="<h1>Hello</h1>",
        )
        # Make headers iterable as items()
        resp.headers = {"content-type": "text/html", "server": "nginx"}
        monkeypatch.setattr("httpx.get", lambda *a, **kw: resp)

        result = web_mod.tool_curl("http://localhost:8080")
        assert "HTTP 200" in result
        assert "<h1>Hello</h1>" in result

    def test_curl_empty_url(self, mock_confirm):
        """An empty URL returns an error."""
        import tools.web as web_mod

        result = web_mod.tool_curl("")
        assert "Error" in result
        assert "Empty" in result

    def test_curl_auto_adds_http(self, monkeypatch, mock_confirm):
        """A bare host:port gets http:// prepended."""
        import tools.web as web_mod

        captured = {}

        def fake_get(url, **kw):
            captured["url"] = url
            r = _make_response(text="ok")
            r.headers = {}
            return r

        monkeypatch.setattr("httpx.get", fake_get)

        web_mod.tool_curl("localhost:3000")
        assert captured["url"] == "http://localhost:3000"

    def test_curl_error(self, monkeypatch, mock_confirm):
        """When httpx.get raises, the error message is returned."""
        import tools.web as web_mod

        monkeypatch.setattr(
            "httpx.get",
            MagicMock(side_effect=ConnectionError("connection refused")),
        )

        result = web_mod.tool_curl("http://localhost:9999")
        assert "Error" in result


# ── TestWebSearch ────────────────────────────────────────────

class TestWebSearch:
    """Tests for tool_web_search: DuckDuckGo search wrapper."""

    def test_web_search_returns_results(self, monkeypatch, mock_confirm):
        """When _web_search_raw returns results, they are formatted as numbered list."""
        import tools.web as web_mod

        fake_results = [
            {"title": "Python Docs", "url": "https://docs.python.org", "snippet": "Official docs"},
            {"title": "PyPI", "url": "https://pypi.org", "snippet": "Package index"},
        ]
        monkeypatch.setattr(web_mod, "_web_search_raw", lambda q, m=5: fake_results)

        result = web_mod.tool_web_search("python")
        assert "Search results for: python" in result
        assert "1. Python Docs" in result
        assert "https://docs.python.org" in result
        assert "2. PyPI" in result
        assert "Official docs" in result

    def test_web_search_no_results(self, monkeypatch, mock_confirm):
        """When _web_search_raw returns empty, a 'No results' message is shown."""
        import tools.web as web_mod

        monkeypatch.setattr(web_mod, "_web_search_raw", lambda q, m=5: [])

        result = web_mod.tool_web_search("xyzzy_nonexistent_topic_12345")
        assert "No results" in result

    def test_web_search_with_max_results(self, monkeypatch, mock_confirm):
        """The max_results parameter is parsed from pipe-separated args."""
        import tools.web as web_mod

        captured = {}

        def fake_raw(query, max_results=5):
            captured["max_results"] = max_results
            return [
                {"title": f"Result {i}", "url": f"https://r{i}.com", "snippet": ""}
                for i in range(max_results)
            ]

        monkeypatch.setattr(web_mod, "_web_search_raw", fake_raw)

        result = web_mod.tool_web_search("python|3")
        assert captured["max_results"] == 3
        assert "1. Result 0" in result

    def test_web_search_empty_query(self, mock_confirm):
        """An empty query returns an error."""
        import tools.web as web_mod

        result = web_mod.tool_web_search("")
        assert "Error" in result
        assert "Empty" in result


# ── TestWebSearchRaw ─────────────────────────────────────────

class TestWebSearchRaw:
    """Tests for _web_search_raw: parse DuckDuckGo HTML into structured results."""

    def test_web_search_raw_parses_results(self, monkeypatch, mock_confirm):
        """Correctly extracts title, URL, and snippet from DuckDuckGo-like HTML."""
        import tools.web as web_mod

        ddg_html = """
        <html><body>
        <div class="result">
            <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&amp;rut=abc">
                <b>Python</b> Documentation
            </a>
            <span class="result__snippet">Official <b>Python</b> documentation for version 3.</span>
        </div>
        <div class="result">
            <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fpypi.org%2F&amp;rut=def">
                PyPI
            </a>
            <span class="result__snippet">The Python Package Index.</span>
        </div>
        </body></html>
        """

        resp = _make_response(text=ddg_html)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: resp)

        results = web_mod._web_search_raw("python", max_results=5)
        assert len(results) == 2

        assert "Python" in results[0]["title"]
        assert "Documentation" in results[0]["title"]
        assert results[0]["url"] == "https://docs.python.org/3/"
        assert "documentation" in results[0]["snippet"].lower()

        assert "PyPI" in results[1]["title"]
        assert results[1]["url"] == "https://pypi.org/"

    def test_web_search_raw_handles_error(self, monkeypatch, mock_confirm):
        """When httpx.post raises, an empty list is returned (never raises)."""
        import tools.web as web_mod

        monkeypatch.setattr(
            "httpx.post",
            MagicMock(side_effect=ConnectionError("network down")),
        )
        # Also make urllib fallback fail
        import builtins
        _real_import = builtins.__import__

        def _import_no_urllib(name, *args, **kwargs):
            if name == "urllib.request":
                raise ImportError("no urllib")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _import_no_urllib)

        results = web_mod._web_search_raw("python")
        assert results == []

    def test_web_search_raw_empty_query(self, mock_confirm):
        """An empty/whitespace query returns an empty list immediately."""
        import tools.web as web_mod

        assert web_mod._web_search_raw("") == []
        assert web_mod._web_search_raw("   ") == []

    def test_web_search_raw_respects_max_results(self, monkeypatch, mock_confirm):
        """The max_results parameter caps the number of returned results."""
        import tools.web as web_mod

        # Build HTML with 5 results
        result_blocks = []
        for i in range(5):
            result_blocks.append(
                f'<a class="result__a" href="https://r{i}.example.com">Result {i}</a>'
                f'<span class="result__snippet">Snippet {i}</span>'
            )
        ddg_html = "<html><body>" + "\n".join(result_blocks) + "</body></html>"

        resp = _make_response(text=ddg_html)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: resp)

        results = web_mod._web_search_raw("test", max_results=2)
        assert len(results) == 2
        assert results[0]["title"] == "Result 0"
        assert results[1]["title"] == "Result 1"

    def test_web_search_raw_decodes_redirect_url(self, monkeypatch, mock_confirm):
        """DuckDuckGo redirect URLs with uddg= parameter are decoded properly."""
        import tools.web as web_mod

        ddg_html = """
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.example.com%2Fpath%3Fq%3D1&rut=abc">
            Example Site
        </a>
        <span class="result__snippet">An example.</span>
        """

        resp = _make_response(text=ddg_html)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: resp)

        results = web_mod._web_search_raw("example")
        assert len(results) == 1
        assert results[0]["url"] == "https://www.example.com/path?q=1"
