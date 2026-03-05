"""Web / HTTP tools — fetch, check, request, curl, serve, screenshot, browser, websocket."""

import os
import sys
import re
import json
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from tools.common import console, _sanitize_tool_args, _confirm_command, _scan_output
from tools.shell import _background_servers

import socket
import time as _time


def tool_fetch_url(args: str) -> str:
    """Fetch content from a URL."""
    url = _sanitize_tool_args(args)

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        import httpx
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        content_type = resp.headers.get("content-type", "")

        if "json" in content_type:
            try:
                parsed = json.dumps(json.loads(resp.text), indent=2)
                return (
                    f"URL: {url}\nStatus: {resp.status_code}\n"
                    f"```json\n{parsed[:5000]}\n```"
                )
            except json.JSONDecodeError:
                return (
                    f"URL: {url}\nStatus: {resp.status_code}\n{resp.text[:3000]}"
                )
        elif "html" in content_type:
            text = re.sub(
                r'<script.*?</script>', '', resp.text, flags=re.DOTALL
            )
            text = re.sub(
                r'<style.*?</style>', '', text, flags=re.DOTALL
            )
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return _scan_output(
                f"URL: {url}\nStatus: {resp.status_code}\n{text[:3000]}"
            )
        else:
            return _scan_output(
                f"URL: {url}\nStatus: {resp.status_code}\n"
                f"Type: {content_type}\n{resp.text[:2000]}"
            )
    except ImportError:
        # Fallback to urllib
        try:
            from urllib.request import urlopen, Request
            from urllib.error import URLError
            req = Request(url, headers={"User-Agent": "AI-CLI/1.0"})
            with urlopen(req, timeout=15) as resp:
                body = resp.read().decode("utf-8", errors="replace")[:3000]
                return _scan_output(f"URL: {url}\nStatus: {resp.status}\n{body}")
        except Exception as e:
            return f"Error fetching {url}: {e}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def tool_check_url(args: str) -> str:
    """Check if a URL is reachable."""
    url = _sanitize_tool_args(args)

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        import httpx
        resp = httpx.head(url, timeout=10, follow_redirects=True)
        return (
            f"URL: {url}\n"
            f"Status: {resp.status_code} ({resp.reason_phrase})\n"
            f"Headers: {dict(list(resp.headers.items())[:10])}"
        )
    except ImportError:
        try:
            from urllib.request import urlopen, Request
            req = Request(url, method="HEAD", headers={"User-Agent": "AI-CLI/1.0"})
            with urlopen(req, timeout=10) as resp:
                return (
                    f"URL: {url}\n"
                    f"Status: {resp.status}\n"
                    f"Headers: {dict(list(resp.headers.items())[:10])}"
                )
        except Exception as e:
            return f"URL: {url}\nError: {e}"
    except Exception as e:
        return f"URL: {url}\nError: {e}"


def tool_http_request(args: str) -> str:
    """Make an HTTP request with method, URL, and optional body."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")

    method = parts[0].strip().upper() if parts else "GET"
    url = parts[1].strip() if len(parts) > 1 else ""
    body = parts[2].strip() if len(parts) > 2 else None

    if not url:
        return "Error: Use format method|url|body_json"
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    console.print(f"\n[yellow]{method} {url}[/yellow]")
    if body:
        console.print(f"[dim]Body: {body[:200]}[/dim]")

    try:
        import httpx

        kwargs = {"timeout": 15, "follow_redirects": True}
        if body:
            try:
                kwargs["json"] = json.loads(body)
            except json.JSONDecodeError:
                kwargs["content"] = body
                kwargs["headers"] = {"Content-Type": "text/plain"}

        resp = httpx.request(method, url, **kwargs)

        output = f"Status: {resp.status_code} {resp.reason_phrase}\n"
        output += f"Headers: {dict(list(resp.headers.items())[:10])}\n"

        content_type = resp.headers.get("content-type", "")
        if "json" in content_type:
            try:
                output += f"Body:\n```json\n{json.dumps(resp.json(), indent=2)[:3000]}\n```"
            except Exception:
                output += f"Body:\n{resp.text[:3000]}"
        else:
            output += f"Body:\n{resp.text[:3000]}"

        return _scan_output(output)
    except ImportError:
        return "Error: httpx not installed \u2014 run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"


def tool_curl(args: str) -> str:
    """Simple curl-like fetch (for testing local servers)."""
    url = _sanitize_tool_args(args)

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        import httpx
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        output = f"HTTP {resp.status_code}\n"
        for k, v in list(resp.headers.items())[:15]:
            output += f"{k}: {v}\n"
        output += f"\n{resp.text[:5000]}"
        return _scan_output(output)
    except ImportError:
        try:
            from urllib.request import urlopen
            with urlopen(url, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="replace")[:5000]
                return _scan_output(f"HTTP {resp.status}\n{body}")
        except Exception as e:
            return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def tool_serve_static(args: str) -> str:
    """Start a static file server for testing web apps."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    from tools.common import _sanitize_path_arg
    directory = _sanitize_path_arg(parts[0]) if parts else "."
    port = 8000

    if len(parts) > 1:
        try:
            port = int(parts[1].strip())
        except ValueError:
            pass

    path = Path(directory).resolve()
    if not path.exists():
        return f"Error: Directory not found: {directory}"
    if not path.is_dir():
        return f"Error: Not a directory: {directory}"

    # Check if port is already in use
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        if result == 0:
            return f"Error: Port {port} is already in use. Try a different port."
    except Exception:
        pass

    console.print(f"\n[yellow]Serve static:[/yellow] {directory} on port {port}")
    if not _confirm_command("Start server? (y/n): "):
        return "Cancelled."

    try:
        # Use Python's built-in HTTP server
        cmd = f'{sys.executable} -m http.server {port} --directory "{path}" --bind 127.0.0.1'

        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )

        # Wait a moment and check it started
        _time.sleep(1)
        if proc.poll() is not None:
            return "Error: Server failed to start (process exited immediately)."

        _background_servers[port] = {
            "process": proc,
            "command": cmd,
            "directory": str(path),
            "started": datetime.now().isoformat(),
        }

        return (
            f"\u2713 Static server started!\n"
            f"  URL: http://localhost:{port}\n"
            f"  Directory: {directory}\n"
            f"  PID: {proc.pid}\n"
            f"  Stop with: <tool:serve_stop>{port}</tool>"
        )
    except Exception as e:
        return f"Error starting server: {e}"


def tool_serve_stop(args: str) -> str:
    """Stop a running development server."""
    cleaned = _sanitize_tool_args(args)

    try:
        port = int(cleaned)
    except (ValueError, TypeError):
        return f"Error: Invalid port: {args}"

    if port in _background_servers:
        info = _background_servers[port]
        try:
            proc = info["process"]
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                info["process"].kill()
            except Exception:
                pass
        del _background_servers[port]
        return f"\u2713 Stopped server on port {port}"

    # Try to kill whatever is on that port
    try:
        if sys.platform != "win32":
            result = subprocess.run(
                f"lsof -ti :{port}",
                shell=True, capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split("\n"):
                    try:
                        os.kill(int(pid.strip()), signal.SIGTERM)
                    except Exception:
                        pass
                return f"\u2713 Killed process(es) on port {port}"
        return f"No tracked server on port {port}"
    except Exception as e:
        return f"Error: {e}"


def tool_serve_list(args: str) -> str:
    """List all running development servers."""
    if not _background_servers:
        return "No servers running."

    output = "Running servers:\n"
    for port, info in _background_servers.items():
        proc = info["process"]
        status = "running" if proc.poll() is None else f"exited({proc.returncode})"
        output += (
            f"  Port {port}: [{status}]\n"
            f"    Dir: {info.get('directory', 'N/A')}\n"
            f"    PID: {proc.pid}\n"
            f"    Started: {info['started']}\n"
        )
    return output


def tool_screenshot_url(args: str) -> str:
    """Take a screenshot of a URL (requires playwright)."""
    url = _sanitize_tool_args(args)

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"screenshot_{timestamp}.png"

    try:
        from playwright.sync_api import sync_playwright

        console.print(f"\n[yellow]Screenshot:[/yellow] {url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.screenshot(path=output_path, full_page=False)
            title = page.title()
            browser.close()

        return (
            f"\u2713 Screenshot saved: {output_path}\n"
            f"  URL: {url}\n"
            f"  Title: {title}\n"
            f"  Size: 1280x720"
        )
    except ImportError:
        # Fallback: try using a command-line tool
        try:
            result = subprocess.run(
                f'npx playwright screenshot "{url}" {output_path}',
                shell=True, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return f"\u2713 Screenshot saved: {output_path}"
            return (
                "Error: playwright not available.\n"
                "Install with: pip install playwright && playwright install chromium"
            )
        except Exception:
            return (
                "Error: playwright not available.\n"
                "Install with: pip install playwright && playwright install chromium"
            )
    except Exception as e:
        return f"Error taking screenshot: {e}"


def tool_browser_open(args: str) -> str:
    """Open a URL in the default browser."""
    url = _sanitize_tool_args(args)

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("http://", "https://", "file://")):
        url = "http://" + url

    try:
        import webbrowser
        webbrowser.open(url)
        return f"\u2713 Opened in browser: {url}"
    except Exception as e:
        return f"Error opening browser: {e}"


def tool_websocket_test(args: str) -> str:
    """Test a WebSocket connection."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    url = parts[0].strip()
    message = parts[1].strip() if len(parts) > 1 else None

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("ws://", "wss://")):
        url = "ws://" + url

    try:
        import websockets
        import asyncio

        async def test_ws():
            async with websockets.connect(url, close_timeout=5) as ws:
                output = f"\u2713 Connected to {url}\n"
                if message:
                    await ws.send(message)
                    output += f"  Sent: {message}\n"
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=5)
                        output += f"  Received: {response[:1000]}\n"
                    except asyncio.TimeoutError:
                        output += "  No response within 5s\n"
                return output

        return asyncio.run(test_ws())
    except ImportError:
        return (
            "Error: websockets not installed.\n"
            "Install with: pip install websockets"
        )
    except Exception as e:
        return f"Error: {e}"
