"""Serve web/ on http://127.0.0.1:8000/ (local only)."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST, PORT = "127.0.0.1", 8000

if __name__ == "__main__":
    handler = partial(SimpleHTTPRequestHandler, directory=str(Path(__file__).parent / "web"))
    with ThreadingHTTPServer((HOST, PORT), handler) as httpd:
        print(f"Serving web/ at http://{HOST}:{PORT}/  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
