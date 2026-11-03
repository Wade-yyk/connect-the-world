"""Start the local assignment preview; close its console to stop it."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen
import webbrowser


ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8765/"


class PreviewHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    handler = partial(PreviewHandler, directory=str(ROOT))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 8765), handler)
    except OSError as error:
        if error.errno not in (48, 98, 10048):
            raise
        with urlopen(URL + "index.html", timeout=3) as response:
            if response.read() != (ROOT / "index.html").read_bytes():
                raise RuntimeError("Port 8765 is being used by a different website.") from error
        print(f"Preview is already running: {URL}", flush=True)
        if not args.no_browser:
            webbrowser.open(URL)
        return
    with server:
        print(f"Assignment preview: {URL}", flush=True)
        print("Keep this window open. Close it or press Ctrl+C to stop the preview.", flush=True)
        if not args.no_browser:
            webbrowser.open(URL)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
