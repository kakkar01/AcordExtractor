#!/usr/bin/env python3
"""Simple local UI for ACORD form upload and extraction review."""
from __future__ import annotations

import argparse
import cgi
import io
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AcorUIHandler(BaseHTTPRequestHandler):
    server_version = "AcordExtractorUI/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._serve_file(TEMPLATES_DIR / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/uploads/"):
            filename = parsed.path[len("/uploads/") :]
            if not filename:
                self._send_json(400, {"error": "Missing upload filename"})
                return
            target = UPLOAD_DIR / filename
            if target.exists() and target.is_file():
                content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                self._serve_file(target, content_type)
                return
            self._send_json(404, {"error": "Upload not found"})
            return
        if parsed.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/extract":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return

        if content_length <= 0:
            self._send_json(400, {"error": "Missing request body"})
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self._send_json(400, {"error": "Expected multipart/form-data upload"})
            return

        try:
            environ = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(content_length),
            }
            form = cgi.FieldStorage(
                fp=io.BytesIO(self.rfile.read(content_length)),
                headers=self.headers,
                environ=environ,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json(400, {"error": f"Failed to parse upload: {exc}"})
            return

        uploaded = form["file"] if "file" in form else None
        if uploaded is None or not getattr(uploaded, "filename", None):
            self._send_json(400, {"error": "No file uploaded"})
            return

        form_name = form.getvalue("form_name", "").strip()
        original_name = uploaded.filename
        suffix = Path(original_name).suffix or ".pdf"
        saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

        try:
            with saved_path.open("wb") as handle:
                shutil.copyfileobj(uploaded.file, handle)

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "main.py"),
                "extract",
                str(saved_path),
                "--no-save",
            ]
            if form_name:
                cmd.extend(["--form", form_name])

            completed = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout).strip()
                self._send_json(
                    500,
                    {
                        "error": "Extraction failed",
                        "details": details or "No output captured from the extractor.",
                    },
                )
                return

            payload = json.loads(completed.stdout)
            self._send_json(
                200,
                {
                    "filename": saved_path.name,
                    "source_name": original_name,
                    "output": payload,
                },
            )
        except subprocess.TimeoutExpired:
            self._send_json(504, {"error": "Extraction timed out"})
        except json.JSONDecodeError as exc:
            self._send_json(500, {"error": f"Failed to parse extractor output: {exc}"})
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def _serve_file(self, path: Path, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ACORD extraction review UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AcorUIHandler)
    print(f"Serving ACORD UI at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
