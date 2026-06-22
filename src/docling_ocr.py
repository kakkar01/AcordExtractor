"""Docling OCR helper for Markdown extraction."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib import error, request


class DoclingOCR:
    """Wrapper around Docling local conversion or a remote Docling Serve API."""

    def __init__(
        self,
        service_url: str = "",
        api_key: str = "",
        timeout: float = 120.0,
        convert_path: str = "/v1/convert/source",
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.convert_path = convert_path if convert_path.startswith("/") else f"/{convert_path}"
        self._converter = None

        if not self.service_url:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError as exc:
                raise ImportError(
                    "Local Docling OCR requires the 'docling' package. "
                    "Install it with: pip install docling"
                ) from exc

            self._converter = DocumentConverter()

    @property
    def provider_label(self) -> str:
        if self.service_url:
            return f"Docling Serve at {self.service_url}"
        return "local Docling"

    @classmethod
    def from_config(cls) -> DoclingOCR | None:
        import config

        if config.DOCLING_SERVE_URL:
            return cls(
                service_url=config.DOCLING_SERVE_URL,
                api_key=config.DOCLING_SERVE_API_KEY,
                timeout=config.DOCLING_TIMEOUT_SECONDS,
                convert_path=config.DOCLING_SERVE_CONVERT_PATH,
            )

        if not config.DOCLING_USE_LOCAL:
            return None

        try:
            return cls(timeout=config.DOCLING_TIMEOUT_SECONDS)
        except ImportError:
            return None

    def extract_markdown(self, file_path: str | Path) -> str:
        return self.extract_markdown_with_metadata(file_path)["markdown"]

    def extract_markdown_with_metadata(self, file_path: str | Path) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        if self.service_url:
            markdown = self._extract_remote_markdown(path)
        else:
            markdown = self._extract_local_markdown(path)

        return {
            "markdown": markdown,
            "key_value_pairs": [],
        }

    def _extract_local_markdown(self, file_path: Path) -> str:
        if self._converter is None:
            raise RuntimeError("Local Docling converter is not initialized.")

        result = self._converter.convert(str(file_path))
        markdown = result.document.export_to_markdown()
        if not isinstance(markdown, str) or not markdown.strip():
            raise RuntimeError("Docling returned empty Markdown content.")

        return markdown.strip()

    def _extract_remote_markdown(self, file_path: Path) -> str:
        payload = {
            "sources": [
                {
                    "kind": "file",
                    "filename": file_path.name,
                    "base64_string": base64.b64encode(file_path.read_bytes()).decode("ascii"),
                }
            ],
            "options": {
                "to_formats": ["md"],
                "do_ocr": True,
                # Force pixel-level OCR instead of text-layer extraction.
                # Necessary for PDFs with custom/remapped font encodings where
                # Docling's text extractor reads raw glyph indices instead of
                # the correct characters (appears as Caesar-shifted ASCII).
                "force_ocr": True,
                "image_export_mode": "placeholder",
                "document_timeout": self.timeout,
            },
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["X-Api-Key"] = self.api_key

        req = request.Request(
            url=f"{self.service_url}{self.convert_path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Docling Serve request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Docling Serve request failed for {self.service_url}: {exc.reason}"
            ) from exc

        body = json.loads(raw_body)
        document = body.get("document")
        if not isinstance(document, dict):
            raise RuntimeError("Docling Serve response did not include a 'document' object.")

        markdown = document.get("md_content")
        if not isinstance(markdown, str) or not markdown.strip():
            raise RuntimeError("Docling Serve returned empty Markdown content.")

        status = str(body.get("status") or "").lower()
        if status and status not in {"success", "partial_success"}:
            raise RuntimeError(f"Docling Serve conversion failed with status '{status}'.")

        return markdown.strip()