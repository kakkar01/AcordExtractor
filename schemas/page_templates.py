from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config


def get_page_templates(form_name: str) -> list[dict[str, Any]]:
    """Load reusable page-wise templates for a form, if present."""
    normalized = form_name.strip().lower().replace(" ", "")
    target_dir = config.PAGE_TEMPLATES_DIR / normalized
    if not target_dir.exists():
        return []

    templates: list[dict[str, Any]] = []
    for path in sorted(target_dir.glob("page_*.json")):
        templates.append(json.loads(path.read_text(encoding="utf-8")))
    return templates


def get_page_template(form_name: str, page_number: int) -> dict[str, Any] | None:
    """Load a single reusable page template by 1-based page number."""
    templates = get_page_templates(form_name)
    index = page_number - 1
    if index < 0 or index >= len(templates):
        return None
    return templates[index]


def get_page_template_paths(form_name: str) -> list[Path]:
    """Return page template file paths for a form."""
    normalized = form_name.strip().lower().replace(" ", "")
    target_dir = config.PAGE_TEMPLATES_DIR / normalized
    if not target_dir.exists():
        return []
    return sorted(target_dir.glob("page_*.json"))