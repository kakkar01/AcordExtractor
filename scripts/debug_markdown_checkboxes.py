"""Debug utility: extract all checkbox key-value pairs from _markdown in an extracted JSON.

Usage:
    python scripts/debug_markdown_checkboxes.py data/outputs/ACORD_125_1_extracted.json
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path


CHECKED_RE = re.compile(r"^-\s+\[x\]\s+(.+)$", re.IGNORECASE)
UNCHECKED_RE = re.compile(r"^-\s+\[\s\]\s+(.+)$")

STANDALONE_X_LABELS = {
    "owner",
    "tenant",
    "inside",
    "outside",
    "inside outside",
    "cell",
    "home",
    "bus",
    "trust",
    "quote",
    "renew",
    "issue policy",
    "change",
    "cancel",
    "bind",
    "direct",
    "agency",
    "safety manual",
    "loss control recs",
    "safety position",
    "monthly meetings",
    "osha",
    "non-payment",
    "non-renewal",
    "underwriting",
    "condition corrected (describe)",
    "condominiums",
}


def clean_label(raw: str) -> str:
    # strip trailing " X" marker that some forms print next to the checkbox
    return re.sub(r"\s+X\s*$", "", raw.strip()).strip()


def extract_checkboxes(markdown: str) -> list[dict]:
    results = []
    lines = [line.strip() for line in markdown.splitlines()]
    index = 0
    while index < len(lines):
        line = lines[index]

        m = CHECKED_RE.match(line)
        if m:
            results.append({"label": clean_label(m.group(1)), "selected": True})
            index += 1
            continue

        m = UNCHECKED_RE.match(line)
        if m:
            results.append({"label": clean_label(m.group(1)), "selected": False})
            index += 1
            continue

        if line in {"X", "x", "[x]", "[X]"}:
            # Look ahead a few non-empty lines and bind the standalone X to
            # the first explicit checkbox-style label we can trust.
            lookahead = index + 1
            while lookahead < len(lines) and lines[lookahead] == "":
                lookahead += 1

            for candidate_index in range(lookahead, min(lookahead + 5, len(lines))):
                candidate = re.sub(r"^#+\s*", "", lines[candidate_index]).strip().lower()
                if candidate in STANDALONE_X_LABELS:
                    results.append({"label": clean_label(lines[candidate_index]), "selected": True})
                    index = candidate_index + 1
                    break
            else:
                index += 1
            continue

        index += 1
    return results


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/outputs/ACORD_125_1_extracted.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    markdown = data.get("_markdown", "")

    if not markdown:
        print("No _markdown field found in output.")
        return

    items = extract_checkboxes(markdown)

    selected = [i for i in items if i["selected"]]
    unselected = [i for i in items if not i["selected"]]

    print(f"\n{'='*60}")
    print(f"  Checkbox debug: {path.name}")
    print(f"  Total checkboxes found: {len(items)}  |  ✓ Selected: {len(selected)}  |  ✗ Unselected: {len(unselected)}")
    print(f"{'='*60}\n")

    print("✓ SELECTED (Docling found [x]):")
    print("-" * 40)
    for item in selected:
        print(f"  [x]  {item['label']}")

    print("\n✗ UNSELECTED (Docling found [ ]):")
    print("-" * 40)
    for item in unselected:
        print(f"  [ ]  {item['label']}")

    print(f"\n{'='*60}")
    print("KEY-VALUE PAIRS (label → selected):")
    print("-" * 40)
    for item in items:
        mark = "[x]" if item["selected"] else "[ ]"
        print(f"  {mark}  {item['label']!r}: {item['selected']}")


if __name__ == "__main__":
    main()
