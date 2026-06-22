from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


META_KEYS = {
    "_markdown",
    "_ocr_key_values",
    "_validation_warnings",
    "_field_reasons",
    "_raw",
    "_schema_mismatch",
}


@dataclass
class FieldCheck:
    path: str
    value: Any
    verdict: str
    reason: str
    suggested_value: Any | None


@dataclass
class FileReport:
    file: str
    total_leaf_fields: int
    checked_fields: int
    wrong_fields: int
    unclear_fields: int
    error_fields: int
    wrong_paths: list[str]
    details: list[dict[str, Any]]


def _coerce_json(value: str) -> Any:
    text = (value or "").strip()
    if not text:
        return None

    if text.lower() in {"null", "none"}:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"

    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except Exception:
            return text

    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except Exception:
            return text

    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            return json.loads(text)
        except Exception:
            return text

    return text


def _extract_first_json_object(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _flatten_fields(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            if key in META_KEYS:
                continue
            if key.startswith("_"):
                continue
            out.extend(_flatten_fields(child, f"{path}.{key}"))
        return out

    if isinstance(value, list):
        if not value:
            out.append((path, value))
            return out
        for idx, child in enumerate(value):
            out.extend(_flatten_fields(child, f"{path}[{idx}]"))
        return out

    out.append((path, value))
    return out


def _verify_one_field(
    client: OpenAI,
    model: str,
    markdown_text: str,
    field_path: str,
    field_value: Any,
    max_tokens: int,
) -> FieldCheck:
    prompt = (
        "You are validating an extracted insurance form field against OCR markdown. "
        "Given a field path and extracted value, decide whether it is correct from the markdown evidence. "
        "Do not guess. If evidence is weak, return 'unclear'.\n\n"
        f"Field path: {field_path}\n"
        f"Extracted value (JSON): {json.dumps(field_value, ensure_ascii=False)}\n\n"
        "OCR Markdown:\n"
        f"{markdown_text}\n\n"
        "Return JSON only with keys: verdict, reason, suggested_value.\n"
        "Rules:\n"
        "- verdict must be one of: correct, wrong, unclear\n"
        "- suggested_value should be null when verdict is correct or unclear\n"
        "- if verdict is wrong, suggested_value should be the corrected JSON value (not prose)."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )

    raw = ""
    if response.choices and response.choices[0].message:
        raw = response.choices[0].message.content or ""

    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(raw)
    except Exception:
        obj = _extract_first_json_object(raw)
        if obj:
            try:
                parsed = json.loads(obj)
            except Exception:
                parsed = None

    if not isinstance(parsed, dict):
        return FieldCheck(
            path=field_path,
            value=field_value,
            verdict="error",
            reason=f"Validator returned non-JSON response: {raw[:200]}",
            suggested_value=None,
        )

    verdict = str(parsed.get("verdict", "unclear")).strip().lower()
    if verdict not in {"correct", "wrong", "unclear"}:
        verdict = "unclear"

    reason = str(parsed.get("reason", "")).strip()
    suggested = parsed.get("suggested_value")

    if isinstance(suggested, str):
        suggested = _coerce_json(suggested)

    if verdict in {"correct", "unclear"}:
        suggested = None

    return FieldCheck(
        path=field_path,
        value=field_value,
        verdict=verdict,
        reason=reason,
        suggested_value=suggested,
    )


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n\n...[TRUNCATED]...\n\n" + text[-tail:]


def _is_benign_unclear_issue(verdict: str, current_value: Any, suggested_value: Any, reason: str) -> bool:
    """Drop non-actionable unclear flags that explicitly confirm consistency."""
    if verdict != "unclear":
        return False

    if suggested_value not in {None, "", "null"}:
        return False

    reason_l = (reason or "").strip().lower()
    if not reason_l:
        return False

    consistency_markers = [
        "consistent with",
        "which is consistent",
        "matches the",
        "all selected fields set to false, which is consistent",
        "blank field",
        "unchecked boxes",
    ]
    return any(marker in reason_l for marker in consistency_markers)


def _validate_file_document_level(
    client: OpenAI,
    model: str,
    file_path: Path,
    max_tokens: int,
    use_model_check: bool,
    markdown_max_chars: int,
    extracted_json_max_chars: int,
) -> FileReport:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    markdown = data.get("_markdown")
    total_leaf = len(_flatten_fields(data))

    if not isinstance(markdown, str) or not markdown.strip():
        return FileReport(
            file=str(file_path),
            total_leaf_fields=total_leaf,
            checked_fields=0,
            wrong_fields=0,
            unclear_fields=0,
            error_fields=1,
            wrong_paths=[],
            details=[
                {
                    "path": "$",
                    "value": None,
                    "verdict": "error",
                    "reason": "Missing _markdown in extracted JSON",
                    "suggested_value": None,
                }
            ],
        )

    if not use_model_check:
        return FileReport(
            file=str(file_path),
            total_leaf_fields=total_leaf,
            checked_fields=0,
            wrong_fields=0,
            unclear_fields=total_leaf,
            error_fields=0,
            wrong_paths=[],
            details=[
                {
                    "path": "$",
                    "value": None,
                    "verdict": "unclear",
                    "reason": "Model check disabled (--no-model-check).",
                    "suggested_value": None,
                }
            ],
        )

    extracted_payload = {k: v for k, v in data.items() if k not in META_KEYS and not k.startswith("_")}
    extracted_json = json.dumps(extracted_payload, ensure_ascii=False)
    markdown_short = _truncate_text(markdown, markdown_max_chars)
    extracted_short = _truncate_text(extracted_json, extracted_json_max_chars)

    prompt = (
        "You are validating a full extracted insurance JSON against OCR markdown for the same document. "
        "Find only high-confidence issues. Do not guess.\n\n"
        "Return JSON only with this shape:\n"
        "{\n"
        "  \"issues\": [\n"
        "    {\n"
        "      \"path\": \"$.field.path\",\n"
        "      \"verdict\": \"wrong\" or \"unclear\",\n"
        "      \"reason\": \"short reason\",\n"
        "      \"current_value\": <any JSON value>,\n"
        "      \"suggested_value\": <any JSON value or null>\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Include only true issues; do not include correct fields.\n"
        "- If extracted value is null/empty and OCR field is blank, treat as correct and DO NOT include it.\n"
        "- If extracted booleans are false and OCR checkboxes are unchecked, treat as correct and DO NOT include it.\n"
        "- path must point to extracted JSON paths.\n"
        "- Use suggested_value when verdict is wrong and correction is clear; else null.\n"
        "- Keep issues list concise and high precision.\n\n"
        f"Extracted JSON:\n{extracted_short}\n\n"
        f"OCR Markdown:\n{markdown_short}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return FileReport(
            file=str(file_path),
            total_leaf_fields=total_leaf,
            checked_fields=0,
            wrong_fields=0,
            unclear_fields=0,
            error_fields=1,
            wrong_paths=[],
            details=[
                {
                    "path": "$",
                    "value": None,
                    "verdict": "error",
                    "reason": f"Validation call failed: {exc}",
                    "suggested_value": None,
                }
            ],
        )

    raw = ""
    if response.choices and response.choices[0].message:
        raw = response.choices[0].message.content or ""

    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(raw)
    except Exception:
        obj = _extract_first_json_object(raw)
        if obj:
            try:
                parsed = json.loads(obj)
            except Exception:
                parsed = None

    if not isinstance(parsed, dict):
        return FileReport(
            file=str(file_path),
            total_leaf_fields=total_leaf,
            checked_fields=0,
            wrong_fields=0,
            unclear_fields=0,
            error_fields=1,
            wrong_paths=[],
            details=[
                {
                    "path": "$",
                    "value": None,
                    "verdict": "error",
                    "reason": f"Validator returned non-JSON response: {raw[:250]}",
                    "suggested_value": None,
                }
            ],
        )

    issues_raw = parsed.get("issues")
    if not isinstance(issues_raw, list):
        issues_raw = []

    details: list[dict[str, Any]] = []
    wrong_paths: list[str] = []
    wrong_count = 0
    unclear_count = 0

    for issue in issues_raw:
        if not isinstance(issue, dict):
            continue
        path = str(issue.get("path", "$")).strip() or "$"
        verdict = str(issue.get("verdict", "unclear")).strip().lower()
        if verdict not in {"wrong", "unclear"}:
            verdict = "unclear"
        reason = str(issue.get("reason", "")).strip()
        current_value = issue.get("current_value")
        suggested_value = issue.get("suggested_value")

        if _is_benign_unclear_issue(verdict, current_value, suggested_value, reason):
            continue

        details.append(
            {
                "path": path,
                "value": current_value,
                "verdict": verdict,
                "reason": reason,
                "suggested_value": suggested_value,
            }
        )

        if verdict == "wrong":
            wrong_count += 1
            wrong_paths.append(path)
        else:
            unclear_count += 1

    return FileReport(
        file=str(file_path),
        total_leaf_fields=total_leaf,
        checked_fields=total_leaf,
        wrong_fields=wrong_count,
        unclear_fields=unclear_count,
        error_fields=0,
        wrong_paths=wrong_paths,
        details=details,
    )


def _validate_file(
    client: OpenAI,
    model: str,
    file_path: Path,
    max_fields: int,
    max_tokens: int,
    sleep_seconds: float,
    use_model_check: bool,
) -> FileReport:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    markdown = data.get("_markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        return FileReport(
            file=str(file_path),
            total_leaf_fields=0,
            checked_fields=0,
            wrong_fields=0,
            unclear_fields=0,
            error_fields=1,
            wrong_paths=[],
            details=[
                {
                    "path": "$",
                    "value": None,
                    "verdict": "error",
                    "reason": "Missing _markdown in extracted JSON",
                    "suggested_value": None,
                }
            ],
        )

    fields = _flatten_fields(data)
    if max_fields > 0:
        fields = fields[:max_fields]

    checks: list[FieldCheck] = []
    wrong_paths: list[str] = []

    for path, value in fields:
        if not use_model_check:
            check = FieldCheck(
                path=path,
                value=value,
                verdict="unclear",
                reason="Model check disabled (--no-model-check).",
                suggested_value=None,
            )
        else:
            try:
                check = _verify_one_field(
                    client=client,
                    model=model,
                    markdown_text=markdown,
                    field_path=path,
                    field_value=value,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                check = FieldCheck(
                    path=path,
                    value=value,
                    verdict="error",
                    reason=f"Validation call failed: {exc}",
                    suggested_value=None,
                )
        checks.append(check)
        if check.verdict == "wrong":
            wrong_paths.append(check.path)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    wrong_count = sum(1 for c in checks if c.verdict == "wrong")
    unclear_count = sum(1 for c in checks if c.verdict == "unclear")
    error_count = sum(1 for c in checks if c.verdict == "error")

    return FileReport(
        file=str(file_path),
        total_leaf_fields=len(_flatten_fields(data)),
        checked_fields=len(checks),
        wrong_fields=wrong_count,
        unclear_fields=unclear_count,
        error_fields=error_count,
        wrong_paths=wrong_paths,
        details=[asdict(c) for c in checks],
    )


def _write_csv(reports: list[FileReport], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "total_leaf_fields",
                "checked_fields",
                "wrong_fields",
                "unclear_fields",
                "error_fields",
                "wrong_paths",
            ],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "file": report.file,
                    "total_leaf_fields": report.total_leaf_fields,
                    "checked_fields": report.checked_fields,
                    "wrong_fields": report.wrong_fields,
                    "unclear_fields": report.unclear_fields,
                    "error_fields": report.error_fields,
                    "wrong_paths": " | ".join(report.wrong_paths),
                }
            )


def _write_json(reports: list[FileReport], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "files": len(reports),
            "files_with_wrong": sum(1 for r in reports if r.wrong_fields > 0),
            "total_checked_fields": sum(r.checked_fields for r in reports),
            "total_wrong_fields": sum(r.wrong_fields for r in reports),
            "total_unclear_fields": sum(r.unclear_fields for r in reports),
            "total_error_fields": sum(r.error_fields for r in reports),
        },
        "files": [asdict(r) for r in reports],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated validation for extracted ACORD JSON outputs.")
    parser.add_argument(
        "--mode",
        choices=["field", "document"],
        default="document",
        help="Validation mode: field (one call per field) or document (one call per file).",
    )
    parser.add_argument(
        "--glob",
        default="data/outputs/*_extracted.json",
        help="Glob pattern for extracted JSON files.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Limit number of files to validate (0 = all).",
    )
    parser.add_argument(
        "--max-fields-per-file",
        type=int,
        default=0,
        help="Limit number of leaf fields to validate per file (0 = all).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Max tokens per validation call.",
    )
    parser.add_argument(
        "--markdown-max-chars",
        type=int,
        default=45000,
        help="Max markdown characters to send in document mode (0 = no truncation).",
    )
    parser.add_argument(
        "--json-max-chars",
        type=int,
        default=20000,
        help="Max extracted JSON characters to send in document mode (0 = no truncation).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/outputs/validation_report.csv"),
        help="Output CSV summary path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/outputs/validation_report.json"),
        help="Output JSON detailed path.",
    )
    parser.add_argument(
        "--no-model-check",
        action="store_true",
        help="Skip NuExtract field verification and run only structural field enumeration.",
    )
    parser.add_argument(
        "--api-base-url",
        default=config.API_BASE_URL,
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-model",
        default=config.API_MODEL,
        help="Model name for validation calls.",
    )
    parser.add_argument(
        "--api-key",
        default=config.API_KEY or "dummy",
        help="API key (if required).",
    )
    args = parser.parse_args()

    files = sorted(Path().glob(args.glob))
    if args.limit_files > 0:
        files = files[: args.limit_files]

    if not files:
        raise SystemExit(f"No files matched pattern: {args.glob}")

    client = OpenAI(
        base_url=args.api_base_url,
        api_key=args.api_key,
        timeout=config.API_TIMEOUT_SECONDS,
        max_retries=config.API_MAX_RETRIES,
    )

    reports: list[FileReport] = []
    for file_path in files:
        print(f"Validating {file_path} ...")
        if args.mode == "field":
            report = _validate_file(
                client=client,
                model=args.api_model,
                file_path=file_path,
                max_fields=args.max_fields_per_file,
                max_tokens=args.max_tokens,
                sleep_seconds=args.sleep_seconds,
                use_model_check=not args.no_model_check,
            )
        else:
            report = _validate_file_document_level(
                client=client,
                model=args.api_model,
                file_path=file_path,
                max_tokens=args.max_tokens,
                use_model_check=not args.no_model_check,
                markdown_max_chars=args.markdown_max_chars,
                extracted_json_max_chars=args.json_max_chars,
            )
        reports.append(report)
        print(
            f"  checked={report.checked_fields} wrong={report.wrong_fields} "
            f"unclear={report.unclear_fields} errors={report.error_fields}"
        )

    _write_csv(reports, args.out_csv)
    _write_json(reports, args.out_json)

    total_wrong = sum(r.wrong_fields for r in reports)
    total_checked = sum(r.checked_fields for r in reports)
    print("Validation complete.")
    print(f"  Files: {len(reports)}")
    print(f"  Fields checked: {total_checked}")
    print(f"  Wrong fields: {total_wrong}")
    print(f"  CSV: {args.out_csv}")
    print(f"  JSON: {args.out_json}")


if __name__ == "__main__":
    main()
