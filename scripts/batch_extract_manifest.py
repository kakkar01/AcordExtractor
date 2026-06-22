"""Batch run extraction using a manifest CSV.

Reads a CSV with columns: index,source_uri,local_pdf,output_json,exit_code
and runs `python main.py extract <local_pdf> --form "ACORD 125" --markdown` for each row.
Updates the `exit_code` field with the process exit code.

Usage:
    python scripts/batch_extract_manifest.py --manifest data/outputs/run_manifest.csv

Options:
    --manifest PATH   Path to manifest CSV (default: data/outputs/run_manifest.csv)
    --form NAME       ACORD form name to pass to main.py (default: "ACORD 125")
    --parallel N      Run up to N workers in parallel (default: 1)
    --no-save         Pass --no-save to main.py to avoid writing outputs

"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shlex
import urllib.request
import urllib.parse
import os

DEFAULT_MANIFEST = Path("data/outputs/run_manifest.csv")


def run_row(row: dict, form: str, no_save: bool) -> int:
    pdf = row.get("local_pdf")
    if not pdf:
        return 2
    cmd = [sys.executable, "main.py", "extract", pdf, "--form", form, "--markdown"]
    if no_save:
        cmd.append("--no-save")
    # Use subprocess.run to preserve environment; return exit code
    proc = subprocess.run(cmd)
    return proc.returncode


def _download_row_if_needed(row: dict, forms_dir: Path, force: bool = False) -> None:
    """Download `source_uri` to `local_pdf` if missing or forced.

    Updates row['local_pdf'] in-place with the downloaded path.
    """
    src = (row.get("source_uri") or "").strip()
    local = (row.get("local_pdf") or "").strip()

    # If no source URI, nothing to download
    if not src:
        return

    # Determine destination filename and optional policy subdir from URL path
    parsed = urllib.parse.urlparse(src)
    path_parts = [p for p in parsed.path.split("/") if p]

    # policy_id candidate = first meaningful segment (often the policy folder)
    policy_id = None
    if path_parts:
        policy_id = path_parts[0]

    # Build filename: prefer explicit local name, otherwise join last up-to-4 segments
    if local:
        dest_name = os.path.basename(local)
    else:
        tail = path_parts[-4:] if len(path_parts) >= 4 else path_parts
        dest_name = "_".join(tail)
        if not dest_name:
            dest_name = f"download_{row.get('index','')}.pdf"
        # ensure extension
        if not dest_name.lower().endswith('.pdf'):
            dest_name = dest_name + '.pdf'

    # Create a policy-specific subdirectory when possible
    if policy_id:
        dest_dir = forms_dir / policy_id
    else:
        dest_dir = forms_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / dest_name

    if dest_path.exists() and not force:
        # already present; update manifest to point to it
        row["local_pdf"] = str(dest_path)
        return

    try:
        print(f"Downloading {src} → {dest_path}")
        urllib.request.urlretrieve(src, dest_path)
        row["local_pdf"] = str(dest_path)
    except Exception as exc:
        print(f"Failed to download {src}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--form", type=str, default="ACORD 125")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--download", action="store_true", help="Download missing PDFs from source_uri before running")
    parser.add_argument("--force-download", action="store_true", help="Always re-download PDFs even if present locally")
    args = parser.parse_args()

    manifest_path: Path = args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    rows: list[dict] = []
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for r in reader:
            rows.append(r)

    if not rows:
        print("No rows in manifest.")
        return

    # Prepare results mapping
    results: dict[int, int] = {}

    forms_dir = Path("data/forms/Acord140")

    # Pre-download rows if requested (or update missing local_pdf entries)
    if args.download or args.force_download:
        for row in rows:
            _download_row_if_needed(row, forms_dir, force=args.force_download)

    if args.parallel and args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            future_to_idx = {}
            for i, row in enumerate(rows):
                future = ex.submit(run_row, row, args.form, args.no_save)
                future_to_idx[future] = i
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    exit_code = future.result()
                except Exception:
                    exit_code = 1
                results[i] = exit_code
    else:
        for i, row in enumerate(rows):
            exit_code = run_row(row, args.form, args.no_save)
            results[i] = exit_code

    # Write updated manifest (overwrite)
    out_path = manifest_path
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for i, row in enumerate(rows):
            row_copy = dict(row)
            row_copy["exit_code"] = str(results.get(i, 1))
            writer.writerow(row_copy)

    print(f"Finished. Updated manifest written to {out_path}")


if __name__ == "__main__":
    main()
