"""
main.py — CLI entry point for ACORD form extraction.

Examples
--------
# Extract using the built-in ACORD 25 schema
python main.py extract data/forms/cert.pdf --form "ACORD 25"

# Extract with reasoning mode enabled (harder documents)
python main.py extract data/forms/cert.pdf --form "ACORD 25" --thinking

# Auto-generate a template then extract
python main.py extract data/forms/cert.pdf

# OCR only (document → Markdown)
python main.py ocr data/forms/cert.pdf

# Generate and print a template for a given form type
python main.py generate-template --form "ACORD 130"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.logging_fallback import logger

import config
from src.api_extractor import ApiNuExtractor
from src.model_manager import load_model
from src.pipeline import AcordPipeline
from src.template_generator import TemplateGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acord-extractor",
        description="ACORD form extraction with NuExtract3 (local or API backend).",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "api"],
        default=config.INFERENCE_BACKEND,
        help="Inference backend. Defaults to INFERENCE_BACKEND env var (or local).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── extract ───────────────────────────────────────────────────────────────
    p_extract = sub.add_parser("extract", help="Extract structured data from a form.")
    p_extract.add_argument("file", type=Path, help="Path to the PDF or image file.")
    p_extract.add_argument(
        "--form",
        type=str,
        default=None,
        help='ACORD form name, e.g. "ACORD 25". Auto-generates if omitted.',
    )
    p_extract.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Optional freeform instructions for the model.",
    )
    p_extract.add_argument(
        "--thinking",
        action="store_true",
        help="Enable chain-of-thought reasoning (slower, better for complex layouts).",
    )
    p_extract.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the result to disk.",
    )
    p_extract.add_argument(
        "--markdown",
        action="store_true",
        help="Also run OCR and include the raw Markdown in the output JSON as '_markdown'.",
    )

    # ── ocr ───────────────────────────────────────────────────────────────────
    p_ocr = sub.add_parser("ocr", help="Convert form to Markdown (OCR mode).")
    p_ocr.add_argument("file", type=Path, help="Path to the PDF or image file.")
    p_ocr.add_argument(
        "--thinking",
        action="store_true",
        help="Enable chain-of-thought reasoning.",
    )
    p_ocr.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the Markdown to disk.",
    )

    # ── generate-template ─────────────────────────────────────────────────────
    p_gen = sub.add_parser(
        "generate-template",
        help="Generate a NuExtract template from a description.",
    )
    group = p_gen.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--form",
        type=str,
        help='Use a built-in ACORD description, e.g. "ACORD 25".',
    )
    group.add_argument(
        "--description",
        type=str,
        help="Free-form description of the document and desired fields.",
    )
    p_gen.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the generated template JSON.",
    )

    return parser


def cmd_extract(args: argparse.Namespace, pipeline: AcordPipeline) -> None:
    result = pipeline.run(
        file_path=args.file,
        form_name=args.form,
        instructions=args.instructions,
        enable_thinking=args.thinking,
        save_output=not args.no_save,
        include_markdown=args.markdown,
    )
    output = json.dumps(result, indent=2, ensure_ascii=False)
    if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp1252"):
        sys.stdout.buffer.write(output.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
    else:
        print(output)


def cmd_ocr(args: argparse.Namespace, pipeline: AcordPipeline) -> None:
    markdown = pipeline.run_ocr(
        file_path=args.file,
        enable_thinking=args.thinking,
        save_output=not args.no_save,
    )
    print(markdown)


def cmd_generate_template(
    args: argparse.Namespace,
    extractor: NuExtractor,
) -> None:
    gen = TemplateGenerator(extractor)

    if args.form:
        template = gen.generate_for_acord(args.form)
    else:
        template = gen.generate(args.description)

    pretty = json.dumps(template, indent=2, ensure_ascii=False)
    print(pretty)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(pretty, encoding="utf-8")
        logger.info(f"Template saved → {args.out}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.backend == "api":
        extractor = ApiNuExtractor(
            base_url=config.API_BASE_URL,
            api_key=config.API_KEY,
            model=config.API_MODEL,
        )
        pipeline = AcordPipeline(extractor=extractor)
        logger.info(f"Using API backend ({config.API_MODEL}) at {config.API_BASE_URL}")
    else:
        model, processor = load_model()
        from src.extractor import NuExtractor

        extractor = NuExtractor(model, processor)
        pipeline = AcordPipeline(model=model, processor=processor)
        logger.info("Using local Transformers backend")

    if args.command == "extract":
        cmd_extract(args, pipeline)
    elif args.command == "ocr":
        cmd_ocr(args, pipeline)
    elif args.command == "generate-template":
        cmd_generate_template(args, extractor)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
