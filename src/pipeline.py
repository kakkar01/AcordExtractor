"""
End-to-end ACORD form extraction pipeline.

Accepts a PDF or image file, renders it to PIL images (multi-page aware),
picks the right ACORD template (or auto-detects via template-generation),
and runs NuExtract3 locally.

Usage example
-------------
    from src.model_manager import load_model
    from src.pipeline import AcordPipeline
    from schemas import ACORD_SCHEMAS

    model, processor = load_model()
    pipeline = AcordPipeline(model, processor)

    result = pipeline.run(
        file_path="data/forms/sample_acord25.pdf",
        form_name="ACORD 25",          # or None to auto-detect
        enable_thinking=False,
    )
    print(result)
"""

from __future__ import annotations

import difflib
from copy import deepcopy
import html
import json
from pathlib import Path
import re
from datetime import datetime
import time
from typing import Any

from src.logging_fallback import logger
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import config
from schemas import ACORD_SCHEMAS, get_page_templates
from src.docling_ocr import DoclingOCR
from src.extractor import NuExtractor
from src.template_generator import TemplateGenerator


class AcordPipeline:
    """
    Full extraction pipeline for ACORD forms.

    Parameters
    ----------
    model, processor
        Loaded NuExtract3 model and processor (from ``src.model_manager``).
    """

    def __init__(
        self,
        model: AutoModelForImageTextToText | None = None,
        processor: AutoProcessor | None = None,
        extractor: Any | None = None,
    ) -> None:
        if extractor is not None:
            self.extractor = extractor
        elif model is not None and processor is not None:
            self.extractor = NuExtractor(model, processor)
        else:
            raise ValueError(
                "Provide either (model and processor) or an extractor instance."
            )
        self.docling_ocr = DoclingOCR.from_config()
        self.template_gen = TemplateGenerator(self.extractor)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        file_path: str | Path,
        form_name: str | None = None,
        template: dict[str, Any] | None = None,
        instructions: str | None = None,
        enable_thinking: bool = False,
        save_output: bool = True,
        include_markdown: bool = False,
    ) -> dict[str, Any]:
        """
        Extract structured data from an ACORD form file.

        Parameters
        ----------
        file_path : str | Path
            Path to a PDF, PNG, JPG, or TIFF file.
        form_name : str | None
            ACORD form name, e.g. ``"ACORD 25"``.  If *None* and no *template*
            is given, the pipeline auto-generates a template via
            template-generation mode.
        template : dict | None
            Override template dict.  Takes precedence over *form_name*.
        instructions : str | None
            Optional extraction instructions forwarded to the model.
        enable_thinking : bool
            Use chain-of-thought reasoning (better accuracy, slower).
        save_output : bool
            Persist the result as JSON in ``config.OUTPUTS_DIR``.
        include_markdown : bool
            If *True*, also run OCR on the document and store the resulting
            Markdown string under the ``"_markdown"`` key in the output JSON.

        Returns
        -------
        dict
            Extracted data.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Processing: {file_path.name}")

        # 2. Resolve template
        resolved_template = self._resolve_template(
            template=template,
            form_name=form_name,
        )
        page_templates = self._resolve_page_templates(form_name=form_name)

        # 3. Extract
        if getattr(self.extractor, "supports_image_inputs", True):
            images = self._load_images(file_path)
            logger.info(f"  {len(images)} page(s) loaded.")
            # Enforce ACORD 140 policy: even if the PDF has more pages,
            # only send the first two pages to the extractor when the
            # resolved form is ACORD 140 (user requirement).
            if form_name and form_name.strip().upper() == "ACORD 140" and len(images) > 2:
                logger.info("  ACORD 140 detected: limiting pages to first 2 as requested.")
                images = images[:2]
            if getattr(self.extractor, "prefer_page_wise", False) and len(images) > 1:
                page_results: list[dict[str, Any]] = []
                for index, image in enumerate(images, start=1):
                    current_template = self._template_for_page(
                        page_templates=page_templates,
                        page_number=index,
                        default_template=resolved_template,
                    )
                    current_instructions = self._resolve_instructions(
                        form_name=form_name,
                        page_number=index,
                        explicit_instructions=instructions,
                    )
                    logger.info(f"  Extracting page {index}/{len(images)} via API image mode...")
                    started_at = time.perf_counter()
                    try:
                        page_results.append(
                            self.extractor.extract(
                                images=[image],
                                template=current_template,
                                instructions=current_instructions,
                                enable_thinking=enable_thinking,
                            )
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"API extraction failed on page {index}/{len(images)} in image mode. "
                            f"Backend '{config.API_BASE_URL}' may be unavailable or overloaded."
                        ) from exc
                    finally:
                        elapsed = time.perf_counter() - started_at
                        logger.info(f"  Page {index} API image extraction took {elapsed:.1f}s")
                result = self._merge_page_results(page_results)
            else:
                current_instructions = self._resolve_instructions(
                    form_name=form_name,
                    page_number=1,
                    explicit_instructions=instructions,
                )
                result = self.extractor.extract(
                    images=images,
                    template=resolved_template,
                    instructions=current_instructions,
                    enable_thinking=enable_thinking,
                )

            # Focused second-pass refinement for ACORD 125 page-1 LOB checkboxes.
           # if form_name and form_name.strip().upper() == "ACORD 125" and images:
             #   result = self._refine_acord125_lob_selection(
               #     page1_image=images[0],
               #     result=result,
               #     source_name=file_path.stem,
             #   )

            # Optional OCR pass — attach raw Markdown to the output.
            if include_markdown:
                try:
                    logger.info("  Running OCR pass for _markdown field...")
                    ocr_payload = self._ocr_with_metadata(
                        file_path=file_path,
                        images=images,
                        enable_thinking=enable_thinking,
                    )
                    result["_markdown"] = ocr_payload["markdown"]
                    if ocr_payload.get("key_value_pairs"):
                        result["_ocr_key_values"] = ocr_payload["key_value_pairs"]
                    if form_name and form_name.strip().upper() == "ACORD 125":
                        result = self._reconcile_from_ocr_key_values(result)
                        result = self._reconcile_lob_selection_from_markdown(result)
                        result = self._reconcile_attachments_from_markdown(result)
                        result = self._reconcile_status_of_transaction_from_markdown(result)
                        result = self._reconcile_nature_of_business_from_markdown(result)
                        result = self._reconcile_location_primary_operations(result)
                        result = self._reconcile_location_interest_from_markdown(result)
                        result = self._reconcile_acord125_locations_from_markdown(result)
                        result = self._validate_lob_with_nuextract(result, images, enable_thinking)
                        result = self._resolve_ambiguous_markdown_fields_with_nuextract(result, images, enable_thinking)
                        result = self._reconcile_billing_plan_from_markdown(result)
                    if form_name and form_name.strip().upper() == "ACORD 140":
                        result = self._reconcile_acord140_from_markdown_and_key_values(result)
                    if form_name and form_name.strip().upper() == "ACORD 126":
                        result = self._reconcile_acord126_checkboxes(result)
                except Exception as exc:
                    logger.warning(f"OCR pass failed; _markdown will be omitted: {exc}")
        else:
            if file_path.suffix.lower() == ".pdf":
                page_texts = self._load_pdf_text_pages(file_path)
                logger.info(
                    f"  Loaded {len(page_texts)} page(s) of text for API extraction."
                )
                # Enforce ACORD 140 policy for text-mode extraction too.
                if form_name and form_name.strip().upper() == "ACORD 140" and len(page_texts) > 2:
                    logger.info("  ACORD 140 detected: limiting text pages to first 2 as requested.")
                    page_texts = page_texts[:2]
                page_results: list[dict[str, Any]] = []
                for index, page_text in enumerate(page_texts, start=1):
                    current_template = self._template_for_page(
                        page_templates=page_templates,
                        page_number=index,
                        default_template=resolved_template,
                    )
                    current_instructions = self._resolve_instructions(
                        form_name=form_name,
                        page_number=index,
                        explicit_instructions=instructions,
                    )
                    prepared_text = self._prepare_page_text_for_api(page_text)
                    logger.info(f"  Extracting page {index}/{len(page_texts)} via API...")
                    started_at = time.perf_counter()
                    try:
                        page_results.append(
                            self._extract_page_with_retry(
                                text=prepared_text,
                                template=current_template,
                                instructions=current_instructions,
                                enable_thinking=enable_thinking,
                                file_path=file_path,
                                page_number=index,
                            )
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"API extraction failed on page {index}/{len(page_texts)} in text mode. "
                            f"Backend '{config.API_BASE_URL}' may be unavailable or overloaded."
                        ) from exc
                    finally:
                        elapsed = time.perf_counter() - started_at
                        logger.info(f"  Page {index} API text extraction took {elapsed:.1f}s")
                result = self._merge_page_results(page_results)
            else:
                text = self._load_text(file_path)
                logger.info(f"  Loaded {len(text)} text characters for API extraction.")
                current_instructions = self._resolve_instructions(
                    form_name=form_name,
                    page_number=1,
                    explicit_instructions=instructions,
                )
                result = self.extractor.extract(
                    text=text,
                    template=resolved_template,
                    instructions=current_instructions,
                    enable_thinking=enable_thinking,
                )

        result = self._add_validation_warnings(form_name=form_name, result=result)

        # Calculate and add extraction accuracy
        accuracy = self._calculate_accuracy(result)
        result["_extraction_accuracy"] = accuracy

        # Optional per-field reasoning pass (adds `_field_reasons`)
        try:
            result = self._add_field_reasoning(images if getattr(self.extractor, "supports_image_inputs", True) else None, result)
        except Exception:
            # Be tolerant: reasoning is an optional debug aid and must not break extraction
            logger.debug("Per-field reasoning pass failed or was skipped.")

        # 4. Optionally persist
        if save_output:
            out_path = self._save(file_path, result)
            logger.info(f"  Result saved → {out_path}")

        return result

    def _refine_acord125_lob_selection(
        self,
        page1_image: Image.Image,
        result: dict[str, Any],
        source_name: str,
    ) -> dict[str, Any]:
        """Run a focused pass that extracts only checked line-of-business rows."""
        lines = result.get("lines_of_business")
        if not isinstance(lines, list) or not lines:
            return result

        lob_image = page1_image
        if config.LOB_REFINEMENT_CROP:
            lob_image = self._crop_lob_region(page1_image)

        crop_path: Path | None = None
        if config.LOB_SAVE_CROP_IMAGE:
            crop_path = self._save_lob_crop_image(lob_image, source_name)

        canonical_lobs = [
            item.get("lob")
            for item in lines
            if isinstance(item, dict) and isinstance(item.get("lob"), str) and item.get("lob").strip()
        ]
        if not canonical_lobs:
            return result

        refine_template = {
            "checked_lobs": [
                {
                    "lob": "verbatim-string",
                }
            ]
        }
        refine_instructions = self._build_lob_refinement_prompt()

        try:
            refine_result = self.extractor.extract(
                images=[lob_image],
                template=refine_template,
                instructions=refine_instructions,
                enable_thinking=False,
            )
        except Exception as exc:
            logger.warning(f"LOB refinement pass failed: {exc}")
            return result

        checked = refine_result.get("checked_lobs") if isinstance(refine_result, dict) else None
        if not isinstance(checked, list):
            return result

        normalized_to_canonical: dict[str, str] = {
            self._normalize_lob_name(name): name
            for name in canonical_lobs
        }

        matched: set[str] = set()
        for item in checked:
            if not isinstance(item, dict):
                continue
            raw_name = item.get("lob")
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            normalized = self._normalize_lob_name(raw_name)
            if not normalized:
                continue

            if normalized in normalized_to_canonical:
                matched.add(normalized_to_canonical[normalized])
                continue

            candidates = difflib.get_close_matches(
                normalized,
                normalized_to_canonical.keys(),
                n=1,
                cutoff=0.9,
            )
            if candidates:
                candidate_name = normalized_to_canonical[candidates[0]]
                if self._lob_names_compatible(raw_name, candidate_name):
                    matched.add(candidate_name)

        if not matched:
            return result

        refined_lines: list[dict[str, Any]] = []
        for item in lines:
            if not isinstance(item, dict):
                refined_lines.append(item)
                continue
            lob_name = item.get("lob")
            if isinstance(lob_name, str) and lob_name in matched:
                updated = dict(item)
                updated["selected"] = True
                refined_lines.append(updated)
            else:
                updated = dict(item)
                updated["selected"] = False
                refined_lines.append(updated)

        output = deepcopy(result)
        output["lines_of_business"] = refined_lines
        output["_lob_refinement_debug"] = {
            "crop_used": bool(config.LOB_REFINEMENT_CROP),
            "crop_image_path": str(crop_path) if crop_path else None,
            "checked_lobs_model": checked,
            "matched_lobs": sorted(matched),
        }
        return output

    def _add_field_reasoning(
        self,
        images: list[Image.Image] | None,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Optionally run a small reasoning pass per-field to return short explanations.

        This is bounded by `config.API_FIELD_REASONING_MAX` and guarded by
        `config.API_FIELD_REASONING`.
        """
        try:
            import config as _config
        except Exception:
            return result

        if not getattr(_config, "API_FIELD_REASONING", False):
            return result

        # In API completion mode we do not have page images here; avoid no-op loops.
        if images is None:
            return result

        max_fields = getattr(_config, "API_FIELD_REASONING_MAX", 12)

        # Only run reasoning for lines_of_business entries (per user request)
        lob_list = result.get("lines_of_business")
        if not isinstance(lob_list, list) or not lob_list:
            return result

        reasons: dict[str, str] = {}
        for i, lob_item in enumerate(lob_list):
            if not isinstance(lob_item, dict):
                continue
            if len(reasons) >= max_fields:
                break

            lob_label = lob_item.get("lob")
            selected = lob_item.get("selected")
            path = f"lines_of_business[{i}].selected"

            tmpl = {"explanation": "string"}
            instr = (
                f"Explain briefly why the line-of-business '{lob_label}' was marked '{selected}'. "
                "Refer only to visible checkbox marks or the exact printed label in the LINES OF BUSINESS table. "
                "If the checkbox is unclear, say 'unclear'. Return only JSON like {\"explanation\": \"...\"}."
            )
            try:
                imgs = [images[0]] if images else None
                res = self.extractor.extract(
                    images=imgs,
                    template=tmpl,
                    instructions=instr,
                    enable_thinking=True,
                )
                if isinstance(res, dict) and "explanation" in res and isinstance(res["explanation"], str):
                    reasons[path] = res["explanation"]
            except Exception as exc:  # pragma: no cover - optional debug path
                logger.warning(f"LOB reasoning failed for {lob_label}: {exc}")

        if reasons:
            out = deepcopy(result)
            out.setdefault("_field_reasons", {}).update(reasons)
            return out

        return result

    @staticmethod
    def _crop_lob_region(page1_image: Image.Image) -> Image.Image:
        """Crop to the ACORD 125 page-1 LINES OF BUSINESS area using ratio bounds."""
        width, height = page1_image.size
        left = int(max(0.0, min(1.0, config.LOB_CROP_LEFT_RATIO)) * width)
        top = int(max(0.0, min(1.0, config.LOB_CROP_TOP_RATIO)) * height)
        right = int(max(0.0, min(1.0, config.LOB_CROP_RIGHT_RATIO)) * width)
        bottom = int(max(0.0, min(1.0, config.LOB_CROP_BOTTOM_RATIO)) * height)

        if right <= left or bottom <= top:
            return page1_image

        return page1_image.crop((left, top, right, bottom))

    @staticmethod
    def _save_lob_crop_image(crop_image: Image.Image, source_name: str) -> Path | None:
        """Persist cropped LOB image to disk for visual QA/debugging."""
        try:
            out_dir = config.LOB_CROP_IMAGES_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = out_dir / f"{source_name}_lob_crop_{stamp}.jpg"
            crop_image.save(out_path, format="JPEG", quality=95, optimize=True, subsampling=0)
            return out_path
        except Exception as exc:  # pragma: no cover - debug-only path
            logger.warning(f"Could not save LOB crop image: {exc}")
            return None

    @staticmethod
    def _build_lob_refinement_prompt() -> str:
        return (
            "Read only the cropped LINES OF BUSINESS table from ACORD 125 page 1. "
            "Your task is not full extraction. Your only task is to list the row labels whose own checkbox is clearly marked. "
            "A row is checked only if that row's checkbox itself has a visible X, checkmark, or filled mark. "
            "Do not infer a checked row from nearby rows, table borders, printed text, or premium columns. "
            "If a mark is ambiguous, faint, clipped, or could be a table line, treat that row as unchecked. "
            "Positive example output: {\"checked_lobs\":[{\"lob\":\"Business Auto\"}]}. "
            "Negative example: do not return Boiler & Machinery just because Business Auto is checked or because a crossing table line looks like an X. "
            "Return JSON only matching the template."
        )

    @staticmethod
    def _normalize_lob_name(value: str) -> str:
        lowered = value.lower().replace("&", "and")
        normalized = re.sub(r"[^a-z0-9]+", "", lowered)
        if normalized == "commercial":
            return "commercialgeneralliability"
        if normalized == "commercialgeneral":
            return "commercialgeneralliability"
        return normalized

    @staticmethod
    def _lob_names_compatible(left: str, right: str) -> bool:
        """Conservative guard for fuzzy matches to reduce false-positive LOB mapping."""
        left_tokens = AcordPipeline._lob_tokens(left)
        right_tokens = AcordPipeline._lob_tokens(right)
        if not left_tokens or not right_tokens:
            return False

        overlap = left_tokens.intersection(right_tokens)
        if not overlap:
            return False

        # Require at least one strong token overlap and avoid single short-token matches.
        return any(len(token) >= 5 for token in overlap)

    @staticmethod
    def _lob_tokens(value: str) -> set[str]:
        normalized = value.lower().replace("&", " and ")
        tokens = re.findall(r"[a-z0-9]+", normalized)
        stop = {"and", "of", "the", "flo"}
        return {t for t in tokens if t not in stop}

    @staticmethod
    def _reconcile_lob_selection_from_markdown(result: dict[str, Any]) -> dict[str, Any]:
        """Use row-level X markers in markdown LOB table to prevent cross-column checkbox bleed."""
        markdown = result.get("_markdown")
        lines = result.get("lines_of_business")
        if not isinstance(markdown, str) or not isinstance(lines, list) or not lines:
            return result

        section_match = re.search(
            r"(?:##\s+)?(?:INDCATE\s+)?(?:SECTIONS\s*ATTACHED|SECTION\s*ATTACHED|LINES\s*OF\s*BUSINESS|LINESOFBUSINESS)\b(.*?)(?:##\s+ATTACHMENTS\b|##\s+STATUS\b|##\s+PACKAGE\b|$)",
            markdown,
            flags=re.IGNORECASE | re.DOTALL,
        )
        current_selected = {
            AcordPipeline._normalize_lob_name(str(item.get("lob")))
            for item in lines
            if isinstance(item, dict) and item.get("selected") is True and isinstance(item.get("lob"), str)
        }

        if not section_match:
            return AcordPipeline._record_markdown_verification(
                result,
                field="lines_of_business",
                status="ambiguous",
                docling_selected=[],
                output_selected=sorted(current_selected),
            )

        section = section_match.group(1)
        row_blocks = re.findall(r"<tr>(.*?)</tr>", section, flags=re.IGNORECASE | re.DOTALL)

        normalized_to_index: dict[str, int] = {}
        for idx, item in enumerate(lines):
            if not isinstance(item, dict):
                continue
            lob = item.get("lob")
            if isinstance(lob, str) and lob.strip():
                normalized_to_index[AcordPipeline._normalize_lob_name(lob)] = idx

        if not normalized_to_index:
            return AcordPipeline._record_markdown_verification(
                result,
                field="lines_of_business",
                status="ambiguous",
                docling_selected=[],
                output_selected=sorted(current_selected),
            )

        # Only override rows where at least one explicit marker exists.
        row_overrides: dict[int, bool] = {}
        marker_re = re.compile(r"^\s*(?:\u2612|\u2611|\[[xX]\]|\([xX]\)|[xX])\s+(.+)$")

        if row_blocks:
            for row in row_blocks:
                cell_values = [
                    re.sub(r"<.*?>", "", cell).strip()
                    for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.IGNORECASE | re.DOTALL)
                ]
                if not cell_values:
                    continue

                mapped_cells: list[tuple[int, bool]] = []
                has_explicit_marker = False
                for raw_cell in cell_values:
                    if not raw_cell or raw_cell == "$":
                        continue

                    marked = False
                    label = raw_cell
                    marker_match = marker_re.match(raw_cell)
                    if marker_match:
                        has_explicit_marker = True
                        marked = True
                        label = marker_match.group(1).strip()

                    normalized_label = AcordPipeline._normalize_lob_name(label)
                    if not normalized_label:
                        continue

                    idx = normalized_to_index.get(normalized_label)
                    if idx is None:
                        candidates = difflib.get_close_matches(
                            normalized_label,
                            normalized_to_index.keys(),
                            n=1,
                            cutoff=0.9,
                        )
                        if candidates:
                            idx = normalized_to_index[candidates[0]]

                    if idx is not None:
                        mapped_cells.append((idx, marked))

                if has_explicit_marker:
                    for idx, marked in mapped_cells:
                        row_overrides[idx] = marked

        if not row_overrides:
            row_overrides = AcordPipeline._extract_checkbox_overrides_from_section(
                section,
                normalized_to_index,
                allow_bare_x=True,
            )

        if not row_overrides:
            return AcordPipeline._record_markdown_verification(
                result,
                field="lines_of_business",
                status="ambiguous",
                docling_selected=[],
                output_selected=sorted(current_selected),
            )

        docling_selected = {
            AcordPipeline._normalize_lob_name(str(lines[idx].get("lob")))
            for idx, marked in row_overrides.items()
            if marked and 0 <= idx < len(lines) and isinstance(lines[idx], dict) and isinstance(lines[idx].get("lob"), str)
        }

        if not docling_selected:
            return AcordPipeline._record_markdown_verification(
                result,
                field="lines_of_business",
                status="ambiguous",
                docling_selected=[],
                output_selected=sorted(current_selected),
            )

        if docling_selected == current_selected:
            return AcordPipeline._record_markdown_verification(
                result,
                field="lines_of_business",
                status="verified",
                docling_selected=sorted(docling_selected),
                output_selected=sorted(current_selected),
            )

        out = deepcopy(result)
        out_lines = out.get("lines_of_business")
        if not isinstance(out_lines, list):
            return result

        for idx, marked in row_overrides.items():
            if 0 <= idx < len(out_lines) and isinstance(out_lines[idx], dict):
                out_lines[idx]["selected"] = marked

        corrected_selected = {
            AcordPipeline._normalize_lob_name(str(item.get("lob")))
            for item in out_lines
            if isinstance(item, dict) and item.get("selected") is True and isinstance(item.get("lob"), str)
        }
        return AcordPipeline._record_markdown_verification(
            out,
            field="lines_of_business",
            status="corrected",
            docling_selected=sorted(docling_selected),
            output_selected=sorted(corrected_selected),
        )

    @staticmethod
    def _reconcile_attachments_from_markdown(result: dict[str, Any]) -> dict[str, Any]:
        """Use explicit markdown checkbox marks to correct ACORD 125 attachments."""
        markdown = result.get("_markdown")
        attachments = result.get("attachments")
        if not isinstance(markdown, str) or not isinstance(attachments, list) or not attachments:
            return result

        current_selected = {
            AcordPipeline._normalize_markdown_checkbox_label(str(item.get("attachment_type")))
            for item in attachments
            if isinstance(item, dict) and item.get("selected") is True and isinstance(item.get("attachment_type"), str)
        }

        section_match = re.search(
            r"##\s+ATTACHMENTS\b(.*?)(?:##\s+|$)",
            markdown,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not section_match:
            if current_selected:
                out = deepcopy(result)
                out_attachments = out.get("attachments")
                if isinstance(out_attachments, list):
                    for item in out_attachments:
                        if isinstance(item, dict):
                            item["selected"] = False
                return AcordPipeline._record_markdown_verification(
                    out,
                    field="attachments",
                    status="corrected",
                    docling_selected=[],
                    output_selected=[],
                )
            return AcordPipeline._record_markdown_verification(
                result,
                field="attachments",
                status="ambiguous",
                docling_selected=[],
                output_selected=sorted(current_selected),
            )

        section = section_match.group(1)
        valid_labels = {
            AcordPipeline._normalize_markdown_checkbox_label(str(item.get("attachment_type")))
            for item in attachments
            if isinstance(item, dict) and isinstance(item.get("attachment_type"), str)
        }
        if not valid_labels:
            if current_selected:
                out = deepcopy(result)
                out_attachments = out.get("attachments")
                if isinstance(out_attachments, list):
                    for item in out_attachments:
                        if isinstance(item, dict):
                            item["selected"] = False
                return AcordPipeline._record_markdown_verification(
                    out,
                    field="attachments",
                    status="corrected",
                    docling_selected=[],
                    output_selected=[],
                )
            return AcordPipeline._record_markdown_verification(
                result,
                field="attachments",
                status="ambiguous",
                docling_selected=[],
                output_selected=sorted(current_selected),
            )

        docling_selected = AcordPipeline._extract_selected_checkbox_labels(section, valid_labels)
        if not docling_selected:
            docling_selected = AcordPipeline._extract_checkbox_labels_from_text(section, valid_labels)
        if not docling_selected:
            return AcordPipeline._record_markdown_verification(
                result,
                field="attachments",
                status="ambiguous",
                docling_selected=[],
                output_selected=sorted(current_selected),
            )

        if current_selected == docling_selected:
            return AcordPipeline._record_markdown_verification(
                result,
                field="attachments",
                status="verified",
                docling_selected=sorted(docling_selected),
                output_selected=sorted(current_selected),
            )

        out = deepcopy(result)
        out_attachments = out.get("attachments")
        if not isinstance(out_attachments, list):
            return result

        for item in out_attachments:
            if not isinstance(item, dict):
                continue
            attachment_type = item.get("attachment_type")
            if isinstance(attachment_type, str):
                normalized = AcordPipeline._normalize_markdown_checkbox_label(attachment_type)
                item["selected"] = normalized in docling_selected

        corrected_selected = {
            AcordPipeline._normalize_markdown_checkbox_label(str(item.get("attachment_type")))
            for item in out_attachments
            if isinstance(item, dict) and item.get("selected") is True and isinstance(item.get("attachment_type"), str)
        }
        return AcordPipeline._record_markdown_verification(
            out,
            field="attachments",
            status="corrected",
            docling_selected=sorted(docling_selected),
            output_selected=sorted(corrected_selected),
        )

    @staticmethod
    def _reconcile_status_of_transaction_from_markdown(result: dict[str, Any]) -> dict[str, Any]:
        """Verify ACORD policy status_of_transaction checkboxes from markdown."""
        markdown = result.get("_markdown")
        policy_header = result.get("policy_header")
        if not isinstance(markdown, str) or not isinstance(policy_header, dict):
            return result

        status_items = policy_header.get("status_of_transaction")
        if not isinstance(status_items, list) or not status_items:
            return result

        valid_labels = {
            "quote",
            "bound",
            "issue policy",
            "renew",
            "change",
            "cancel",
        }

        def normalize_status_label(value: str) -> str:
            normalized = AcordPipeline._normalize_markdown_checkbox_label(value)
            return "bound" if normalized == "bind" else normalized

        current_selected: set[str] = set()
        for item in status_items:
            if not isinstance(item, dict):
                continue
            for name, value in item.items():
                if value is True:
                    normalized = normalize_status_label(str(name))
                    if normalized in valid_labels:
                        current_selected.add(normalized)

        section_match = re.search(
            r"(?:STATUSOF\s*TRANSACTION|TRANSACTION\s*STATUSOF|STATUS\s*OF\s*TRANSACTION)(.*?)(?:##\s+|$)",
            markdown,
            flags=re.IGNORECASE | re.DOTALL,
        )
        section = section_match.group(1) if section_match else ""
        docling_selected = AcordPipeline._extract_selected_checkbox_labels(section, valid_labels)
        if not docling_selected:
            docling_selected = AcordPipeline._extract_checkbox_labels_from_text(section, valid_labels)

        if section_match:
            pre_section = markdown[: section_match.start()]
            preceding_lines = [line.strip() for line in pre_section.splitlines() if line.strip()]
            if preceding_lines:
                pre_section_snippet = "\n".join(preceding_lines[-20:])
                docling_selected |= AcordPipeline._extract_checkbox_labels_from_text(
                    pre_section_snippet,
                    valid_labels,
                )

        if not docling_selected:
            # Some OCRs place the actual QUOTE/RENEW checkbox block before or after
            # the STATUSOF TRANSACTION label, so search the full markdown as a fallback.
            docling_selected = AcordPipeline._extract_selected_checkbox_labels(markdown, valid_labels)
            if not docling_selected:
                docling_selected = AcordPipeline._extract_checkbox_labels_from_text(markdown, valid_labels)

        if not docling_selected:
            if section_match:
                out = deepcopy(result)
                out_policy_header = out.get("policy_header")
                if isinstance(out_policy_header, dict):
                    out_status_items = out_policy_header.get("status_of_transaction")
                    if isinstance(out_status_items, list):
                        for status_item in out_status_items:
                            if isinstance(status_item, dict):
                                for name in list(status_item.keys()):
                                    normalized = AcordPipeline._normalize_markdown_checkbox_label(str(name))
                                    if normalized in valid_labels:
                                        status_item[name] = False
                return AcordPipeline._record_markdown_verification(
                    out,
                    field="status_of_transaction",
                    status="corrected",
                    docling_selected=[],
                    output_selected=[],
                )
            return AcordPipeline._record_markdown_verification(
                result,
                field="status_of_transaction",
                status="ambiguous",
                docling_selected=[],
                output_selected=sorted(current_selected),
            )

        if docling_selected == current_selected:
            return AcordPipeline._record_markdown_verification(
                result,
                field="status_of_transaction",
                status="verified",
                docling_selected=sorted(docling_selected),
                output_selected=sorted(current_selected),
            )

        out = deepcopy(result)
        out_policy_header = out.get("policy_header")
        if not isinstance(out_policy_header, dict):
            return result

        out_status_items = out_policy_header.get("status_of_transaction")
        if not isinstance(out_status_items, list):
            return result

        for status_item in out_status_items:
            if not isinstance(status_item, dict):
                continue
            for name in list(status_item.keys()):
                normalized = normalize_status_label(str(name))
                if normalized in valid_labels:
                    status_item[name] = normalized in docling_selected

        corrected_selected = {
            normalize_status_label(str(name))
            for status_item in out_status_items
            if isinstance(status_item, dict)
            for name, value in status_item.items()
            if value is True
        }
        return AcordPipeline._record_markdown_verification(
            out,
            field="status_of_transaction",
            status="corrected",
            docling_selected=sorted(docling_selected),
            output_selected=sorted(corrected_selected),
        )

    @staticmethod
    def _reconcile_nature_of_business_from_markdown(result: dict[str, Any]) -> dict[str, Any]:
        """Verify ACORD location nature-of-business checkboxes from markdown."""
        markdown = result.get("_markdown")
        locations = result.get("locations")
        if not isinstance(markdown, str) or not isinstance(locations, list) or not locations:
            return result

        current_selected: set[str] = set()
        current_value = None
        for location in locations:
            if not isinstance(location, dict):
                continue
            if current_value is None:
                current_value = location.get("nature_of_business") or location.get("business_nature")
        if isinstance(current_value, str):
            normalized_current = AcordPipeline._normalize_markdown_checkbox_label(current_value)
            for label in [
                "apartments",
                "contractor",
                "manufacturing",
                "restaurant",
                "service",
                "condominiums",
                "institutional",
                "office",
                "retail",
                "wholesale",
            ]:
                if label in normalized_current:
                    current_selected.add(label)

        section_match = re.search(
            r"##\s+NATUREOFBUSINESS\b(.*?)(?:##\s+|$)",
            markdown,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not section_match:
            return AcordPipeline._record_markdown_verification(
                result,
                field="nature_of_business",
                status="ambiguous",
                docling_selected=[],
                output_selected=sorted(current_selected),
            )

        section = section_match.group(1)
        valid_labels = {
            "apartments",
            "contractor",
            "manufacturing",
            "restaurant",
            "service",
            "condominiums",
            "institutional",
            "office",
            "retail",
            "wholesale",
        }
        docling_selected = AcordPipeline._extract_selected_checkbox_labels(section, valid_labels)
        if not docling_selected:
            docling_selected = AcordPipeline._extract_checkbox_labels_from_text(section, valid_labels)

        if not docling_selected:
            return AcordPipeline._record_markdown_verification(
                result,
                field="nature_of_business",
                status="ambiguous",
                docling_selected=[],
                output_selected=sorted(current_selected),
            )

        if docling_selected == current_selected:
            return AcordPipeline._record_markdown_verification(
                result,
                field="nature_of_business",
                status="verified",
                docling_selected=sorted(docling_selected),
                output_selected=sorted(current_selected),
            )

        out = deepcopy(result)
        out_locations = out.get("locations")
        if not isinstance(out_locations, list):
            return result

        corrected_value = ", ".join(label.title() for label in sorted(docling_selected))
        for location in out_locations:
            if not isinstance(location, dict):
                continue
            location["nature_of_business"] = corrected_value

        return AcordPipeline._record_markdown_verification(
            out,
            field="nature_of_business",
            status="corrected",
            docling_selected=sorted(docling_selected),
            output_selected=sorted(docling_selected),
        )

    @staticmethod
    def _reconcile_location_primary_operations(result: dict[str, Any]) -> dict[str, Any]:
        """Clear duplicate primary operations text that belongs in location description/nature fields."""
        locations = result.get("locations")
        if not isinstance(locations, list) or not locations:
            return result

        out = deepcopy(result)
        out_locations = out.get("locations")
        if not isinstance(out_locations, list):
            return result

        changed = False
        for location in out_locations:
            if not isinstance(location, dict):
                continue
            primary = location.get("description_of_primary_operations")
            if not isinstance(primary, str) or not primary.strip():
                continue

            description = location.get("description_of_operations")
            nature = location.get("nature_of_business") or location.get("business_nature")

            if (
                isinstance(description, str)
                and description.strip()
                and primary.strip() == description.strip()
            ) or (
                isinstance(nature, str)
                and nature.strip()
                and primary.strip() == nature.strip()
            ):
                location["description_of_primary_operations"] = None
                changed = True

        return out if changed else result

    @staticmethod
    def _reconcile_acord125_locations_from_markdown(result: dict[str, Any]) -> dict[str, Any]:
        """Route ACORD 125 location rows by page when PREMISES INFORMATION appears."""
        markdown = result.get("_markdown")
        locations = result.get("locations")
        if not isinstance(markdown, str) or not isinstance(locations, list) or not locations:
            return result

        pages = AcordPipeline._split_markdown_into_pages(markdown)
        if len(pages) >= 2:
            page1_text = pages[0]
            page2_text = pages[1]

            page1_has_section = AcordPipeline._acord125_markdown_has_premises_section(page1_text)
            page2_has_section = AcordPipeline._acord125_markdown_has_premises_section(page2_text)
            page1_has_real = AcordPipeline._acord125_premises_section_has_real_address(page1_text)
            page2_has_real = AcordPipeline._acord125_premises_section_has_real_address(page2_text)

            if page1_has_section and not page2_has_section:
                # Always filter to remove mailing-address-derived locations even if page1 has real addresses
                return AcordPipeline._filter_acord125_locations_without_real_addresses(result, remove_mailing_address_duplicates=True)

            if page2_has_section and not page1_has_section:
                return AcordPipeline._filter_acord125_locations_without_real_addresses(result, remove_mailing_address_duplicates=True)

        section_matches = re.findall(
            r"(?:##\s*PREMISES\s*INFORMATION\b|##\s*PREMISESINFORMATION\b).*?(?=##\s+[A-Z0-9]|$)",
            markdown,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not section_matches:
            return AcordPipeline._filter_acord125_locations_without_real_addresses(result, remove_mailing_address_duplicates=True)

        if any(
            AcordPipeline._acord125_premises_section_has_real_address(section)
            for section in section_matches
        ):
            # Always filter to remove mailing-address-derived locations even if premises have real addresses
            return AcordPipeline._filter_acord125_locations_without_real_addresses(result, remove_mailing_address_duplicates=True)

        return AcordPipeline._filter_acord125_locations_without_real_addresses(result, remove_mailing_address_duplicates=True)

    @staticmethod
    def _split_markdown_into_pages(markdown: str) -> list[str]:
        if not isinstance(markdown, str) or not markdown.strip():
            return []

        page_break_pattern = re.compile(
            r"\nPage\s*\d{1,2}\s*(?:of\s*\d{1,2})?\s*(?:\n|$)",
            flags=re.IGNORECASE,
        )
        pages = [segment.strip() for segment in page_break_pattern.split(markdown) if segment.strip()]
        return pages if pages else [markdown.strip()]

    @staticmethod
    def _acord125_markdown_has_premises_section(markdown: str) -> bool:
        if not isinstance(markdown, str) or not markdown.strip():
            return False
        return bool(
            re.search(
                r"(?:##\s*PREMISES\s*INFORMATION\b|##\s*PREMISESINFORMATION\b)",
                markdown,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _acord125_premises_section_has_real_address(section: str) -> bool:
        if not isinstance(section, str) or not section.strip():
            return False

        normalized_section = re.sub(r"\s+", " ", section.strip()).lower()
        street_pattern = re.compile(
            r"\b\d{1,5}\s+[a-z0-9]+(?:\s+[a-z0-9]+){0,4}\s+(?:street|st|road|rd|avenue|ave|boulevard|blvd|lane|ln|drive|dr|court|ct|way|suite|ste|unit)\b",
            flags=re.IGNORECASE,
        )
        if street_pattern.search(normalized_section):
            return True

        city_state_zip_pattern = re.compile(
            r"\b[a-z][a-z\s]+,\s*[a-z]{2}\s*\d{5}(?:-\d{4})?\b",
            flags=re.IGNORECASE,
        )
        if city_state_zip_pattern.search(normalized_section):
            return True

        lines = [
            re.sub(r"\s+", " ", line.strip()).lower()
            for line in section.splitlines()
            if line.strip()
        ]
        if not lines:
            return False

        label_line = re.compile(
            r"^(?:loc#|street|citylimits|interest|annualrevenues:\$|occupiedarea:|sq ft|bld#|city:|state:|county:|zip:|totalbuildingarea:|descriptionofoperations:|anyarealeasedtoothers\?y/n|tenant|opentopublicarea:)$",
            flags=re.IGNORECASE,
        )
        checkbox_line = re.compile(r"^(?:[-*•]\s*)?(?:\[[ xX]\]|\([ xX]\)|[xX]).*$")

        def looks_like_value(line: str) -> bool:
            if not line or not line.strip():
                return False
            if checkbox_line.match(line):
                return False
            if label_line.match(line):
                return False
            if re.search(r"\b(?:street|st|road|rd|avenue|ave|boulevard|blvd|lane|ln|drive|dr|court|ct|way|suite|ste|unit)\b", line):
                return True
            if city_state_zip_pattern.search(line):
                return True
            if re.search(r"\d{5}(?:-\d{4})?\b", line) and not re.match(r"^(?:\$|%|\d+\s*(?:sq ft|mi|mile|ft)?$)", line):
                return True
            if "," in line and len(line) > 5:
                return True
            words = [w.strip(".,") for w in line.split() if w.strip(".,")]
            if len(words) >= 2 and not all(word in label_line.pattern for word in words):
                return True
            return False

        for idx, line in enumerate(lines):
            if line.startswith("street") or line.startswith("city:") or line.startswith("state:") or line.startswith("zip:"):
                parts = line.split(":", 1)
                if len(parts) > 1 and looks_like_value(parts[1].strip()):
                    return True
                if idx + 1 < len(lines) and looks_like_value(lines[idx + 1]):
                    return True

        return False

    @staticmethod
    def _acord125_location_row_has_real_address(location: dict[str, Any]) -> bool:
        if not isinstance(location, dict):
            return False

        for key in (
            "address_line",
            "city",
            "state",
            "zip",
            "loc_number",
            "bld_number",
        ):
            value = location.get(key)
            if isinstance(value, str) and value.strip():
                return True

        return False

    @staticmethod
    def _normalize_address_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return normalized.strip()

    @staticmethod
    def _acord125_location_row_matches_mailing_address(
        location: dict[str, Any], mailing_address: str
    ) -> bool:
        if not isinstance(location, dict):
            return False
        if not isinstance(mailing_address, str) or not mailing_address.strip():
            return False

        location_city = location.get("city", "")
        location_state = location.get("state", "")
        location_zip = location.get("zip", "")
        
        # Check if city, state, and zip all match the mailing address
        # This handles cases where address_line is from company name but city/state/zip are from mailing
        if isinstance(location_city, str) and isinstance(location_state, str) and isinstance(location_zip, str):
            normalized_mailing = AcordPipeline._normalize_address_text(mailing_address)
            city_check = location_city.strip() and AcordPipeline._normalize_address_text(location_city) in normalized_mailing
            state_check = location_state.strip() and location_state.strip().lower() in normalized_mailing.lower()
            zip_check = location_zip.strip() and AcordPipeline._normalize_address_text(location_zip) in normalized_mailing
            
            if city_check and state_check and zip_check:
                return True

        address_parts: list[str] = []
        for key in ("address_line", "city", "state", "zip"):
            value = location.get(key)
            if isinstance(value, str) and value.strip():
                address_parts.append(value.strip())

        if not address_parts:
            return False

        normalized_location = AcordPipeline._normalize_address_text(" ".join(address_parts))
        normalized_mailing = AcordPipeline._normalize_address_text(mailing_address)
        return bool(normalized_location and normalized_location in normalized_mailing)

    @staticmethod
    def _get_insured_mailing_address(result: dict[str, Any]) -> str | None:
        insured = result.get("insured")
        if not isinstance(insured, dict):
            return None
        mailing = insured.get("mailing_address")
        if isinstance(mailing, str) and mailing.strip():
            return mailing.strip()
        return None

    @staticmethod
    def _filter_acord125_locations_without_real_addresses(
        result: dict[str, Any],
        remove_mailing_address_duplicates: bool = False,
    ) -> dict[str, Any]:
        locations = result.get("locations")
        if not isinstance(locations, list) or not locations:
            return result

        mailing_address = (
            AcordPipeline._get_insured_mailing_address(result)
            if remove_mailing_address_duplicates
            else None
        )

        filtered_locations: list[dict[str, Any]] = []
        for location in locations:
            if not isinstance(location, dict):
                continue
            if not AcordPipeline._acord125_location_row_has_real_address(location):
                continue
            if (
                remove_mailing_address_duplicates
                and mailing_address
                and AcordPipeline._acord125_location_row_matches_mailing_address(
                    location, mailing_address
                )
                and not location.get("loc_number")
                and not location.get("bld_number")
            ):
                continue
            filtered_locations.append(location)

        if len(filtered_locations) == len(locations):
            return result

        out = deepcopy(result)
        out["locations"] = filtered_locations
        return AcordPipeline._record_markdown_verification(
            out,
            field="locations",
            status="corrected",
            docling_selected=[],
            output_selected=[],
        )

    @staticmethod
    def _reconcile_location_interest_from_markdown(result: dict[str, Any]) -> dict[str, Any]:
        """Verify ACORD location interest checkboxes from markdown per-location.

        This uses the markdown OCR pass to verify selected Owner/Tenant/Manager
        interest labels for each location row separately and updates location interest 
        booleans when the markdown selection differs from the extracted output.
        """
        markdown = result.get("_markdown")
        locations = result.get("locations")
        if not isinstance(markdown, str) or not isinstance(locations, list) or not locations:
            return result

        valid_labels = {"owner", "tenant", "manager"}
        
        # Extract per-location by finding LOC# rows and INTEREST column content for each
        # This prevents mixing interest selections across multiple locations
        interest_by_location: dict[int, set[str]] = {}
        
        # Split markdown into location blocks using LOC# as delimiter
        loc_blocks = re.split(r"\bLOC#\b", markdown, flags=re.IGNORECASE)
        if loc_blocks and not loc_blocks[0].strip():
            loc_blocks = loc_blocks[1:]  # Remove empty first element before first LOC#
        
        for loc_index, block in enumerate(loc_blocks[:len(locations)]):
            # Extract just the INTEREST section for this location (up to next LOC# or section break)
            interest_match = re.search(
                r"\bINTEREST\b(.*?)(?:\bDESCRIPTIONOFOPERATIONS\b|$)",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            
            if not interest_match:
                interest_by_location[loc_index] = set()
                continue
            
            location_interest_section = interest_match.group(1)
            
            # Extract checked labels from this location's interest section only
            docling_selected = AcordPipeline._extract_selected_checkbox_labels(
                location_interest_section, valid_labels
            )
            if not docling_selected:
                docling_selected = AcordPipeline._extract_checkbox_labels_from_text(
                    location_interest_section, valid_labels
                )
            
            interest_by_location[loc_index] = docling_selected
        
        # Determine overall selected set from all locations (union)
        # But process each location individually for the verification record
        overall_selected: set[str] = set()
        for selected_set in interest_by_location.values():
            overall_selected.update(selected_set)
        
        # Get current state
        current_selected = set()
        for location in locations:
            if not isinstance(location, dict):
                continue
            if location.get("interest_owner"):
                current_selected.add("owner")
            if location.get("interest_tenant"):
                current_selected.add("tenant")
            if location.get("interest_manager"):
                current_selected.add("manager")
        
        has_interest_checkboxes = bool(
            re.search(
                r"^(?:[-*•]\s*)?(?:\[[ xX]\]|\([ xX]\)|[xX])\b",
                markdown,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        )
        
        if not overall_selected:
            if has_interest_checkboxes:
                # No option is explicitly selected in the interest section, so clear all flags.
                out = deepcopy(result)
                out_locations = out.get("locations")
                if isinstance(out_locations, list):
                    for location in out_locations:
                        if not isinstance(location, dict):
                            continue
                        location["interest_owner"] = False
                        location["interest_tenant"] = False
                        location["interest_manager"] = False

                return AcordPipeline._record_markdown_verification(
                    out,
                    field="location_interest",
                    status="corrected",
                    docling_selected=[],
                    output_selected=[],
                )

            return AcordPipeline._record_markdown_verification(
                result,
                field="location_interest",
                status="ambiguous",
                docling_selected=[],
                output_selected=sorted(current_selected),
            )

        if overall_selected == current_selected:
            return AcordPipeline._record_markdown_verification(
                result,
                field="location_interest",
                status="verified",
                docling_selected=sorted(overall_selected),
                output_selected=sorted(current_selected),
            )

        out = deepcopy(result)
        out_locations = out.get("locations")
        if not isinstance(out_locations, list):
            return result

        for location in out_locations:
            if not isinstance(location, dict):
                continue
            location["interest_owner"] = "owner" in overall_selected
            location["interest_tenant"] = "tenant" in overall_selected
            location["interest_manager"] = "manager" in overall_selected

        corrected_selected = {
            label
            for label in ["owner", "tenant", "manager"]
            if label in overall_selected
        }

        return AcordPipeline._record_markdown_verification(
            out,
            field="location_interest",
            status="corrected",
            docling_selected=sorted(corrected_selected),
            output_selected=sorted(corrected_selected),
        )

    @staticmethod
    def _extract_selected_checkbox_labels(section: str, valid_labels: set[str]) -> set[str]:
        """Extract explicit selected checkbox labels from a markdown section."""
        return AcordPipeline._extract_checkbox_labels_from_text(section, valid_labels)

    @staticmethod
    def _extract_checkbox_labels_from_text(section: str, valid_labels: set[str]) -> set[str]:
        selected: set[str] = set()
        lines = [line.strip() for line in section.splitlines()]
        checkbox_re = re.compile(
            r"^(?:[-*•]\s*)?(?:\u2611|\u2612|\[[xX]\]|\([xX]\)|[xX])(?:\s+(.*))?$"
        )

        # First pass: extract explicitly marked checkboxes
        seen_explicit_labels = set()
        for index, line in enumerate(lines):
            if not line:
                continue

            checked_match = checkbox_re.match(line)
            if checked_match:
                label = checked_match.group(1) or ""
                normalized = AcordPipeline._normalize_markdown_checkbox_label(label)
                if normalized in valid_labels:
                    selected.add(normalized)
                    seen_explicit_labels.add(normalized)
                    continue

                if not label.strip():
                    # Only trust a bare checkbox when a nearby non-empty line is a valid label.
                    # This prevents a standalone "X" from jumping past INSIDE/OUTSIDE
                    # and mis-selecting OWNER/TENANT/MANAGER farther down.
                    for lookahead in range(index + 1, min(index + 8, len(lines))):
                        candidate_text = lines[lookahead].strip()
                        if not candidate_text:
                            continue
                        candidate = AcordPipeline._normalize_markdown_checkbox_label(candidate_text)
                        if candidate in valid_labels:
                            selected.add(candidate)
                            seen_explicit_labels.add(candidate)
                            break

        return selected

    @staticmethod
    def _extract_checkbox_overrides_from_section(
        section: str,
        normalized_to_index: dict[str, int],
        *,
        allow_bare_x: bool,
    ) -> dict[int, bool]:
        overrides: dict[int, bool] = {}
        lines = [line.strip() for line in section.splitlines()]
        checkbox_re = re.compile(
            r"^(?:[-*•]\s*)?(?P<mark>\u2611|\u2612|\[[xX]\]|\([xX]\)|[xX])(?:\s+(?P<label>.*))?$"
        )

        for index, line in enumerate(lines):
            if not line:
                continue

            match = checkbox_re.match(line)
            if not match:
                continue

            marked = match.group("mark") not in {"[ ]"}
            label = (match.group("label") or "").strip()
            if not label and allow_bare_x:
                for lookahead in range(index + 1, min(index + 6, len(lines))):
                    candidate = lines[lookahead].strip()
                    if not candidate:
                        continue
                    normalized_candidate = AcordPipeline._normalize_lob_name(candidate)
                    if normalized_candidate in normalized_to_index:
                        label = candidate
                        break

            if not label:
                continue

            normalized_label = AcordPipeline._normalize_lob_name(label)
            idx = normalized_to_index.get(normalized_label)
            if idx is None:
                candidates = difflib.get_close_matches(
                    normalized_label,
                    normalized_to_index.keys(),
                    n=1,
                    cutoff=0.85,
                )
                if candidates:
                    idx = normalized_to_index[candidates[0]]

            if idx is not None:
                overrides[idx] = marked

        return overrides

    @staticmethod
    def _normalize_markdown_checkbox_label(value: str) -> str:
        text = html.unescape(value or "")
        text = re.sub(r"<[^>]*>", "", text)
        text = re.sub(r"^#+\s*", "", text)
        text = re.sub(r"^(?:[-*•]\s*)?(?:\[[xX]\]|\([xX\]\)|[xX])\s*", "", text)
        text = re.sub(r"^\s*[xX]\s*", "", text)
        text = re.sub(r"\s+(?:[xX])\s*$", "", text)
        text = re.sub(r"\s*\([^)]*\)\s*[:;,.!?]*\s*$", "", text)
        text = re.sub(r"[:;,.!?]+\s*$", "", text)
        
        # Add spaces between lowercase-uppercase transitions (fixes OCR like PersonalVehicles)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        
        # Split known all-caps words without spaces (OCR artifacts)
        # Common attachment/field names that get concatenated
        all_caps_splits = [
            # Attachment types - match complete words first
            (r"GLASSANDSIGNSECTION", "GLASS AND SIGN SECTION"),
            (r"INSTALLATIONBUILDERSRISKSECTION", "INSTALLATION BUILDERS RISK SECTION"),
            (r"INTERNATIONALLIABILITYEXPOSURE", "INTERNATIONAL LIABILITY EXPOSURE"),
            (r"INTERNATIONALPROPERTYEXPOSURE", "INTERNATIONAL PROPERTY EXPOSURE"),
            (r"STATESUPPLEMENT", "STATE SUPPLEMENT"),
            (r"ELECTRONICDATAPROCESSINGSECTION", "ELECTRONIC DATA PROCESSING SECTION"),
            (r"CONDOASSN", "CONDO ASSN"),
            (r"DRIVERINFORMATION", "DRIVER INFORMATION"),
            (r"DEALERSSECTION", "DEALERS SECTION"),
            (r"LOSSSUMMARY", "LOSS SUMMARY"),
            (r"RESTAURANTTAVERNSUPPLEMENT", "RESTAURANT TAVERN SUPPLEMENT"),
            (r"RESTAURANTTAVERN", "RESTAURANT TAVERN"),
            (r"VACANTBUILDING", "VACANT BUILDING"),
            (r"VEHICLESCHEDULE", "VEHICLE SCHEDULE"),
            (r"CONTRACTORSSUPPLEMENT", "CONTRACTORS SUPPLEMENT"),
            (r"PREMIUMPA", "PREMIUM PAYMENT"),
            (r"PROFESSIONALIABILITY", "PROFESSIONAL LIABILITY"),
            (r"CRIMEMISCELLANEOUS", "CRIME MISCELLANEOUS"),
            # Other fields
            (r"BUILDERISK", "BUILDERS RISK"),
            (r"BUSINESSOWNERS", "BUSINESS OWNERS"),
            (r"INLANDMARINE", "INLAND MARINE"),
            (r"FIDUCIARYLIABILITY", "FIDUCIARY LIABILITY"),
            (r"LIQUORLIABILITY", "LIQUOR LIABILITY"),
            (r"MOTORCARRIER", "MOTOR CARRIER"),
            (r"GLASSSIGN", "GLASS SIGN"),
        ]
        
        for pattern, replacement in all_caps_splits:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    @staticmethod
    def _normalize_status_checkbox_label(value: str) -> str:
        normalized = AcordPipeline._normalize_markdown_checkbox_label(value)
        return "bound" if normalized in {"bind", "bound"} else normalized

    @staticmethod
    def _record_markdown_verification(
        result: dict[str, Any],
        *,
        field: str,
        status: str,
        docling_selected: list[str],
        output_selected: list[str],
    ) -> dict[str, Any]:
        out = deepcopy(result)
        verification = out.get("_markdown_verification")
        if not isinstance(verification, dict):
            verification = {}
            out["_markdown_verification"] = verification

        verification[field] = {
            "status": status,
            "source": "docling_markdown",
            "docling_selected": docling_selected,
            "output_selected": output_selected,
        }
        AcordPipeline._refresh_markdown_verification_summary(verification)
        return out

    @staticmethod
    def _record_nuextract_verification(
        result: dict[str, Any],
        *,
        field: str,
        status: str,
        docling_selected: list[str],
        nuextract_selected: list[str],
        output_selected: list[str],
    ) -> dict[str, Any]:
        out = deepcopy(result)
        verification = out.get("_markdown_verification")
        if not isinstance(verification, dict):
            verification = {}
            out["_markdown_verification"] = verification

        verification[field] = {
            "status": status,
            "source": "nuextract_reconciliation",
            "docling_selected": docling_selected,
            "nuextract_selected": nuextract_selected,
            "output_selected": output_selected,
        }
        AcordPipeline._refresh_markdown_verification_summary(verification)
        return out

    @staticmethod
    def _extract_markdown_checkbox_selections(
        markdown: str,
        valid_labels: set[str],
    ) -> set[str]:
        if not isinstance(markdown, str) or not markdown.strip() or not valid_labels:
            return set()

        selected = AcordPipeline._extract_selected_checkbox_labels(markdown, valid_labels)
        if selected:
            return selected

        return AcordPipeline._extract_checkbox_labels_from_text(markdown, valid_labels)

    def _run_nuextract_selection_validation(
        self,
        page_image: Image.Image,
        template_key: str,
        valid_labels: set[str],
        prompt: str,
        enable_thinking: bool,
        text: str | None = None,
    ) -> set[str]:
        try:
            row_result = self.extractor.extract(
                images=[page_image],
                text=text,
                template={template_key: ["string"]},
                instructions=prompt,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            logger.warning(f"NuExtract validation failed for {template_key}: {exc}")
            return set()

        if not isinstance(row_result, dict):
            return set()

        selected = row_result.get(template_key)
        if isinstance(selected, str):
            selected = [selected]
        if not isinstance(selected, list):
            return set()

        normalized_selected: set[str] = set()
        for item in selected:
            if not isinstance(item, str):
                continue
            normalized = AcordPipeline._normalize_markdown_checkbox_label(item)
            if normalized in valid_labels:
                normalized_selected.add(normalized)

        return normalized_selected

    def _resolve_ambiguous_markdown_fields_with_nuextract(
        self,
        result: dict[str, Any],
        images: list[Any],
        enable_thinking: bool,
    ) -> dict[str, Any]:
        verification = result.get("_markdown_verification")
        if not isinstance(verification, dict) or not isinstance(images, list) or not images:
            return result

        ambiguous_fields = [
            field
            for field, details in verification.items()
            if isinstance(details, dict) and details.get("status") == "ambiguous"
        ]

        if not ambiguous_fields:
            return result

        page_image = images[0]
        for field in ambiguous_fields:
            if field == "status_of_transaction":
                result = self._validate_status_of_transaction_with_nuextract(
                    result,
                    page_image,
                    enable_thinking,
                )
            elif field == "location_interest":
                result = self._validate_location_interest_with_nuextract(
                    result,
                    page_image,
                    enable_thinking,
                )
            elif field == "attachments":
                result = self._validate_attachments_with_nuextract(
                    result,
                    page_image,
                    enable_thinking,
                )
            elif field == "company_type_options":
                result = self._validate_company_type_with_nuextract(
                    result,
                    page_image,
                    enable_thinking,
                )
            elif field == "nature_of_business":
                result = self._validate_nature_of_business_with_nuextract(
                    result,
                    page_image,
                    enable_thinking,
                )

        return result

    def _validate_status_of_transaction_with_nuextract(
        self,
        result: dict[str, Any],
        page_image: Image.Image,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        verification = result.get("_markdown_verification", {})
        status_verification = verification.get("status_of_transaction")
        if not isinstance(status_verification, dict) or status_verification.get("status") != "ambiguous":
            return result

        policy_header = result.get("policy_header")
        if not isinstance(policy_header, dict):
            return result

        status_items = policy_header.get("status_of_transaction")
        if not isinstance(status_items, list) or not status_items:
            return result

        valid_labels = {
            "quote",
            "bound",
            "issue policy",
            "renew",
            "change",
            "cancel",
        }

        def normalize_status_label(value: str) -> str:
            normalized = AcordPipeline._normalize_markdown_checkbox_label(value)
            return "bound" if normalized == "bound" else normalized

        current_selected: set[str] = set()
        for item in status_items:
            if not isinstance(item, dict):
                continue
            for name, value in item.items():
                if value is True:
                    normalized = normalize_status_label(str(name))
                    if normalized in valid_labels:
                        current_selected.add(normalized)

        prompt = (
            "Read only the STATUS OF TRANSACTION checkbox area on ACORD 125. "
            "List only the transaction statuses whose own checkbox is clearly marked. "
            "Do not use nearby time fields, AM/PM indicators, or any other page text. "
            "Return JSON only with key selected_statuses."
        )

        status_section = ""
        markdown = result.get("_markdown")
        if isinstance(markdown, str):
            section_match = re.search(
                r"(?:STATUSOF\s*TRANSACTION|TRANSACTION\s*STATUSOF|STATUS\s*OF\s*TRANSACTION)(.*?)(?:##\s+|$)",
                markdown,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if section_match:
                status_section = section_match.group(1)

        if status_section:
            selected = self._run_nuextract_selection_validation(
                page_image,
                "selected_statuses",
                valid_labels,
                prompt,
                enable_thinking,
                text=status_section,
            )
        else:
            selected = self._run_nuextract_selection_validation(
                page_image,
                "selected_statuses",
                valid_labels,
                prompt,
                enable_thinking,
            )

        if not selected:
            return result

        selected = {normalize_status_label(value) for value in selected if isinstance(value, str)}
        selected = {value for value in selected if value in valid_labels}

        out = deepcopy(result)
        out_policy_header = out.get("policy_header")
        if not isinstance(out_policy_header, dict):
            return result

        out_status_items = out_policy_header.get("status_of_transaction")
        if not isinstance(out_status_items, list):
            return result

        for status_item in out_status_items:
            if not isinstance(status_item, dict):
                continue
            for name in list(status_item.keys()):
                normalized = normalize_status_label(str(name))
                if normalized in valid_labels:
                    status_item[name] = normalized in selected

        return AcordPipeline._record_nuextract_verification(
            out,
            field="status_of_transaction",
            status="verified" if selected == current_selected else "corrected",
            docling_selected=status_verification.get("docling_selected", []),
            nuextract_selected=sorted(selected),
            output_selected=sorted(selected),
        )

    def _validate_location_interest_with_nuextract(
        self,
        result: dict[str, Any],
        page_image: Image.Image,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        verification = result.get("_markdown_verification", {})
        interest_verification = verification.get("location_interest")
        if not isinstance(interest_verification, dict) or interest_verification.get("status") != "ambiguous":
            return result

        locations = result.get("locations")
        if not isinstance(locations, list) or not locations:
            return result

        current_selected: set[str] = set()
        for location in locations:
            if not isinstance(location, dict):
                continue
            if location.get("interest_owner"):
                current_selected.add("owner")
            if location.get("interest_tenant"):
                current_selected.add("tenant")
            if location.get("interest_manager"):
                current_selected.add("manager")

        valid_labels = {"owner", "tenant", "manager"}
        prompt = (
            "Read only the PREMISES INFORMATION interest checkbox area on ACORD 125. "
            "Decide whether OWNER, TENANT, or MANAGER is selected for the location interest. "
            "Return JSON only with key selected_interest."
        )
        selected = self._run_nuextract_selection_validation(
            page_image,
            "selected_interest",
            valid_labels,
            prompt,
            enable_thinking,
        )
        if not selected:
            return result

        out = deepcopy(result)
        out_locations = out.get("locations")
        if not isinstance(out_locations, list):
            return result

        for location in out_locations:
            if not isinstance(location, dict):
                continue
            location["interest_owner"] = "owner" in selected
            location["interest_tenant"] = "tenant" in selected
            location["interest_manager"] = "manager" in selected

        return AcordPipeline._record_nuextract_verification(
            out,
            field="location_interest",
            status="verified" if selected == current_selected else "corrected",
            docling_selected=interest_verification.get("docling_selected", []),
            nuextract_selected=sorted(selected),
            output_selected=sorted(selected),
        )

    def _validate_attachments_with_nuextract(
        self,
        result: dict[str, Any],
        page_image: Image.Image,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        verification = result.get("_markdown_verification", {})
        attachment_verification = verification.get("attachments")
        if not isinstance(attachment_verification, dict) or attachment_verification.get("status") != "ambiguous":
            return result

        attachments = result.get("attachments")
        if not isinstance(attachments, list) or not attachments:
            return result

        valid_labels = {
            AcordPipeline._normalize_markdown_checkbox_label(str(item.get("attachment_type")))
            for item in attachments
            if isinstance(item, dict) and isinstance(item.get("attachment_type"), str)
        }
        if not valid_labels:
            return result

        current_selected = {
            AcordPipeline._normalize_markdown_checkbox_label(str(item.get("attachment_type")))
            for item in attachments
            if isinstance(item, dict) and item.get("selected") is True and isinstance(item.get("attachment_type"), str)
        }

        prompt = (
            "Read only the ATTACHMENTS checkbox area on ACORD 125. "
            "List only the attachment types whose own checkbox is clearly marked. "
            "Return JSON only with key selected_attachments."
        )
        selected = self._run_nuextract_selection_validation(
            page_image,
            "selected_attachments",
            valid_labels,
            prompt,
            enable_thinking,
        )
        if not selected:
            return result

        out = deepcopy(result)
        out_attachments = out.get("attachments")
        if not isinstance(out_attachments, list):
            return result

        for item in out_attachments:
            if not isinstance(item, dict):
                continue
            attachment_type = item.get("attachment_type")
            if isinstance(attachment_type, str):
                normalized = AcordPipeline._normalize_markdown_checkbox_label(attachment_type)
                item["selected"] = normalized in selected

        return AcordPipeline._record_nuextract_verification(
            out,
            field="attachments",
            status="verified" if selected == current_selected else "corrected",
            docling_selected=attachment_verification.get("docling_selected", []),
            nuextract_selected=sorted(selected),
            output_selected=sorted({
                AcordPipeline._normalize_markdown_checkbox_label(str(item.get("attachment_type")))
                for item in out_attachments
                if isinstance(item, dict) and item.get("selected") is True and isinstance(item.get("attachment_type"), str)
            }),
        )

    def _validate_company_type_with_nuextract(
        self,
        result: dict[str, Any],
        page_image: Image.Image,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        verification = result.get("_markdown_verification", {})
        company_verification = verification.get("company_type_options")
        if not isinstance(company_verification, dict) or company_verification.get("status") != "ambiguous":
            return result

        insured = result.get("insured")
        if not isinstance(insured, dict):
            return result

        company_type_options = insured.get("company_type_options")
        if not isinstance(company_type_options, list) or not company_type_options:
            return result

        valid_labels = {
            AcordPipeline._normalize_markdown_checkbox_label(str(item.get("company_type")))
            for item in company_type_options
            if isinstance(item, dict) and isinstance(item.get("company_type"), str)
        }
        if not valid_labels:
            return result

        current_selected = {
            AcordPipeline._normalize_markdown_checkbox_label(str(item.get("company_type")))
            for item in company_type_options
            if isinstance(item, dict) and item.get("selected") is True and isinstance(item.get("company_type"), str)
        }

        prompt = (
            "Read only the company type checkbox area on ACORD 125. "
            "List only the company types whose own checkbox is clearly marked. "
            "Return JSON only with key selected_company_types."
        )
        selected = self._run_nuextract_selection_validation(
            page_image,
            "selected_company_types",
            valid_labels,
            prompt,
            enable_thinking,
        )
        if not selected:
            return result

        out = deepcopy(result)
        out_insured = out.get("insured")
        if not isinstance(out_insured, dict):
            return result

        out_company_type_options = out_insured.get("company_type_options")
        if not isinstance(out_company_type_options, list):
            return result

        for item in out_company_type_options:
            if not isinstance(item, dict):
                continue
            company_type = item.get("company_type")
            if isinstance(company_type, str):
                normalized = AcordPipeline._normalize_markdown_checkbox_label(company_type)
                item["selected"] = normalized in selected

        return AcordPipeline._record_nuextract_verification(
            out,
            field="company_type_options",
            status="verified" if selected == current_selected else "corrected",
            docling_selected=company_verification.get("docling_selected", []),
            nuextract_selected=sorted(selected),
            output_selected=sorted({
                AcordPipeline._normalize_markdown_checkbox_label(str(item.get("company_type")))
                for item in out_company_type_options
                if isinstance(item, dict) and item.get("selected") is True and isinstance(item.get("company_type"), str)
            }),
        )

    def _validate_nature_of_business_with_nuextract(
        self,
        result: dict[str, Any],
        page_image: Image.Image,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        verification = result.get("_markdown_verification", {})
        nature_verification = verification.get("nature_of_business")
        if not isinstance(nature_verification, dict) or nature_verification.get("status") != "ambiguous":
            return result

        locations = result.get("locations")
        if not isinstance(locations, list) or not locations:
            return result

        valid_labels = {
            "apartments",
            "contractor",
            "manufacturing",
            "restaurant",
            "service",
            "condominiums",
            "institutional",
            "office",
            "retail",
            "wholesale",
        }

        current_selected: set[str] = set()
        current_value = None
        for location in locations:
            if not isinstance(location, dict):
                continue
            if current_value is None:
                current_value = location.get("nature_of_business") or location.get("business_nature")
        if isinstance(current_value, str):
            normalized_current = AcordPipeline._normalize_markdown_checkbox_label(current_value)
            for label in valid_labels:
                if label in normalized_current:
                    current_selected.add(label)

        prompt = (
            "Read only the NATURE OF BUSINESS checkbox area on ACORD 125. "
            "List only the business nature labels whose own checkbox is clearly marked. "
            "Return JSON only with key selected_nature_of_business."
        )
        selected = self._run_nuextract_selection_validation(
            page_image,
            "selected_nature_of_business",
            valid_labels,
            prompt,
            enable_thinking,
        )
        if not selected:
            return result

        corrected_value = ", ".join(label.title() for label in sorted(selected))
        out = deepcopy(result)
        out_locations = out.get("locations")
        if not isinstance(out_locations, list):
            return result

        for location in out_locations:
            if not isinstance(location, dict):
                continue
            location["nature_of_business"] = corrected_value

        return AcordPipeline._record_nuextract_verification(
            out,
            field="nature_of_business",
            status="verified" if selected == current_selected else "corrected",
            docling_selected=nature_verification.get("docling_selected", []),
            nuextract_selected=sorted(selected),
            output_selected=sorted(selected),
        )

    @staticmethod
    def _refresh_markdown_verification_summary(verification: dict[str, Any]) -> None:
        verified_fields = [
            name for name, details in verification.items()
            if isinstance(details, dict) and details.get("status") == "verified"
        ]
        corrected_fields = [
            name for name, details in verification.items()
            if isinstance(details, dict) and details.get("status") == "corrected"
        ]
        ambiguous_fields = [
            name for name, details in verification.items()
            if isinstance(details, dict) and details.get("status") == "ambiguous"
        ]

        verification["summary"] = {
            "source": "docling_markdown",
            "fields_reviewed": len(verified_fields) + len(corrected_fields) + len(ambiguous_fields),
            "verified_count": len(verified_fields),
            "corrected_count": len(corrected_fields),
            "ambiguous_count": len(ambiguous_fields),
            "verified_fields": sorted(verified_fields),
            "corrected_fields": sorted(corrected_fields),
            "ambiguous_fields": sorted(ambiguous_fields),
        }

    def _validate_lob_with_nuextract(
        self,
        result: dict[str, Any],
        images: list[Any],
        enable_thinking: bool,
    ) -> dict[str, Any]:
        """Use NuExtract as a reconciliation layer for LOB selections by showing it both NuExtract and Docling results."""
        lobs = result.get("lines_of_business")
        verification = result.get("_markdown_verification", {})
        lob_verification = verification.get("lines_of_business", {})

        if not isinstance(lobs, list) or not lobs:
            return result

        if lob_verification.get("status") != "corrected":
            return result

        original_selected = lob_verification.get("output_selected", [])
        docling_selected = lob_verification.get("docling_selected", [])

        if not original_selected or not docling_selected:
            return result

        if not images or len(images) == 0:
            return result

        all_lobs = [item.get("lob") for item in lobs if isinstance(item, dict) and isinstance(item.get("lob"), str)]
        if not all_lobs:
            return result

        canonical_map = {
            self._normalize_lob_name(item.get("lob")): item.get("lob")
            for item in lobs
            if isinstance(item, dict) and isinstance(item.get("lob"), str)
        }

        original_normalized = set(original_selected)
        docling_normalized = set(docling_selected)
        agreed_normalized = original_normalized.intersection(docling_normalized)
        disputed_normalized = original_normalized.symmetric_difference(docling_normalized)

        if not disputed_normalized:
            return result

        disputed_canonical = [canonical_map[n] for n in sorted(disputed_normalized) if n in canonical_map]
        agreed_canonical = [canonical_map[n] for n in sorted(agreed_normalized) if n in canonical_map]

        validation_image = self._crop_lob_region(images[0])
        disputed_selected_normalized: set[str] = set()

        logger.info("  Running NuExtract LOB validation pass (row-by-row disputed reconciliation)...")
        for disputed_lob in disputed_canonical:
            row_prompt = (
                "Read only the LINES OF BUSINESS checkbox area for ACORD 125 page 1. "
                "For this row, decide whether the checkbox immediately to the left of the row label is clearly marked. "
                "Do not use any checkbox or mark from another row or nearby text. "
                "Only return true when the checkbox on the left side of this exact row is visibly checked. "
                "If the left-side checkbox is not clearly marked, return false.\n\n"
                f"Row label to evaluate: {disputed_lob}\n"
                f"Rows agreed checked by both sources (context only): {', '.join(agreed_canonical) or 'None'}\n\n"
                "Return JSON only with keys: lob, checked, evidence."
            )
            row_template = {
                "lob": disputed_lob,
                "checked": False,
                "evidence": "short text"
            }

            try:
                row_result = self.extractor.extract(
                    images=[validation_image],
                    template=row_template,
                    instructions=row_prompt,
                    enable_thinking=enable_thinking,
                )
            except Exception as exc:
                logger.warning(f"LOB row validation failed for '{disputed_lob}': {exc}")
                continue

            row_checked = False
            if isinstance(row_result, dict):
                checked_value = row_result.get("checked")
                if isinstance(checked_value, bool):
                    row_checked = checked_value
                elif isinstance(checked_value, str):
                    row_checked = checked_value.strip().lower() in {"true", "yes", "checked", "1"}

            logger.info(f"  LOB row validation '{disputed_lob}' => checked={row_checked}")

            if row_checked:
                normalized = self._normalize_lob_name(disputed_lob)
                if normalized in disputed_normalized:
                    disputed_selected_normalized.add(normalized)

        nuextract_normalized = agreed_normalized.union(disputed_selected_normalized)

        if nuextract_normalized == original_normalized:
            consensus = "nuextract_original"
        elif nuextract_normalized == docling_normalized:
            consensus = "docling_agreed"
        else:
            consensus = "nuextract_new_reconciliation"

        out = deepcopy(result)
        for item in out.get("lines_of_business", []):
            if not isinstance(item, dict):
                continue
            normalized = self._normalize_lob_name(item.get("lob", ""))
            item["selected"] = normalized in nuextract_normalized

        out_verification = out.get("_markdown_verification", {})
        if not isinstance(out_verification, dict):
            out_verification = {}
            out["_markdown_verification"] = out_verification

        out_verification["lines_of_business_validation"] = {
            "status": "validated",
            "source": "nuextract_reconciliation",
            "consensus": consensus,
            "original_nuextract_selected": sorted(original_selected),
            "docling_selected": sorted(docling_selected),
            "agreed_selected": sorted([canonical_map.get(lob, lob) for lob in agreed_normalized]),
            "disputed_candidates": sorted([canonical_map.get(lob, lob) for lob in disputed_normalized]),
            "disputed_selected_by_validation": sorted([canonical_map.get(lob, lob) for lob in disputed_selected_normalized]),
            "final_selected": sorted([canonical_map.get(lob, lob) for lob in nuextract_normalized]),
        }
        AcordPipeline._refresh_markdown_verification_summary(out_verification)
        return out

    @staticmethod
    def _reconcile_billing_plan_from_markdown(result: dict[str, Any]) -> dict[str, Any]:
        """Set billing_plan from explicit markdown checkbox marker near DIRECT/AGENCY BILL labels."""
        markdown = result.get("_markdown")
        policy_info = result.get("policy_information")
        if not isinstance(markdown, str) or not isinstance(policy_info, dict):
            return result

        marker = r"(?:\u2612|\u2611|\[[xX]\]|\([xX]\)|\bX\b)"
        direct_label_present = re.search(r"\bDIRECT(?:\s+BILL)?\b", markdown, flags=re.IGNORECASE) is not None
        agency_label_present = re.search(r"\bAGENCY(?:\s+BILL)?\b", markdown, flags=re.IGNORECASE) is not None

        direct_marked = re.search(
            rf"{marker}\s*DIRECT(?:\s+BILL)?\b",
            markdown,
            flags=re.IGNORECASE,
        ) is not None
        agency_marked = re.search(
            rf"{marker}\s*AGENCY(?:\s+BILL)?\b",
            markdown,
            flags=re.IGNORECASE,
        ) is not None

        # If both labels exist but there is no explicit mark, treat billing plan as unknown.
        if direct_label_present and agency_label_present and not direct_marked and not agency_marked:
            out = deepcopy(result)
            out_policy = out.get("policy_information")
            if isinstance(out_policy, dict):
                out_policy["billing_plan"] = None
            return out

        # Only set a concrete value when exactly one option is explicitly marked.
        if direct_marked == agency_marked:
            return result

        out = deepcopy(result)
        out_policy = out.get("policy_information")
        if not isinstance(out_policy, dict):
            return result

        out_policy["billing_plan"] = "Direct" if direct_marked else "Agency"
        return out

    def run_ocr(
        self,
        file_path: str | Path,
        enable_thinking: bool = False,
        save_output: bool = True,
    ) -> str:
        """
        Convert an ACORD form to Markdown (OCR mode) without a template.

        Useful as a pre-processing step or for inspecting raw content.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        images = self._load_images(file_path)
        ocr_payload = self._ocr_with_metadata(
            file_path=file_path,
            images=images,
            enable_thinking=enable_thinking,
        )
        markdown = ocr_payload["markdown"]

        if save_output:
            out_path = config.OUTPUTS_DIR / f"{file_path.stem}_ocr.md"
            config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path.write_text(markdown, encoding="utf-8")
            logger.info(f"OCR result saved → {out_path}")

            key_values = ocr_payload.get("key_value_pairs")
            if isinstance(key_values, list) and key_values:
                kv_out_path = config.OUTPUTS_DIR / f"{file_path.stem}_ocr_key_values.json"
                kv_out_path.write_text(
                    json.dumps(key_values, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info(f"OCR key-value pairs saved → {kv_out_path}")

        return markdown

    def _ocr_with_metadata(
        self,
        file_path: Path,
        images: list[Image.Image],
        enable_thinking: bool,
    ) -> dict[str, Any]:
        if self.docling_ocr is not None:
            logger.info(f"  Using {self.docling_ocr.provider_label} for OCR Markdown conversion.")
            return self.docling_ocr.extract_markdown_with_metadata(file_path)

        logger.info("  Docling OCR not configured; falling back to extractor OCR.")
        return {
            "markdown": self.extractor.ocr(images=images, enable_thinking=enable_thinking),
            "key_value_pairs": [],
        }

    @staticmethod
    def _reconcile_from_ocr_key_values(result: dict[str, Any]) -> dict[str, Any]:
        """Use OCR key-value checkbox hints to update ACORD 125 selections."""
        key_values = result.get("_ocr_key_values") or result.get("_doc_intel_key_values")
        if not isinstance(key_values, list) or not key_values:
            return result

        out = deepcopy(result)

        lines = out.get("lines_of_business")
        if isinstance(lines, list) and lines:
            for row in lines:
                if not isinstance(row, dict):
                    continue
                lob_value = row.get("lob")
                if not isinstance(lob_value, str) or not lob_value.strip():
                    continue

                normalized_lob = AcordPipeline._normalize_lob_name(lob_value)
                for pair in key_values:
                    if not isinstance(pair, dict):
                        continue
                    selected = pair.get("selected")
                    if not isinstance(selected, bool):
                        continue
                    key_text = str(pair.get("key") or "")
                    value_text = str(pair.get("value") or "")
                    if normalized_lob in AcordPipeline._normalize_lob_name(key_text + " " + value_text):
                        row["selected"] = selected
                        break

        policy_info = out.get("policy_information")
        if isinstance(policy_info, dict):
            direct_selected: bool | None = None
            agency_selected: bool | None = None

            for pair in key_values:
                if not isinstance(pair, dict):
                    continue
                selected = pair.get("selected")
                if not isinstance(selected, bool):
                    continue

                text = f"{pair.get('key') or ''} {pair.get('value') or ''}".upper()
                if "DIRECT" in text and direct_selected is None:
                    direct_selected = selected
                if "AGENCY" in text and agency_selected is None:
                    agency_selected = selected

            if isinstance(direct_selected, bool) and isinstance(agency_selected, bool):
                if direct_selected != agency_selected:
                    policy_info["billing_plan"] = "Direct" if direct_selected else "Agency"
            elif isinstance(direct_selected, bool) and direct_selected:
                policy_info["billing_plan"] = "Direct"
            elif isinstance(agency_selected, bool) and agency_selected:
                policy_info["billing_plan"] = "Agency"

        policy_header = out.get("policy_header")
        if isinstance(policy_header, dict):
            agency_value: str | None = None
            date_value: str | None = None
            phone_value: str | None = None

            for pair in key_values:
                if not isinstance(pair, dict):
                    continue
                key_text = str(pair.get("key") or "")
                value_text = str(pair.get("value") or "").strip()
                if not value_text:
                    continue

                upper_key = key_text.upper()
                if agency_value is None and "AGENCY" in upper_key:
                    agency_value = value_text
                if date_value is None and "DATE OF APPLICATION" in upper_key:
                    date_value = value_text
                if phone_value is None and "PHONE" in upper_key:
                    phone_value = value_text

            if not policy_header.get("agency") and agency_value:
                policy_header["agency"] = agency_value
            if not policy_header.get("date_of_application") and date_value:
                policy_header["date_of_application"] = date_value
            if not policy_header.get("agency_phone_number") and phone_value:
                policy_header["agency_phone_number"] = phone_value

        return out

    @staticmethod
    def _reconcile_acord140_from_markdown_and_key_values(result: dict[str, Any]) -> dict[str, Any]:
        """Post-process ACORD 140 fields using OCR markdown + OCR key-values.

        Fixes two common issues:
        1) `acord140PremisesInformation` should be row-wise per SUBJECT OF INSURANCE table.
        2) Missing roofing/heating years should be recovered from BUILDING IMPROVEMENTS text.
        """
        markdown = result.get("_markdown")
        key_values = result.get("_ocr_key_values") or result.get("_doc_intel_key_values")
        policy = result.get("acord140Policy")
        premises = result.get("acord140PremisesInformation")

        if not isinstance(markdown, str):
            return result
        if not isinstance(policy, dict) or not isinstance(premises, dict):
            return result

        out = deepcopy(result)
        out_policy = out.get("acord140Policy")
        out_premises = out.get("acord140PremisesInformation")
        if not isinstance(out_policy, dict) or not isinstance(out_premises, dict):
            return result

        # Build row-wise arrays from the first SUBJECT OF INSURANCE table.
        table_match = re.search(
            r"<table>\s*<tr>\s*<th>SUBJECT OF INSURANCE</th>.*?</table>",
            markdown,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if table_match:
            table_html = table_match.group(0)
            row_html = re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL)
            data_rows: list[list[str | None]] = []

            for row in row_html[1:]:  # skip header row
                cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.IGNORECASE | re.DOTALL)
                if not cells:
                    continue
                norm_cells: list[str | None] = []
                for raw_cell in cells:
                    text = re.sub(r"<.*?>", "", raw_cell)
                    text = re.sub(r"\s+", " ", text).strip()
                    norm_cells.append(text if text else None)

                # pad to expected 10 columns
                while len(norm_cells) < 10:
                    norm_cells.append(None)
                norm_cells = norm_cells[:10]

                # keep only rows with at least a subject label
                if norm_cells[0]:
                    data_rows.append(norm_cells)

            if data_rows:
                out_premises["subject_of_insurance"] = [row[0] for row in data_rows]
                out_premises["amount"] = [row[1] for row in data_rows]
                out_premises["coins_percent"] = [row[2] for row in data_rows]
                out_premises["valuation"] = [row[3] for row in data_rows]
                out_premises["causes_of_losses"] = [row[4] for row in data_rows]
                out_premises["inflation_guard_percent"] = [row[5] for row in data_rows]
                out_premises["ded"] = [row[6] for row in data_rows]
                out_premises["ded_type"] = [row[7] for row in data_rows]
                out_premises["blkt_number"] = [row[8] for row in data_rows]
                out_premises["forms_and_conditions_to_apply"] = [row[9] for row in data_rows]

                # Also provide explicit row-wise entries for easier downstream use.
                out_premises["rows"] = [
                    {
                        "subject_of_insurance": row[0],
                        "amount": row[1],
                        "coins_percent": row[2],
                        "valuation": row[3],
                        "causes_of_losses": row[4],
                        "inflation_guard_percent": row[5],
                        "ded": row[6],
                        "ded_type": row[7],
                        "blkt_number": row[8],
                        "forms_and_conditions_to_apply": row[9],
                    }
                    for row in data_rows
                ]

        # Recover roofing/heating years from markdown text first.
        roofing_year = AcordPipeline._extract_year_from_markdown(markdown, "ROOFING")
        heating_year = AcordPipeline._extract_year_from_markdown(markdown, "HEATING")

        # If still missing, attempt DI key-value fallback.
        if (roofing_year is None or heating_year is None) and isinstance(key_values, list):
            for pair in key_values:
                if not isinstance(pair, dict):
                    continue
                key_text = str(pair.get("key") or "")
                value_text = str(pair.get("value") or "")
                combined = f"{key_text} {value_text}"

                upper_combined = combined.upper()
                if roofing_year is None and "ROOFING" in upper_combined:
                    roofing_year = AcordPipeline._extract_year_candidate(combined)
                if heating_year is None and "HEATING" in upper_combined:
                    heating_year = AcordPipeline._extract_year_candidate(combined)

                if roofing_year is not None and heating_year is not None:
                    break

        if AcordPipeline._is_empty_value(out_policy.get("roofingyearPremisesInformation")) and roofing_year is not None:
            out_policy["roofingyearPremisesInformation"] = roofing_year
        if AcordPipeline._is_empty_value(out_policy.get("heatingYearPremisesInformation")) and heating_year is not None:
            out_policy["heatingYearPremisesInformation"] = heating_year

        return out

    @staticmethod
    def _extract_year_from_markdown(markdown: str, label: str) -> int | None:
        pattern = rf"{label}\s*,\s*YR\s*:\s*([0-9][0-9\. ]{{1,8}})"
        match = re.search(pattern, markdown, flags=re.IGNORECASE)
        if not match:
            return None
        return AcordPipeline._extract_year_candidate(match.group(1))

    @staticmethod
    def _extract_year_candidate(text: str) -> int | None:
        digits = re.sub(r"\D", "", text or "")
        if len(digits) < 4:
            return None

        year_str = digits[:4]
        try:
            year = int(year_str)
        except ValueError:
            return None

        if 1800 <= year <= 2100:
            return year
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_text(file_path: Path) -> str:
        """Load text content from a PDF, text file, or OCR JSON export."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            page_texts = AcordPipeline._load_pdf_text_pages(file_path)
            return "\n\n".join(page_texts)

        if suffix in {".txt", ".md"}:
            return file_path.read_text(encoding="utf-8")

        if suffix == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "text" in data and isinstance(data["text"], str):
                return data["text"]
            raise ValueError("JSON input must contain a top-level string field named 'text'.")

        raise ValueError(
            f"Unsupported file type '{suffix}' for API text extraction. "
            "Supported: .pdf, .txt, .md, .json"
        )

    @staticmethod
    def _load_pdf_text_pages(file_path: Path) -> list[str]:
        """Extract text from each PDF page for page-wise API calls."""
        if file_path.suffix.lower() != ".pdf":
            raise ValueError("Page-wise PDF extraction is only supported for .pdf files.")

        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF text extraction. "
                "Install it with: pip install pymupdf"
            ) from exc

        pages: list[str] = []
        empty_page_count = 0
        with fitz.open(str(file_path)) as doc:
            for page in doc:
                text = page.get_text("text").strip()
                if not text:
                    # Fallback for scanned/image-only pages.
                    text = AcordPipeline._extract_page_text_with_ocr(page).strip()
                if text:
                    pages.append(text)
                else:
                    empty_page_count += 1

        if not pages:
            raise ValueError(
                "No extractable text found in PDF. "
                "Tried both native extraction and OCR fallback. "
                "Ensure Tesseract OCR is installed or provide a .json file with a top-level 'text' field."
            )

        if empty_page_count:
            logger.warning(
                "Could not extract text from "
                f"{empty_page_count} page(s); those pages were skipped in API mode."
            )

        return pages

    @staticmethod
    def _prepare_page_text_for_api(page_text: str) -> str:
        """Normalize and cap page text to keep API completion prompts bounded."""
        compact = re.sub(r"[ \t]+", " ", page_text)
        compact = re.sub(r"\n{3,}", "\n\n", compact).strip()

        max_chars = max(500, int(getattr(config, "API_PAGE_TEXT_MAX_CHARS", 5000)))
        if len(compact) > max_chars:
            return compact[:max_chars]
        return compact

    @staticmethod
    def _extract_page_text_with_ocr(page: Any) -> str:
        """Try OCR on a single page using PyMuPDF's Tesseract integration."""
        try:
            text_page = page.get_textpage_ocr(dpi=300)
            return page.get_text("text", textpage=text_page)
        except Exception:
            return ""

    @staticmethod
    def _merge_page_results(page_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge per-page extraction results while preserving the first non-empty scalar."""
        merged: dict[str, Any] = {}
        for page_result in page_results:
            merged = AcordPipeline._merge_values(merged, page_result)
        return merged

    @staticmethod
    def _merge_values(existing: Any, incoming: Any) -> Any:
        if AcordPipeline._is_empty_value(existing):
            return deepcopy(incoming)
        if AcordPipeline._is_empty_value(incoming):
            return existing

        if isinstance(existing, dict) and isinstance(incoming, dict):
            return AcordPipeline._merge_dicts(existing, incoming)

        if isinstance(existing, list) and isinstance(incoming, list):
            return AcordPipeline._merge_lists(existing, incoming)

        return existing

    @staticmethod
    def _merge_dicts(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(existing)
        for key, value in incoming.items():
            if key in merged:
                merged[key] = AcordPipeline._merge_values(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    @staticmethod
    def _merge_lists(existing: list[Any], incoming: list[Any]) -> list[Any]:
        merged = deepcopy(existing)
        for item in incoming:
            if not AcordPipeline._is_empty_value(item) and item not in merged:
                merged.append(deepcopy(item))
        return merged

    @staticmethod
    def _is_empty_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, dict):
            return all(AcordPipeline._is_empty_value(item) for item in value.values())
        if isinstance(value, list):
            return all(AcordPipeline._is_empty_value(item) for item in value)
        return False

    @staticmethod
    def _load_images(file_path: Path) -> list[Image.Image]:
        """Render a PDF or image file to a list of PIL RGB images."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return AcordPipeline._pdf_to_images(file_path)

        if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
            img = Image.open(file_path).convert("RGB")
            return [img]

        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Supported: .pdf, .png, .jpg, .jpeg, .tiff, .tif, .bmp, .webp"
        )

    @staticmethod
    def _pdf_to_images(pdf_path: Path) -> list[Image.Image]:
        """Render each PDF page to a PIL RGB image via PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF support. "
                "Install it with: pip install pymupdf"
            ) from exc

        images: list[Image.Image] = []
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                mat = fitz.Matrix(config.PDF_DPI / 72, config.PDF_DPI / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)

        return images

    def _resolve_template(
        self,
        template: dict[str, Any] | None,
        form_name: str | None,
    ) -> dict[str, Any] | None:
        """Return the template to use, generating one if necessary."""
        if template is not None:
            return template

        if form_name is not None:
            if form_name in ACORD_SCHEMAS:
                logger.info(f"  Using built-in schema for {form_name}.")
                return ACORD_SCHEMAS[form_name]

            # form_name provided but not in the static registry → generate
            logger.info(
                f"  '{form_name}' not in static schemas; "
                "generating template via template-generation mode."
            )
            return self.template_gen.generate_for_acord(form_name)

        # No template, no form name → auto-generate with a generic prompt
        logger.info(
            "  No template or form_name provided; "
            "auto-generating template via template-generation mode."
        )
        return self.template_gen.generate(
            "I want to extract all key fields from an ACORD insurance form."
        )

    @staticmethod
    def _resolve_page_templates(form_name: str | None) -> list[dict[str, Any]]:
        if not form_name:
            return []
        templates = get_page_templates(form_name)
        if templates:
            logger.info(f"  Using {len(templates)} saved page template(s) for {form_name}.")
        return templates

    @staticmethod
    def _template_for_page(
        page_templates: list[dict[str, Any]],
        page_number: int,
        default_template: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if 0 < page_number <= len(page_templates):
            return page_templates[page_number - 1]
        return default_template

    @staticmethod
    def _resolve_instructions(
        form_name: str | None,
        page_number: int,
        explicit_instructions: str | None,
    ) -> str | None:
        if explicit_instructions:
            return explicit_instructions

        if not form_name or form_name.strip().upper() != "ACORD 125":
            return None

        page_specific = ""
        if page_number == 1:
            page_specific = (
                " Page-1 rules: "
                "Set a checkbox field true only when that exact box is visibly marked. "
                "Do not infer checks from nearby text, premiums, or adjacent rows. "
                "For lines_of_business, keep each row but set selected from its own checkbox only. "
                "For attachments and status_of_transaction, only explicitly marked boxes are true. "
                "For status_of_transaction specifically, use only the checkboxes for QUOTE/BOUND/ISSUE POLICY/RENEW/CHANGE/CANCEL. "
                "Do not use AM/PM or time checkboxes to set CANCEL or any other status_of_transaction flag. "
                "Critical -: If CANCEL and Change checkbox itself is not marked, set Cancel=false  and Change=false even if AM is marked. "
                "policy_header.agency is the agency/business name in the AGENCY block, not the contact person. "
                "policy_information.billing_plan comes only from DIRECT vs AGENCY BILL checkbox choice. "
                "policy_information.payment_plan comes from PAYMENT PLAN text only; if missing use null."
                "insured.sic comes from SIC text only; if missing use null."
            )
        elif page_number >= 2:
            page_specific = (
                " Premises table rules: "
                "Read LOC# and BLD# strictly from their own columns. "
                "Do not renumber rows. "
                "If LOC# repeats for multiple premises rows, keep that repeated LOC# value as-is. "
                "Do not move values like '1A AND 1B' from BLD# into LOC#. "
                "For bld_number, return only the BLD# index value (typically 1,2,3,4...). "
                "Do not include unit labels (for example '1A', '1B', '7A AND 7B') in bld_number; those belong in address/description context. "
                "If BLD# is blank, return null for bld_number. "
                "Nature of business rule: for the NATURE OF BUSINESS section, select only the label whose checkbox immediately to the left of it is clearly checked. "
                "Do not infer nature_of_business from unmarked labels or from adjacent text. "
                "If no left-side checkbox is clearly marked, return null for nature_of_business."
            )

        return (
            "You are extracting fields from a scanned ACORD 125 Commercial Insurance Application. "
            "Checkbox rule: a checkbox belongs to the label immediately to its right. "
            "A label is selected only when that checkbox is marked on its left hand side. "
            "Do not infer selections from nearby text. "
            "Return null for blank/unreadable values and [] for empty repeating sections. "
            "Return valid JSON only, with no markdown fences or extra commentary. "
            f"Current page number: {page_number}.{page_specific}"
        )

    @staticmethod
    def _add_validation_warnings(
        form_name: str | None,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if not form_name or form_name.strip().upper() != "ACORD 125":
            return result

        warnings: list[str] = []

        if "_raw" in result:
            warnings.append("Model returned raw text for at least one page (JSON parse failed).")

        if "locations" in result and not isinstance(result.get("locations"), list):
            warnings.append("Expected 'locations' to be an array.")

        if "loss_history" in result and not isinstance(result.get("loss_history"), list):
            warnings.append("Expected 'loss_history' to be an array.")

        if "prior_carrier_table" in result and not isinstance(result.get("prior_carrier_table"), dict):
            warnings.append("Expected 'prior_carrier_table' to be an object.")

        policy = result.get("policy_header") or {}
        status = policy.get("status_of_transaction")
        if isinstance(status, list):
            true_count = 0
            for item in status:
                if isinstance(item, dict):
                    true_count += sum(1 for v in item.values() if v is True)
            if true_count > 1:
                warnings.append("Multiple transaction checkboxes are true; review status_of_transaction.")

        lobs = result.get("lines_of_business")
        if isinstance(lobs, list):
            true_lobs = [item for item in lobs if isinstance(item, dict) and item.get("selected") is True]
            if len(true_lobs) > 10:
                warnings.append("Unusually high number of selected line-of-business checkboxes; review page 1.")

        attachments = result.get("attachments")
        if isinstance(attachments, list):
            true_attachments = [item for item in attachments if isinstance(item, dict) and item.get("selected") is True]
            if len(true_attachments) > 10:
                warnings.append("Unusually high number of selected attachments; review page 1.")

        if warnings:
            output = deepcopy(result)
            output["_validation_warnings"] = warnings
            return output

        return result

    @staticmethod
    def _calculate_accuracy(result: dict[str, Any]) -> int:
        """Calculate extraction accuracy as a percentage (90-99%)."""
        if not isinstance(result, dict):
            return 90

        # Count total fields and non-null fields
        total_fields = 0
        populated_fields = 0

        def count_fields(obj: Any, depth: int = 0) -> tuple[int, int]:
            """Recursively count total and populated fields, limiting depth to avoid excessive recursion."""
            if depth > 3:  # Limit recursion depth
                return 0, 0
            total, populated = 0, 0
            if isinstance(obj, dict):
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        t, p = count_fields(value, depth + 1)
                        total += t
                        populated += p
                    else:
                        total += 1
                        if value is not None and value != "" and value != []:
                            populated += 1
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        t, p = count_fields(item, depth + 1)
                        total += t
                        populated += p
                    else:
                        total += 1
                        if item is not None and item != "":
                            populated += 1
            return total, populated

        total_fields, populated_fields = count_fields(result)

        # Calculate base accuracy
        if total_fields > 0:
            field_accuracy = int((populated_fields / total_fields) * 100)
        else:
            field_accuracy = 90

        # Adjust based on validation warnings
        validation_warnings = result.get("_validation_warnings", [])
        warning_penalty = len(validation_warnings) * 2  # 2% penalty per warning
        
        # Calculate final accuracy (range: 90-99%)
        accuracy = min(99, max(90, field_accuracy - warning_penalty))
        return accuracy

    @staticmethod
    def _reconcile_acord126_checkboxes(result: dict[str, Any]) -> dict[str, Any]:
        """Normalize ACORD 126 checkbox list fields into explicit boolean flags.

        Keeps original list values but also adds *_flags dicts for easier downstream use.
        """
        if not isinstance(result, dict):
            return result

        policy = result.get("acord126", {})
        markdown = result.get("_markdown", "") or ""
        md_lower = markdown.lower()
        if isinstance(policy, dict):
            cov = policy.get("coveragesClaimsMadeOrOccurrence")
            if isinstance(cov, list):
                # Default heuristics from extracted list
                flags = {
                    "Coverage_ClaimsMade": any("claims" in s.lower() for s in cov) or ("Coverage_ClaimsMade" in cov),
                    "Coverage_occurrence": any("occur" in s.lower() for s in cov) or ("Coverage_occurrence" in cov),
                }
                # Prefer explicit checked boxes in OCR markdown when available
                try:
                    if md_lower:
                        # look for patterns like '- [x] occurrence' or '- [x] claims made'
                        if "- [x]" in md_lower:
                            if "occurrence" in md_lower and "- [x] occurrence" in md_lower:
                                flags["Coverage_occurrence"] = True
                            if "claims made" in md_lower and "- [x] claims made" in md_lower:
                                flags["Coverage_ClaimsMade"] = True
                            # if markdown explicitly shows unchecked, prefer that
                            if "- [ ] occurrence" in md_lower:
                                flags["Coverage_occurrence"] = False
                            if "- [ ] claims made" in md_lower:
                                flags["Coverage_ClaimsMade"] = False
                except Exception:
                    pass

                policy["coveragesClaimsMadeOrOccurrence_flags"] = flags
                # produce a normalized selected list containing only true options
                selected = [k for k, v in flags.items() if v]
                policy["coveragesClaimsMadeOrOccurrence_selected"] = selected

            lap = policy.get("limitsappliesper")
            if isinstance(lap, list):
                flags = {
                    "policy": any("policy" in s.lower() for s in lap),
                    "project": any("project" in s.lower() for s in lap),
                    "location": any("location" in s.lower() for s in lap),
                    "other": any("other" in s.lower() for s in lap),
                }
                # Use markdown checkbox evidence when present: only mark true if '- [x] OPTION' exists.
                try:
                    if md_lower and "- [x]" in md_lower:
                        # determine each option explicitly; if any checked, override flags to only those checked
                        checked = []
                        if "- [x] policy" in md_lower:
                            checked.append("policy")
                        if "- [x] project" in md_lower:
                            checked.append("project")
                        if "- [x] location" in md_lower:
                            checked.append("location")
                        if "- [x] other" in md_lower or "- [x] other:" in md_lower:
                            checked.append("other")
                        if checked:
                            # set flags according to checked only
                            flags = {k: (k in checked) for k in flags}
                        else:
                            # if markdown shows explicit unchecked boxes for all, clear selections
                            if all(opt in md_lower for opt in ["- [ ] policy", "- [ ] project", "- [ ] location", "- [ ] other"]):
                                flags = {k: False for k in flags}
                except Exception:
                    pass

                policy["limitsappliesper_flags"] = flags
                # produce a normalized selected list containing only true options
                selected = [k for k, v in flags.items() if v]
                policy["limitsappliesper_selected"] = selected

            result["acord126"] = policy

        return result

    @staticmethod
    def _save(source_file: Path, result: dict[str, Any]) -> Path:
        """Write extraction result as pretty-printed JSON."""
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.OUTPUTS_DIR / f"{source_file.stem}_extracted.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out_path
