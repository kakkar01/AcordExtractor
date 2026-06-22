"""
Core NuExtract3 extractor — runs entirely local via Hugging Face Transformers.
No server required.

Key design:
  - ``apply_chat_template`` receives the same kwargs that vLLM's
    ``chat_template_kwargs`` / ``extra_body`` would, keeping the interface
    consistent with the vLLM docs.
  - ``enable_thinking=False``  → fast, deterministic (recommended for production)
  - ``enable_thinking=True``   → reasoning mode for difficult documents
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from src.logging_fallback import logger
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import config


class NuExtractor:
    """
    Thin wrapper around NuExtract3 for local, server-free inference.

    Parameters
    ----------
    model : AutoModelForImageTextToText
        Already-loaded model (call ``src.model_manager.load_model()``).
    processor : AutoProcessor
        Corresponding processor / tokenizer.
    """

    def __init__(
        self,
        model: AutoModelForImageTextToText,
        processor: AutoProcessor,
    ) -> None:
        self.model = model
        self.processor = processor

    _OCR_MARKDOWN_INSTRUCTIONS = (
        "Transcribe the document to Markdown literally. "
        "For checkboxes, output checked only when a visible mark is present in that exact box. "
        "Do not infer checkmarks from nearby text, table alignment, or premiums. "
        "If checkbox state is unclear, leave it unchecked."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def extract(
        self,
        images: list[Image.Image] | None = None,
        text: str | None = None,
        template: dict[str, Any] | None = None,
        instructions: str | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """
        Run structured extraction and return a parsed JSON dict.

        Parameters
        ----------
        images : list of PIL images, optional
            Pages of the document (one image per page for multi-page PDFs).
        text : str, optional
            Plain-text content of the document (use instead of / alongside images).
        template : dict, optional
            NuExtract JSON template.  If omitted the model returns raw Markdown.
        instructions : str, optional
            Free-text instructions appended to the template context.
        enable_thinking : bool
            ``True`` activates chain-of-thought reasoning (better for complex
            layouts; slower and slightly non-deterministic).
        max_new_tokens : int
            Hard cap on generated tokens (thinking + answer combined).

        Returns
        -------
        dict
            Parsed extraction result.  If the model output is not valid JSON,
            the raw string is returned under the key ``"_raw"``.
        """
        messages = self._build_messages(images=images, text=text)

        # chat_template_kwargs mirror the vLLM extra_body schema exactly
        chat_template_kwargs: dict[str, Any] = {
            "enable_thinking": enable_thinking,
        }
        if template is not None:
            chat_template_kwargs["template"] = json.dumps(template, indent=4)
        if instructions is not None:
            chat_template_kwargs["instructions"] = instructions

        raw = self._run(
            messages=messages,
            chat_template_kwargs=chat_template_kwargs,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
        )

        return self._parse_output(raw, template)

    def ocr(
        self,
        images: list[Image.Image],
        enable_thinking: bool = False,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
    ) -> str:
        """
        Convert document images to clean Markdown (content/OCR mode).

        Returns the Markdown string directly.
        """
        messages = self._build_messages(images=images)

        chat_template_kwargs: dict[str, Any] = {
            "mode": "markdown",
            "enable_thinking": enable_thinking,
            "instructions": self._OCR_MARKDOWN_INSTRUCTIONS,
        }

        return self._run(
            messages=messages,
            chat_template_kwargs=chat_template_kwargs,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
        )

    def generate_template(
        self,
        description: str,
        max_new_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Generate a NuExtract JSON template from a natural-language description.

        This mirrors the vLLM ``extra_body`` schema:
            extra_body={
                "chat_template_kwargs": {
                    "mode": "template-generation"
                }
            }

        Parameters
        ----------
        description : str
            Natural-language description of the document and what to extract.

        Returns
        -------
        dict
            Generated template as a Python dict (ready to pass back to
            ``extract()``).
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": description},
                ],
            }
        ]

        # This is the exact mode used in the official NuExtract docs
        chat_template_kwargs: dict[str, Any] = {
            "mode": "template-generation",
        }

        raw = self._run(
            messages=messages,
            chat_template_kwargs=chat_template_kwargs,
            enable_thinking=False,
            max_new_tokens=max_new_tokens,
        )

        return self._parse_output(raw)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        images: list[Image.Image] | None = None,
        text: str | None = None,
    ) -> list[dict]:
        """Construct the messages list for apply_chat_template."""
        content: list[dict] = []

        if images:
            for img in images:
                content.append({"type": "image", "image": img})

        if text:
            content.append({"type": "text", "text": text})

        if not content:
            raise ValueError("Provide at least one of: images, text.")

        return [{"role": "user", "content": content}]

    def _run(
        self,
        messages: list[dict],
        chat_template_kwargs: dict[str, Any],
        enable_thinking: bool,
        max_new_tokens: int,
    ) -> str:
        """Tokenise → generate → decode; return the raw model string."""
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **chat_template_kwargs,
        ).to(self.model.device)

        temperature = (
            config.TEMPERATURE_THINK if enable_thinking
            else config.TEMPERATURE_NO_THINK
        )
        do_sample = enable_thinking  # deterministic when thinking is off

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

        # Strip the prompt tokens
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]

        raw = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        # Remove thinking trace if present
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        return raw

    @staticmethod
    def _parse_output(raw: str, template: dict[str, Any] | None = None) -> dict[str, Any]:
        """Try to parse *raw* as JSON; if template requests ACORD 140,
        validate top-level keys and return a diagnostic on mismatch.
        """
        # Strip common markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[-1] if cleaned.count("```") >= 2 else cleaned
            cleaned = cleaned.lstrip("json").strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Model output is not valid JSON; returning raw string.")
            return {"_raw": raw}

        try:
            form_name = template.get("form_name") if isinstance(template, dict) else None
        except Exception:
            form_name = None

        if form_name == "ACORD 140":
            expected = set(template.keys()) if isinstance(template, dict) else set()
            parsed_keys = set(parsed.keys()) if isinstance(parsed, dict) else set()
            if expected and not expected.issubset(parsed_keys):
                # Attempt to extract the first JSON object and reparse
                start = cleaned.find("{")
                if start != -1:
                    salvaged = cleaned[start:]
                    try:
                        reparsed = json.loads(salvaged)
                        reparsed_keys = set(reparsed.keys()) if isinstance(reparsed, dict) else set()
                        if expected.issubset(reparsed_keys):
                            return reparsed
                        else:
                            return {
                                "_schema_mismatch": True,
                                "expected_keys": sorted(list(expected)),
                                "parsed_keys": sorted(list(parsed_keys)),
                                "_parsed": parsed,
                            }
                    except Exception:
                        return {"_schema_mismatch": True, "_parsed": parsed}

        return parsed
