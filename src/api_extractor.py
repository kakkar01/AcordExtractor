"""API-backed NuExtract wrapper for a vLLM completion endpoint."""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from src.logging_fallback import logger
from openai import OpenAI
from PIL import Image

import config


class ApiNuExtractor:
    """NuExtract client for completion-text or chat-image API transports."""

    _THINK_END = "</think>"
    _saved_image_counter = 0

    _PROMPT_TEMPLATE = """<|input|>
### Template:
{template}

### Text:
{text}

### Instructions:
{instructions}

<|output|>"""

    _OCR_MARKDOWN_INSTRUCTIONS = (
        "Transcribe the document to Markdown literally. "
        "For checkboxes, output checked only when a visible mark is present in that exact box. "
        "Do not infer checkmarks from nearby text, table alignment, or premiums. "
        "If checkbox state is unclear, leave it unchecked."
    )

    def __init__(self, base_url: str, model: str, api_key: str | None = None) -> None:
        if not base_url:
            raise ValueError(
                "API_BASE_URL is required for API mode. "
                "Set INFERENCE_BACKEND=api and API_BASE_URL in your environment."
            )

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "dummy",
            timeout=config.API_TIMEOUT_SECONDS,
            max_retries=config.API_MAX_RETRIES,
        )
        self.model = model
        self.input_mode = config.API_INPUT_MODE
        if self.input_mode not in {"completion", "chat"}:
            raise ValueError("API_INPUT_MODE must be either 'completion' or 'chat'.")

        self.supports_image_inputs = self.input_mode == "chat"
        self.prefer_page_wise = True

    def extract(
        self,
        images: list[Any] | None = None,
        text: str | None = None,
        template: dict[str, Any] | None = None,
        instructions: str | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int = config.API_MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        if template is None:
            raise ValueError("A template is required for API extraction mode.")

        if self.supports_image_inputs:
            if not images:
                raise ValueError("Image input is required when API_INPUT_MODE=chat.")
            raw = self._run_chat(
                images=images,
                text=text,
                template=template,
                instructions=instructions,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
            )
        else:
            if images is not None:
                raise ValueError(
                    "This API deployment expects extracted document text, not images. "
                    "Set API_INPUT_MODE=chat if your server supports vision chat requests."
                )
            if not text:
                raise ValueError("Text input is required for API extraction mode.")

            prompt = self._build_prompt(
                text=text,
                template=template,
                instructions=instructions,
            )
            raw = self._run_completion(
                prompt=prompt,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
            )
        return self._parse_output(raw, template)

    def ocr(
        self,
        images: list[Any],
        enable_thinking: bool = False,
        max_new_tokens: int = config.API_MAX_NEW_TOKENS,
    ) -> str:
        """Convert document images to Markdown via the chat API (markdown mode)."""
        if self.input_mode != "chat":
            raise NotImplementedError(
                "OCR mode requires API_INPUT_MODE=chat. "
                "Switch to chat mode or use the local backend."
            )

        bounded_max_tokens = min(max_new_tokens, config.API_MAX_NEW_TOKENS)
        temperature = (
            config.TEMPERATURE_THINK if enable_thinking
            else config.TEMPERATURE_NO_THINK
        )

        content: list[dict[str, Any]] = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise ValueError("OCR mode image inputs must be PIL images.")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(image)},
                }
            )

        content.append(
            {
                "type": "text",
                "text": self._OCR_MARKDOWN_INSTRUCTIONS,
            }
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=bounded_max_tokens,
            temperature=temperature,
            extra_body={
                "chat_template_kwargs": {
                    "mode": "markdown",
                    "enable_thinking": enable_thinking,
                    "instructions": self._OCR_MARKDOWN_INSTRUCTIONS,
                }
            },
        )

        if not response.choices:
            raise RuntimeError("API response did not contain any choices.")

        raw = self._content_to_string(response.choices[0].message.content)
        cleaned, _ = self._clean_model_answer(raw)
        return cleaned.strip()

    def generate_template(
        self,
        description: str,
        max_new_tokens: int = 2048,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Template generation is not enabled for this API deployment. "
            "Use a built-in schema with --form or local backend."
        )

    def _build_prompt(
        self,
        text: str,
        template: dict[str, Any],
        instructions: str | None,
    ) -> str:
        # If the template explicitly identifies ACORD 140, append a
        # strict JSON-only enforcement marker so downstream callers can
        # force deterministic generation and validate the output.
        extra_instructions = instructions or "Extract all fields exactly as described by the template."
        try:
            form_name = template.get("form_name") if isinstance(template, dict) else None
        except Exception:
            form_name = None

        if form_name == "ACORD 140":
            strict_text = (
                "\n\nIMPORTANT: RETURN ONLY a single JSON object that EXACTLY matches the"
                " template keys for ACORD 140. Do NOT include any explanatory text, markdown,"
                " or extra fields. Use null for unknown values. End the output with the"
                " sentinel string STRICT_JSON_ENFORCE:ACORD_140 and nothing else."
            )
            extra_instructions = extra_instructions + strict_text

        return self._PROMPT_TEMPLATE.format(
            template=json.dumps(template, ensure_ascii=False, separators=(",", ":")),
            text=text,
            instructions=extra_instructions,
        )

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        max_dim = max(256, int(config.API_IMAGE_MAX_DIM))
        img = image.convert("RGB")
        width, height = img.size
        largest = max(width, height)
        if largest > max_dim:
            scale = max_dim / float(largest)
            resized = (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            )
            img = img.resize(resized, Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(
            buf,
            format="JPEG",
            quality=max(40, min(100, int(config.API_IMAGE_JPEG_QUALITY))),
            subsampling=max(0, min(2, int(config.API_IMAGE_JPEG_SUBSAMPLING))),
            optimize=True,
        )

        jpeg_bytes = buf.getvalue()
        if config.API_SAVE_REQUEST_IMAGES:
            try:
                out_dir: Path = config.API_REQUEST_IMAGES_DIR
                out_dir.mkdir(parents=True, exist_ok=True)
                ApiNuExtractor._saved_image_counter += 1
                out_path = out_dir / f"request_image_{ApiNuExtractor._saved_image_counter:05d}.jpg"
                out_path.write_bytes(jpeg_bytes)
            except Exception as exc:  # pragma: no cover - debug-only path
                logger.warning(f"Could not save API request image: {exc}")

        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _run_completion(
        self,
        prompt: str,
        enable_thinking: bool,
        max_new_tokens: int,
    ) -> str:
        bounded_max_tokens = min(max_new_tokens, config.API_MAX_NEW_TOKENS)
        # Force deterministic temperature(=0) for strict ACORD 140 requests
        if "STRICT_JSON_ENFORCE:ACORD_140" in prompt:
            temperature = 0.0
        else:
            temperature = (
                config.TEMPERATURE_THINK if enable_thinking
                else config.TEMPERATURE_NO_THINK
            )

        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=bounded_max_tokens,
            temperature=temperature,
        )

        if not response.choices:
            raise RuntimeError("API response did not contain any choices.")

        choice = response.choices[0]
        raw = choice.text or ""
        if getattr(choice, "finish_reason", None) == "length":
            logger.warning(
                "Completion response hit max_tokens and may be truncated. "
                f"Increase API_MAX_NEW_TOKENS (current cap: {bounded_max_tokens})."
            )

        raw, _ = self._clean_model_answer(raw)

        return raw.strip()

    def _run_chat(
        self,
        images: list[Any],
        text: str | None,
        template: dict[str, Any],
        instructions: str | None,
        enable_thinking: bool,
        max_new_tokens: int,
    ) -> str:
        bounded_max_tokens = min(max_new_tokens, config.API_MAX_NEW_TOKENS)
        temperature = (
            config.TEMPERATURE_THINK if enable_thinking
            else config.TEMPERATURE_NO_THINK
        )

        content: list[dict[str, Any]] = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise ValueError("Chat mode image inputs must be PIL images.")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(image)},
                }
            )

        if text:
            content.append({"type": "text", "text": text})

        # If the instructions include the strict ACORD 140 marker, force
        # deterministic generation (temperature=0.0) for JSON-only output.
        marker_in_instructions = False
        if instructions and "STRICT_JSON_ENFORCE:ACORD_140" in instructions:
            marker_in_instructions = True

        if marker_in_instructions:
            temperature = 0.0

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=bounded_max_tokens,
            temperature=temperature,
            extra_body={
                "chat_template_kwargs": {
                    "template": json.dumps(template, ensure_ascii=False, separators=(",", ":")),
                    "instructions": instructions
                    or "Extract all fields exactly as described by the template.",
                    "enable_thinking": enable_thinking,
                }
            },
        )

        if not response.choices:
            raise RuntimeError("API response did not contain any choices.")

        choice = response.choices[0]
        message = choice.message
        raw = self._content_to_string(message.content)
        if getattr(choice, "finish_reason", None) == "length":
            logger.warning(
                "Chat response hit max_tokens and may be truncated. "
                f"Increase API_MAX_NEW_TOKENS (current cap: {bounded_max_tokens}) "
                "or further reduce image size/quality."
            )

        raw, _ = self._clean_model_answer(raw)

        return raw.strip()

    @staticmethod
    def _parse_output(raw: str, template: dict[str, Any] | None = None) -> dict[str, Any]:
        """Parse model raw output to JSON and enforce ACORD 140 schema when requested.

        If strict ACORD 140 enforcement is requested via the template, attempt
        to salvage JSON from the output, validate top-level keys against the
        provided template, and return a diagnostic object on mismatch.
        """
        try:
            cleaned, _ = ApiNuExtractor._clean_model_answer(raw)
            try:
                parsed = ApiNuExtractor._parse_json_safely(cleaned)
            except Exception:
                # Direct parse failed — model may have appended trailing text
                # after the closing brace (common on page 1 of dense forms).
                # Try extracting the first complete JSON object before giving up.
                salvaged = ApiNuExtractor._extract_first_json_object(cleaned)
                parsed = ApiNuExtractor._parse_json_safely(salvaged)
        except Exception:
            logger.warning("Model output is not valid JSON; returning raw string.")
            return {"_raw": raw}

        # If the template requests strict ACORD 140 enforcement, validate
        # the top-level keys are present. If not, attempt to extract the
        # first JSON object and reparsing once more before returning a
        # schema-mismatch diagnostic.
        try:
            form_name = template.get("form_name") if isinstance(template, dict) else None
        except Exception:
            form_name = None

        if form_name == "ACORD 140":
            expected = set(template.keys()) if isinstance(template, dict) else set()
            parsed_keys = set(parsed.keys()) if isinstance(parsed, dict) else set()
            if expected and not expected.issubset(parsed_keys):
                # Try to salvage a JSON object from noisy text
                salvaged = ApiNuExtractor._extract_first_json_object(cleaned)
                try:
                    reparsed = ApiNuExtractor._parse_json_safely(salvaged)
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

    @staticmethod
    def _content_to_string(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks).strip()
        return ""

    @staticmethod
    def _clean_model_answer(answer: str) -> tuple[str, str | None]:
        reasoning: str | None = None
        text = answer.strip()

        if ApiNuExtractor._THINK_END in text:
            think_part, _, text = text.partition(ApiNuExtractor._THINK_END)
            reasoning = think_part.replace("<think>", "").strip()
            text = text.strip()

        text = re.sub(r"<\|.*?\|>", "", text).strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        return text, reasoning

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        start = text.find("{")
        if start == -1:
            return text

        depth = 0
        in_string = False
        escape = False

        for index in range(start, len(text)):
            ch = text[index]

            if escape:
                escape = False
                continue

            if ch == "\\":
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]

        return text[start:]

    @staticmethod
    def _parse_json_safely(answer: str) -> dict[str, Any]:
        # Pass 1: standard json.loads
        try:
            parsed = json.loads(answer)
            return parsed if isinstance(parsed, dict) else {"_raw": answer}
        except json.JSONDecodeError:
            pass

        # Pass 2: extract the outermost JSON object then retry
        try:
            cleaned = ApiNuExtractor._extract_first_json_object(answer)
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {"_raw": answer}
        except (json.JSONDecodeError, ValueError):
            pass

        # Pass 3: json-repair handles common LLM syntax errors such as
        # swapped ]} vs }], missing quotes, trailing commas, etc.
        try:
            from json_repair import repair_json
            repaired = repair_json(answer, return_objects=True)
            if isinstance(repaired, dict) and repaired:
                return repaired
        except ImportError:
            pass
        except Exception:
            pass

        raise json.JSONDecodeError("All JSON parse attempts failed", answer, 0)
