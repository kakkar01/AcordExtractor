"""
Template generator — wraps NuExtractor.generate_template() to produce
ACORD-oriented NuExtract JSON templates from natural-language descriptions.

Usage example
-------------
    from src.model_manager import load_model
    from src.template_generator import TemplateGenerator

    model, processor = load_model()
    gen = TemplateGenerator(model, processor)

    template = gen.generate("ACORD 25 Certificate of Liability Insurance")
    print(template)
"""

from __future__ import annotations

from typing import Any

from src.logging_fallback import logger

from src.extractor import NuExtractor


# Pre-written prompts for common ACORD forms so the user can generate
# refined templates without crafting the description from scratch.
ACORD_DESCRIPTIONS: dict[str, str] = {
    "ACORD 25": (
        "I want to extract all key fields from an ACORD 25 Certificate of "
        "Liability Insurance form.  The form contains producer information "
        "(name, address, phone, fax, email), insured information (name, "
        "address), coverage details (type, insurer, NAIC number, policy "
        "number, effective and expiration dates, all monetary limits), "
        "certificate holder information, description of operations, "
        "cancellation notice days, and the authorized representative name."
    ),
    "ACORD 125": (
        "I want to extract all key fields from an ACORD 125 Commercial "
        "Insurance Application.  The form includes applicant details "
        "(name, address, FEIN, business type, years in business), proposed "
        "policy dates, coverages requested, description of operations, "
        "list of premises with employee counts and revenues, and prior "
        "loss history."
    ),
    "ACORD 130": (
        "I want to extract all key fields from an ACORD 130 Workers "
        "Compensation Application.  The form includes applicant name, "
        "FEIN, address, policy dates, state, experience modification "
        "factor, class codes with annual remuneration, prior carrier "
        "information, and loss history."
    ),
    "ACORD 140": (
        "I want to extract all key fields from an ACORD 140 Property "
        "Section form.  The form contains location details, building "
        "characteristics (year built, stories, area, construction type, "
        "occupancy, protection class, sprinkler and alarm information), "
        "coverage items (type, limit, cause of loss, deductible, "
        "valuation), and mortgage/lienholder information."
    ),
}


class TemplateGenerator:
    """
    Generates NuExtract JSON templates using the ``template-generation`` mode.

    This mirrors the vLLM ``extra_body`` request:
        extra_body={
            "chat_template_kwargs": {
                "mode": "template-generation"
            }
        }

    but runs entirely locally via Hugging Face Transformers.
    """

    def __init__(self, extractor: Any) -> None:
        self.extractor = extractor

    def generate(
        self,
        description: str,
        max_new_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Generate a NuExtract template from a free-form description.

        Parameters
        ----------
        description : str
            Natural-language description of the document type and desired fields.
        max_new_tokens : int
            Token budget for the generated template.

        Returns
        -------
        dict
            Generated template dict, ready to pass to ``NuExtractor.extract()``.
        """
        logger.info("Generating template via template-generation mode…")
        template = self.extractor.generate_template(
            description=description,
            max_new_tokens=max_new_tokens,
        )
        logger.info(f"Template generated with {len(template)} top-level keys.")
        return template

    def generate_for_acord(self, form_name: str) -> dict[str, Any]:
        """
        Generate a template using a pre-written ACORD form description.

        Parameters
        ----------
        form_name : str
            One of ``"ACORD 25"``, ``"ACORD 125"``, ``"ACORD 130"``,
            ``"ACORD 140"``.

        Returns
        -------
        dict
            Generated template dict.

        Raises
        ------
        KeyError
            If *form_name* is not in the built-in descriptions.
        """
        if form_name not in ACORD_DESCRIPTIONS:
            available = list(ACORD_DESCRIPTIONS.keys())
            raise KeyError(
                f"Unknown ACORD form '{form_name}'. "
                f"Available: {available}"
            )

        description = ACORD_DESCRIPTIONS[form_name]
        logger.info(f"Generating template for {form_name}…")
        return self.generate(description)
