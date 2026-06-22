"""
Model manager — downloads NuExtract3 from Hugging Face once and
keeps the weights on disk for fast, server-free production use.
"""

from __future__ import annotations

from pathlib import Path

import torch
from src.logging_fallback import logger
from transformers import AutoModelForImageTextToText, AutoProcessor

import config


def is_model_cached(model_dir: Path | None = None) -> bool:
    """Return True if the model weights are already on disk."""
    target = model_dir or config.MODEL_DIR
    return target.exists() and any(target.iterdir())


def download_model(model_dir: Path | None = None) -> None:
    """
    Download NuExtract3 from Hugging Face and save it to *model_dir*.

    Run this once before production deployment:
        python download_model.py
    """
    target = model_dir or config.MODEL_DIR
    target.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading '{config.HF_MODEL_ID}' → {target}")

    processor = AutoProcessor.from_pretrained(
        config.HF_MODEL_ID,
        trust_remote_code=True,
    )
    processor.save_pretrained(str(target))
    logger.info("Processor saved.")

    model = AutoModelForImageTextToText.from_pretrained(
        config.HF_MODEL_ID,
        dtype=config.TORCH_DTYPE,
        device_map=config.DEVICE_MAP,
        trust_remote_code=True,
    )
    model.save_pretrained(str(target))
    logger.info(f"Model weights saved to {target}.")


def load_model(
    model_dir: Path | None = None,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Load NuExtract3 from the local *model_dir*.

    If the weights are not present, raises FileNotFoundError rather than
    silently downloading — call ``download_model()`` (or run
    ``download_model.py``) first.
    """
    source = model_dir or config.MODEL_DIR

    if not is_model_cached(source):
        raise FileNotFoundError(
            f"Model weights not found at '{source}'. "
            "Run `python download_model.py` to download them first."
        )

    logger.info(f"Loading NuExtract3 from local path: {source}")

    processor = AutoProcessor.from_pretrained(
        str(source),
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(source),
        dtype=config.TORCH_DTYPE,
        device_map=config.DEVICE_MAP,
        trust_remote_code=True,
    ).eval()

    logger.info("Model loaded and ready.")
    return model, processor
