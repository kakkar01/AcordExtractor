"""
download_model.py — run this ONCE before production deployment.

    python download_model.py

Downloads NuExtract3 from Hugging Face and stores it under ./models/NuExtract3/
so that subsequent runs never need an internet connection.
"""

import sys

from src.logging_fallback import logger

import config
from src.model_manager import download_model, is_model_cached


def main() -> None:
    if is_model_cached():
        logger.info(
            f"Model already cached at '{config.MODEL_DIR}'. "
            "Delete the directory and re-run to force a fresh download."
        )
        sys.exit(0)

    download_model()
    logger.success(
        f"NuExtract3 downloaded successfully to '{config.MODEL_DIR}'.\n"
        "You can now run the extractor offline."
    )


if __name__ == "__main__":
    main()
