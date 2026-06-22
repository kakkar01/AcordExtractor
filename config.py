"""
Central configuration for the ACORD Form Extraction project.
All paths and model settings are controlled here.
"""

from pathlib import Path
import os


# Root of this project
PROJECT_ROOT = Path(__file__).parent.resolve()

try:
	from dotenv import load_dotenv
except Exception:
	load_dotenv = None

if load_dotenv is not None:
	load_dotenv(PROJECT_ROOT / ".env")


def _env_flag(name: str, default: str = "0") -> bool:
	return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

# ─── Paths ────────────────────────────────────────────────────────────────────

# Local directory where the model weights are stored (no re-download in production)
MODEL_DIR = PROJECT_ROOT / "models" / "NuExtract3"

# Hugging Face model ID used only during the first download
HF_MODEL_ID = "numind/NuExtract3"

# Inference backend: "local" (Transformers) or "api" (OpenAI-compatible HTTP API)
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "local").lower()

# API backend settings (used only when INFERENCE_BACKEND="api")
API_BASE_URL = os.getenv("API_BASE_URL", "")
API_KEY = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
API_MODEL = os.getenv("API_MODEL", HF_MODEL_ID)
API_INPUT_MODE = os.getenv("API_INPUT_MODE", "completion").lower()
API_MAX_NEW_TOKENS = int(os.getenv("API_MAX_NEW_TOKENS", "4096"))
API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "90"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "1"))
API_PAGE_TEXT_MAX_CHARS = int(os.getenv("API_PAGE_TEXT_MAX_CHARS", "5000"))
OCR_MAX_NEW_TOKENS = int(os.getenv("OCR_MAX_NEW_TOKENS", "4096"))
API_IMAGE_MAX_DIM = int(os.getenv("API_IMAGE_MAX_DIM", "1280"))
API_IMAGE_JPEG_QUALITY = int(os.getenv("API_IMAGE_JPEG_QUALITY", "90"))
API_IMAGE_JPEG_SUBSAMPLING = int(os.getenv("API_IMAGE_JPEG_SUBSAMPLING", "0"))
API_SAVE_REQUEST_IMAGES = _env_flag("API_SAVE_REQUEST_IMAGES", "0")
API_REQUEST_IMAGES_DIR = Path(
	os.getenv("API_REQUEST_IMAGES_DIR", str(PROJECT_ROOT / "data" / "debug_images"))
)
API_FIELD_REASONING = _env_flag("API_FIELD_REASONING", "0")
API_FIELD_REASONING_MAX = int(os.getenv("API_FIELD_REASONING_MAX", "12"))
API_FIELD_CONFIDENCE = _env_flag("API_FIELD_CONFIDENCE", "0")
API_FIELD_CONFIDENCE_MAX_FIELDS = int(os.getenv("API_FIELD_CONFIDENCE_MAX_FIELDS", "120"))

# Docling OCR settings
DOCLING_SERVE_URL = os.getenv("DOCLING_SERVE_URL", "").strip().rstrip("/")
DOCLING_SERVE_API_KEY = os.getenv("DOCLING_SERVE_API_KEY", "").strip()
DOCLING_SERVE_CONVERT_PATH = (
	os.getenv("DOCLING_SERVE_CONVERT_PATH", "/v1/convert/source").strip()
	or "/v1/convert/source"
)
DOCLING_TIMEOUT_SECONDS = float(os.getenv("DOCLING_TIMEOUT_SECONDS", "120"))
DOCLING_USE_LOCAL = _env_flag("DOCLING_USE_LOCAL", "1")

# ACORD 125 page-1 LOB refinement crop/debug settings
LOB_REFINEMENT_CROP = _env_flag("LOB_REFINEMENT_CROP", "0")
LOB_CROP_LEFT_RATIO = float(os.getenv("LOB_CROP_LEFT_RATIO", "0.02"))
LOB_CROP_TOP_RATIO = float(os.getenv("LOB_CROP_TOP_RATIO", "0.27"))
LOB_CROP_RIGHT_RATIO = float(os.getenv("LOB_CROP_RIGHT_RATIO", "0.98"))
LOB_CROP_BOTTOM_RATIO = float(os.getenv("LOB_CROP_BOTTOM_RATIO", "0.57"))
LOB_SAVE_CROP_IMAGE = _env_flag("LOB_SAVE_CROP_IMAGE", "0")
LOB_CROP_IMAGES_DIR = Path(
	os.getenv("LOB_CROP_IMAGES_DIR", str(PROJECT_ROOT / "data" / "debug_lob_crops"))
)
API_SAVE_LOB_CROP = _env_flag("API_SAVE_LOB_CROP", "0")
API_LOB_CROP_DIR = Path(
	os.getenv("API_LOB_CROP_DIR", str(PROJECT_ROOT / "data" / "debug_images" / "lob_crops"))
)
API_LOB_CROP_LEFT = float(os.getenv("API_LOB_CROP_LEFT", "0.02"))
API_LOB_CROP_TOP = float(os.getenv("API_LOB_CROP_TOP", "0.28"))
API_LOB_CROP_RIGHT = float(os.getenv("API_LOB_CROP_RIGHT", "0.98"))
API_LOB_CROP_BOTTOM = float(os.getenv("API_LOB_CROP_BOTTOM", "0.56"))

# Where input ACORD form images / PDFs live
FORMS_DIR = PROJECT_ROOT / "data" / "forms"

# Where extraction results (JSON) are written
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

# Where reusable page-wise extraction templates are stored
PAGE_TEMPLATES_DIR = PROJECT_ROOT / "schemas" / "page_templates"

# ─── Inference settings ───────────────────────────────────────────────────────

# Render PDF pages at this DPI before passing to the model
PDF_DPI: int = 170

# Maximum tokens the model may generate (thinking + answer combined)
MAX_NEW_TOKENS: int = 768

# Temperature for non-thinking (deterministic) extraction
TEMPERATURE_NO_THINK: float = 0.0

# Temperature for thinking (reasoning) extraction
TEMPERATURE_THINK: float = 0.0

# Device map passed to from_pretrained(); "auto" lets accelerate choose
DEVICE_MAP: str = "auto"

# PyTorch dtype for the model weights (only relevant for local backend)
try:
	import torch  # noqa: E402
	TORCH_DTYPE = torch.bfloat16
except Exception:  # pragma: no cover - API-only environments may not have torch
	TORCH_DTYPE = "auto"
