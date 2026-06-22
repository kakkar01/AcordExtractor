# ACORD Form Extractor

ACORD insurance form extraction powered by **NuExtract3** (4B VLM) for structured extraction, with optional Docling OCR for Markdown/layout output.

## Features

| Capability | Details |
|---|---|
| **Structured extraction** | Document → JSON using built-in ACORD schemas |
| **Template generation** | Auto-generate schemas from natural language (`mode: "template-generation"`) |
| **OCR mode** | Document → Markdown via Docling Serve or local Docling when configured |
| **Reasoning mode** | Chain-of-thought for complex / multi-column layouts |
| **Local / offline** | Model weights stored on disk — no API calls at inference time |
| **Remote API mode** | Use an OpenAI-compatible endpoint instead of local model loading |
| **Multi-page PDF** | PyMuPDF renders each page before passing to the model |

### Supported ACORD Forms (built-in schemas)

| Form | Description |
|---|---|
| ACORD 25 | Certificate of Liability Insurance |
| ACORD 125 | Commercial Insurance Application |
| ACORD 130 | Workers Compensation Application |
| ACORD 140 | Property Section |

---

## Project Structure

```
acord-extractor/
├── config.py                  # All paths and model settings
├── download_model.py          # One-time model download script
├── main.py                    # CLI entry point
├── requirements.txt
├── models/
│   └── NuExtract3/            # Model weights stored here (git-ignored)
├── data/
│   ├── forms/                 # Drop input PDFs / images here
│   └── outputs/               # Extracted JSON / Markdown results
├── schemas/
│   ├── __init__.py
│   └── acord_schemas.py       # NuExtract JSON templates for ACORD forms
└── src/
    ├── __init__.py
    ├── model_manager.py       # Download & load model from local disk
    ├── extractor.py           # Core NuExtract3 wrapper (Transformers)
    ├── template_generator.py  # Template-generation mode wrapper
    └── pipeline.py            # End-to-end extraction pipeline
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you are using only the remote API backend, you can install a lighter set:

```bash
pip install -r requirements-api.txt
```

> **GPU recommended** — NuExtract3 is a 4B model.  
> CPU inference is possible but significantly slower.

### 2. Download the model (once)

```bash
python download_model.py
```

Weights are saved to `./models/NuExtract3/` and never downloaded again.

If you prefer **remote inference** (no local model weights / CPU model load), skip this step and use API mode.

### 3. Extract an ACORD form

```bash
# Using the built-in ACORD 25 schema
python main.py extract data/forms/cert.pdf --form "ACORD 25"

# Enable reasoning for complex/scanned documents
python main.py extract data/forms/cert.pdf --form "ACORD 25" --thinking

# Auto-generate template and extract (no schema needed)
python main.py extract data/forms/cert.pdf

# OCR only — convert to Markdown
python main.py ocr data/forms/cert.pdf

# Use remote API backend instead of local model
python main.py --backend api extract data/forms/cert.pdf --form "ACORD 25"
```

### API backend setup (optional)

Set environment variables for your OpenAI-compatible NuExtract endpoint:

```powershell
$env:INFERENCE_BACKEND="api"
$env:API_BASE_URL="https://your-endpoint.example.com/v1"
$env:API_KEY="your-token-if-required"
$env:API_MODEL="numind/NuExtract3"
$env:API_IMAGE_MAX_DIM="1400"
$env:API_IMAGE_JPEG_QUALITY="90"
$env:API_IMAGE_JPEG_SUBSAMPLING="0"
$env:API_SAVE_REQUEST_IMAGES="1"
$env:API_REQUEST_IMAGES_DIR="data/debug_images"
$env:LOB_REFINEMENT_CROP="1"
$env:LOB_SAVE_CROP_IMAGE="1"
$env:LOB_CROP_IMAGES_DIR="data/debug_lob_crops"
```

Then run as usual:

```bash
python main.py extract data/forms/cert.pdf --form "ACORD 25"
```

Or explicitly force backend per command:

```bash
python main.py --backend api extract data/forms/cert.pdf --form "ACORD 25"
python main.py --backend local extract data/forms/cert.pdf --form "ACORD 25"
```

You can also store these variables in a repo-local `.env` file. The app loads `.env` automatically on startup.

### Docling OCR setup (optional)

Set environment variables to route OCR and `_markdown` generation through Docling. A hosted `docling-serve` instance is preferred when local CPU inference is too slow:

```powershell
$env:DOCLING_SERVE_URL="http://your-docling-host:8000"
$env:DOCLING_SERVE_API_KEY="your-docling-api-key-if-needed"
$env:DOCLING_SERVE_CONVERT_PATH="/v1/convert/source"
$env:DOCLING_TIMEOUT_SECONDS="120"
$env:DOCLING_USE_LOCAL="0"
```

When `DOCLING_SERVE_URL` is present, both `python main.py ocr ...` and `python main.py extract ... --markdown` send the document to Docling Serve first. If the URL is absent and `DOCLING_USE_LOCAL=1`, the pipeline uses the local `docling` Python package. If neither is available, it falls back to the existing NuExtract OCR path.

In `ocr` mode, Markdown is saved as `*_ocr.md`. When the OCR provider returns extra metadata, it is also saved as `*_ocr_key_values.json` in `data/outputs`.

To run Docling Serve on any port, expose that port and point `DOCLING_SERVE_URL` at it. For example:

```bash
docker run -p 8000:5001 quay.io/docling-project/docling-serve-cpu
```

Then set `DOCLING_SERVE_URL=http://your-hostname:8000`. If Akamai fronts the service, use the Akamai origin URL or hostname and keep the same path `/v1/convert/source` unless you rewrite it upstream.

### `.env` file

An example env file is available at `.env.example`. A local `.env` is also supported for direct runs without re-exporting variables in PowerShell.

### 4. Generate a template

```bash
# From a built-in ACORD description
python main.py generate-template --form "ACORD 130" --out schemas/my_acord130.json

# From a custom description
python main.py generate-template \
  --description "Extract policy number, insured name, effective date, and premium amounts from a commercial auto policy declaration page."
```

---

## Python API

```python
from src.model_manager import load_model
from src.pipeline import AcordPipeline
from src.template_generator import TemplateGenerator
from src.extractor import NuExtractor
from schemas import ACORD_SCHEMAS

# Load once — reuse across many documents
model, processor = load_model()
pipeline = AcordPipeline(model, processor)

# --- Structured extraction with built-in schema ---
result = pipeline.run(
    file_path="data/forms/acord25_sample.pdf",
    form_name="ACORD 25",
    enable_thinking=False,   # True for difficult scanned documents
)
print(result)

# --- OCR / Markdown conversion ---
markdown = pipeline.run_ocr("data/forms/acord25_sample.pdf")

# --- Template generation ---
extractor = NuExtractor(model, processor)
gen = TemplateGenerator(extractor)
template = gen.generate(
    "I want to extract the insured name, policy number, effective date, "
    "and all coverage limits from an ACORD 25 certificate."
)

# --- Extraction with a generated template ---
result = pipeline.run(
    file_path="data/forms/acord25_sample.pdf",
    template=template,
)
```

---

## Template Generation Schema

The `template-generation` mode uses this call internally (mirrors the vLLM `extra_body`):

```python
extra_body={
    "chat_template_kwargs": {
        "mode": "template-generation"
    }
}
```

For local Transformers inference this translates to:

```python
processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    mode="template-generation",   # ← same kwarg, passed directly
)
```

---

## Configuration

Edit `config.py` to change model location, PDF DPI, token limits, etc.

| Setting | Default | Description |
|---|---|---|
| `MODEL_DIR` | `./models/NuExtract3` | Local model weights directory |
| `HF_MODEL_ID` | `numind/NuExtract3` | HF repo used during download |
| `INFERENCE_BACKEND` | `local` | `local` or `api` |
| `API_BASE_URL` | `""` | OpenAI-compatible API base URL |
| `API_KEY` | `""` | API key/token (if your endpoint requires auth) |
| `API_MODEL` | `numind/NuExtract3` | Model name sent in API requests |
| `PDF_DPI` | `170` | Page rendering resolution |
| `MAX_NEW_TOKENS` | `8192` | Max generated tokens |
| `DEVICE_MAP` | `"auto"` | Accelerate device placement |
| `TORCH_DTYPE` | `bfloat16` | Model weight dtype |

---

## NuExtract3 Field Types Reference

| Type | Format | Example |
|---|---|---|
| `"verbatim-string"` | Exact text from document | `"ACORD 25"` |
| `"string"` | Normalized string | `"United States"` |
| `"integer"` | Whole number | `30` |
| `"number"` | Int or decimal | `1000000.00` |
| `"date"` | ISO 8601 date | `"2025-01-01"` |
| `"date-time"` | ISO 8601 date-time | `"2025-01-01T00:00:00"` |
| `"boolean"` | true / false | `true` |
| `"currency"` | ISO 4217 | `"USD"` |
| `"country"` | ISO 3166-1 alpha-2 | `"US"` |
| `["a","b","c"]` | Single-select enum | `"Corporation"` |
| `[["a","b","c"]]` | Multi-select enum | `["Fire","Theft"]` |

---

## License

Apache 2.0 — NuExtract3 model weights are released under Apache 2.0 by NuMind.
