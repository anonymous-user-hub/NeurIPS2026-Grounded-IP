"""
Grounded-IP configuration: paths, model names, hyperparameters.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_ROOT       = ROOT / "data" / "Radiology"
REXVQA_DIR      = DATA_ROOT / "ReXVQA"
REXGRADIENT_DIR = DATA_ROOT / "ReXGradient-160K"
IMAGE_DIR       = REXGRADIENT_DIR / "deid_png"
RADKNOWLEDGE_DIR = DATA_ROOT / "RadKnowledge"

QUERY_FILE         = RADKNOWLEDGE_DIR / "4_Paper_IP-CRR_Queries_520.txt"
REFINED_QUERY_FILE = RADKNOWLEDGE_DIR / "4_1_refined_queries_image_only.txt"

KB_DIR         = ROOT / "saved" / "knowledge_base"
KB_INDEX_PATH  = KB_DIR / "faiss_index.bin"
KB_CHUNKS_PATH = KB_DIR / "chunks.json"
KB_EMBEDS_PATH = KB_DIR / "embeddings.npy"

RESULTS_DIR = ROOT / "saved" / "results"

# ── Model configuration ───────────────────────────────────────────────────────
# Model assignment design (two vLLM servers on one B200 GPU — see job/run_qwen_local.job):
#
#   Main model (VISION_MODEL / TEXT_MODEL):
#     querier, answerer, predictor, explanation generator
#   Validator model (VALIDATOR_VISION_MODEL / VALIDATOR_TEXT_MODEL):
#     v_img, v_kb, v_exp  — must differ architecturally from main for genuine independence
#
# Override via .env or CLI args. Supported local models (vLLM OpenAI-compatible):
#   Qwen/Qwen3.5-9B                    — strong general VLM
#   Qwen/Qwen3-VL-30B-A3B-Instruct    — larger MoE VLM
#   google/medgemma-4b-it              — medical VLM (gated; needs HF_TOKEN)
#   google/medgemma-1.5-4b-it          — smaller medical VLM
#   OpenGVLab/InternVL3-9B             — alternative VLM
# Do NOT use: microsoft/llava-med-v1.5-mistral-7b — not compatible with vLLM OpenAI API.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL   = os.getenv("VISION_MODEL",  "google/medgemma-4b-it")   # multimodal: querier, answerer
TEXT_MODEL     = os.getenv("TEXT_MODEL",    "google/medgemma-4b-it")    # text-only: predictor, explanation generator
EMBED_MODEL    = os.getenv("EMBED_MODEL",   "text-embedding-3-small")

# Validator models: use a different model family from main for independent cross-checking.
# Default: medgemma-4b-it as validator when running Qwen as main, and vice versa.
VALIDATOR_VISION_MODEL = os.getenv("VALIDATOR_VISION_MODEL", "google/medgemma-4b-it")  # v_img
VALIDATOR_TEXT_MODEL   = os.getenv("VALIDATOR_TEXT_MODEL",   "google/medgemma-4b-it")  # v_kb, v_exp

# ── Local / vLLM endpoint (OpenAI-compatible) ─────────────────────────────────
# Set LOCAL_BASE_URL to point to a local vLLM server, e.g. http://localhost:8000/v1
# Used when --local_model is passed to run_rexvqa.py
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
LOCAL_API_KEY  = os.getenv("LOCAL_API_KEY",  "EMPTY")

# Optional second vLLM server for validator models (e.g. MedGemma on port 8001).
# If set, any call whose model matches VALIDATOR_VISION_MODEL or VALIDATOR_TEXT_MODEL
# is routed here instead of the main LOCAL_BASE_URL.
VALIDATOR_BASE_URL = os.getenv("VALIDATOR_BASE_URL", "")

# ── Pipeline hyperparameters ──────────────────────────────────────────────────
K_MAX          = 25     # maximum queries per image (was 15)
MAX_RETRIES    = 2      # max re-tries per validator failure
TOP_R          = 12     # knowledge retrieval top-k chunks per step (was 8)
STOP_ENTROPY   = 0.3    # entropy threshold for early stopping (bits)
CHUNK_SIZE     = 600    # characters per knowledge chunk (was 400; needs KB rebuild to take effect)
CHUNK_OVERLAP  = 150    # overlap between consecutive chunks (was 80; needs KB rebuild)

# ── Knowledge files (TXT and PDF supported) ───────────────────────────────────
KNOWLEDGE_FILES = [
    RADKNOWLEDGE_DIR / "1_Tutorial_RadiologyAssistance_Foundations.txt",
    RADKNOWLEDGE_DIR / "2_Tutorial_RadiologyAssistance_HeartFailure.txt",
    RADKNOWLEDGE_DIR / "3_Tutorial_RadiologyAssistance_LungDisease.txt",
    RADKNOWLEDGE_DIR / "5_1_book_Lacey.pdf",
    RADKNOWLEDGE_DIR / "5_2_book_chapter_Klein.pdf",
]
