"""Settings, read from environment variables so nothing secret lives in the code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
TEST_SET = ROOT / "eval" / "test_set.json"
RESULTS_DIR = ROOT / "eval" / "results"
CHROMA_DIR = ROOT / ".chroma"

# Embeddings: "minilm" (sentence-transformers, best quality) or "tfidf" (no download, for offline tests)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "minilm")
MINILM_MODEL = os.getenv("MINILM_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Answer model: "gemini" (needs GEMINI_API_KEY) or "mock" (no API, for offline tests)
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

TOP_K = int(os.getenv("TOP_K", "4"))

# Hand off to a human advisor when the best retrieved section scores below this
# (cosine similarity). Tune it with the evaluation script.
DEFAULT_THRESHOLDS = {"minilm": 0.35, "tfidf": 0.12}
ESCALATE_BELOW = float(os.getenv("ESCALATE_BELOW", DEFAULT_THRESHOLDS.get(EMBEDDING_BACKEND, 0.3)))
