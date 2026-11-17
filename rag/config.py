"""Central settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "documents"
INDEX_DIR = BASE_DIR / "data" / "vector_store"
load_dotenv(BASE_DIR / ".env")  # reads GROQ_API_KEY from the .env file

# ---- Embeddings (free, runs locally) ----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---- Chunking ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# ---- Retrieval ----
TOP_K = int(os.getenv("TOP_K", 4))
# Chunks with cosine similarity below this are treated as "not relevant".
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", 0.25))

# ---- Free LLM backends ----
# "ollama"       -> local Ollama server (recommended: fast, good quality)
# "transformers" -> Hugging Face model loaded 
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# ---- Free LLM via API (Groq) ----
LLM_BACKEND = "groq"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")  # default selection

# Label shown in the dropdown -> model ID sent to Groq
GROQ_MODELS = {
    "GPT-OSS 20B: fast (recommended)": "openai/gpt-oss-20b",
    "GPT-OSS 120B: best quality, slower": "openai/gpt-oss-120b"
}

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 1500))

# ---- Generation ----
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))

# ---- Query guardrails ----
MIN_QUERY_WORDS = 2
