"""Central configuration for the Employee Onboarding AI Agent.

Every path, model name, and tunable number the application needs lives here, so
there is exactly one place to change them. Values are read from environment
variables where it makes sense (with safe defaults) and secrets are NEVER stored
in this file - the API key is only ever read from the environment via
``model_client``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths (all relative to the project root, i.e. the parent of this app/ folder)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPANY_DOCS_DIR = PROJECT_ROOT / "company_docs"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
EMPLOYEES_PATH = PROJECT_ROOT / "data" / "employees.json"
TICKETS_DB_PATH = PROJECT_ROOT / "data" / "tickets.db"

# --------------------------------------------------------------------------- #
# Models (kept identical to the existing ingestion so vectors stay compatible)
# --------------------------------------------------------------------------- #
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-flash-latest")
# The embedding model must match the one used at ingestion time (vector_store.py).
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
# LangChain's Google wrapper expects the "models/" prefix.
LC_EMBED_MODEL = f"models/{EMBED_MODEL}"

# --------------------------------------------------------------------------- #
# Chroma collection (reuse the collection the root ingestion already fills)
# --------------------------------------------------------------------------- #
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company_policies")

# --------------------------------------------------------------------------- #
# Chunking (centralized so it is easy to tune in one place)
# --------------------------------------------------------------------------- #
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# --------------------------------------------------------------------------- #
# Retrieval / agent
# --------------------------------------------------------------------------- #
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))

# --------------------------------------------------------------------------- #
# Streamlit -> FastAPI
# --------------------------------------------------------------------------- #
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# The exact user-facing sentence used whenever evidence is missing. Kept as one
# constant so the RAG layer, the agent, and the tests all agree on the wording.
NOT_FOUND_MESSAGE = (
    "I could not find this information in the available company documents."
)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure simple, readable logging once for the whole application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
