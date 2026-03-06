import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Tuple

class Settings(BaseSettings):
    # --- Directory Configuration ---
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    MARKDOWN_DIR: str = os.path.join(BASE_DIR, "markdown_docs")
    PARENT_STORE_PATH: str = os.path.join(BASE_DIR, "parent_store")
    QDRANT_DB_PATH: str = os.path.join(BASE_DIR, "qdrant_db")

    # --- Qdrant Configuration ---
    CHILD_COLLECTION: str = "document_child_chunks"
    SPARSE_VECTOR_NAME: str = "sparse"

    # --- Model Configuration ---
    DENSE_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    SPARSE_MODEL: str = "Qdrant/bm25"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_MODEL: str = "qwen3:4b-instruct-2507-q4_K_M"
    LLM_TEMPERATURE: float = 0.0

    # --- Agent Configuration ---
    MAX_TOOL_CALLS: int = 8
    MAX_ITERATIONS: int = 10
    BASE_TOKEN_THRESHOLD: int = 2000
    TOKEN_GROWTH_FACTOR: float = 0.9

    # --- Text Splitter Configuration ---
    CHILD_CHUNK_SIZE: int = 500
    CHILD_CHUNK_OVERLAP: int = 100
    MIN_PARENT_SIZE: int = 2000
    MAX_PARENT_SIZE: int = 4000
    HEADERS_TO_SPLIT_ON: List[Tuple[str, str]] = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3")
    ]

    # --- LLM Provider Settings ---
    ACTIVE_LLM_CONFIG: str = "ollama"  # Options: ollama, openai, anthropic, google
    
    # API Keys (will be pulled from env if present)
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Initialize settings
settings = Settings()
