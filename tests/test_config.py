import pytest
from agentic_rag.config import settings

def test_config_initialization():
    """Verify that settings are correctly loaded with defaults."""
    assert settings.ACTIVE_LLM_CONFIG in ["ollama", "openai", "anthropic", "google"]
    assert settings.CHILD_CHUNK_SIZE > 0
    assert settings.MAX_TOOL_CALLS > 0

def test_directory_structure():
    """Verify that base directories are correctly resolved."""
    assert "src/agentic_rag" in settings.BASE_DIR
    assert settings.QDRANT_DB_PATH.endswith("qdrant_db")
