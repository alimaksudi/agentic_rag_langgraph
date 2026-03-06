import pytest
import os
from pydantic import ValidationError
from agentic_rag.config import Settings
def test_default_config_loading(monkeypatch):
    """Verify that the default settings load correctly by masking the .env file."""
    # Mask any variables that might exist in the local developer's .env file
    monkeypatch.setenv("ACTIVE_LLM_CONFIG", "ollama")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    
    config = Settings()
    assert config.DENSE_MODEL == "sentence-transformers/all-mpnet-base-v2"
    assert config.CHILD_CHUNK_SIZE == 500
    assert config.ACTIVE_LLM_CONFIG == "ollama"

def test_dynamic_env_override(monkeypatch):
    """Verify that environment variables properly override defaults."""
    monkeypatch.setenv("ACTIVE_LLM_CONFIG", "openai")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
    
    config = Settings()
    
    assert config.ACTIVE_LLM_CONFIG == "openai"
    assert config.LLM_TEMPERATURE == 0.5

def test_invalid_type_conversion(monkeypatch):
    """Verify pydantic enforces type hints (e.g. integer conversions)."""
    monkeypatch.setenv("MAX_ITERATIONS", "not_a_number")
    
    with pytest.raises(ValidationError):
        Settings()
