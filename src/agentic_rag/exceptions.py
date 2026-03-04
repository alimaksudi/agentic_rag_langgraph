"""
Custom Exception Hierarchy for the Agentic RAG System.

This module defines domain-specific exceptions to enable precise error handling,
programmatic recovery, and clear debugging telemetry across the application layers.
"""

class RAGSystemError(Exception):
    """Base exception for all RAG system related errors."""
    pass

class ConfigurationError(RAGSystemError):
    """Raised when there is an issue with the system configuration or environment variables."""
    pass

class VectorStorageError(RAGSystemError):
    """Raised when the vector database encounters an initialization or operational failure."""
    pass

class ParentStoreError(RAGSystemError):
    """Raised when the cold storage (parent documents) cannot be accessed or written to."""
    pass

class DocumentProcessingError(RAGSystemError):
    """Raised when a document cannot be parsed, chunked, or ingested."""
    pass

class LLMProviderError(RAGSystemError):
    """Raised when the LLM provider fails to initialize or respond correctly."""
    pass

class AgentExecutionError(RAGSystemError):
    """Raised when the LangGraph agent encounters an unrecoverable state or validation error."""
    pass
