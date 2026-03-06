from agentic_rag.exceptions import RAGSystemError, ConfigurationError, VectorStorageError

def test_rag_error_inheritance():
    """Verify that custom exceptions correctly inherit from RAGSystemError."""
    assert issubclass(ConfigurationError, RAGSystemError)
    assert issubclass(VectorStorageError, RAGSystemError)

def test_exception_message():
    """Verify that exception messages are correctly stored."""
    msg = "Test error message"
    error = ConfigurationError(msg)
    assert str(error) == msg
