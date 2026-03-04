import pytest
from pathlib import Path
from agentic_rag.document_chunker import DocumentChuncker

@pytest.fixture
def chunker():
    return DocumentChuncker()

def test_chunking_logic(chunker, tmp_path):
    """Test the Parent-Child splitting logic with a sample markdown file."""
    # Create a dummy markdown file
    test_file = tmp_path / "test.md"
    test_file.write_text("# H1\nContent 1\n## H2\nContent 2\n" * 100) # Ensure enough content for parent size
    
    parents, children = chunker.create_chunks_single(str(test_file))
    
    # Verify hierarchical relationship
    assert len(parents) > 0
    assert len(children) >= len(parents)
    
    # Check if child metadata points to a parent
    if children:
        assert "parent_id" in children[0].metadata
        assert "source" in children[0].metadata
