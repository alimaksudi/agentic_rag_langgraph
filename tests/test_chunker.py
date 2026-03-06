import pytest
import os
from tempfile import NamedTemporaryFile
from agentic_rag.document_chunker import DocumentChuncker

@pytest.fixture
def chunker():
    return DocumentChuncker()

def test_markdown_header_splitting(chunker):
    """Verify that the chunker correctly splits documents at H1/H2 boundaries."""
    md_content = """# Main Title
This is the introduction.

## Section 1
Here is the detail for section 1.

### Subsection A
Specifics about A."""

    with NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(md_content)
        temp_path = f.name
        
    try:
        parents, children = chunker.create_chunks_single(temp_path)
        
        # We expect 3 parent chunks based on the 3 headers
        assert len(parents) == 3
        
        # Verify metadata extraction
        assert parents[0][1].metadata.get("H1") == "Main Title"
        assert parents[1][1].metadata.get("H2") == "Section 1"
        assert parents[2][1].metadata.get("H3") == "Subsection A"
        
        # Ensure children maintain the parent ID linkage
        assert len(children) >= len(parents)
        assert children[0].metadata.get("parent_id") == parents[0][0]
    finally:
        os.unlink(temp_path)

def test_empty_file_handling(chunker):
    """Verify the chunker handles empty files gracefully."""
    with NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        temp_path = f.name
        
    try:
        parents, children = chunker.create_chunks_single(temp_path)
        assert len(parents) == 0
        assert len(children) == 0
    finally:
        os.unlink(temp_path)
