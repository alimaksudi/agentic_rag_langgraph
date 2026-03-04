from typing import List, Optional
from pathlib import Path
import glob
import tiktoken
from loguru import logger
from .config import settings

def pdf_to_markdown(pdf_path: str, output_dir: str) -> None:
    """
    Core Document Transformer.
    Converts binary PDF layouts into semantic Markdown structures for LLM ingestion.
    
    Implementation:
    - Leverages PyMuPDF4LLM for structural analysis (headings, tables).
    - Cleans UTF-8 artifacts to ensure prompt integrity.
    """
    try:
        import pymupdf
        import pymupdf4llm
        doc = pymupdf.open(pdf_path)
        md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=True, ignore_images=True, write_images=False, image_path=None)
        md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
        output_path = Path(output_dir) / Path(pdf_path).stem
        output_path.with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))
        logger.debug(f"Knowledge conversion successful: {pdf_path}")
    except Exception as e:
        logger.error(f"Knowledge conversion aborted for {pdf_path}: {e}")
        raise

def pdfs_to_markdowns(path_pattern: str, overwrite: bool = False) -> None:
    """
    Batch Knowledge Ingestion Utility.
    Orchestrates the conversion of multiple PDF sources into the RAG-ready markdown directory.
    """
    output_dir = Path(settings.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(str(pdf_path), str(output_dir))

def estimate_context_tokens(messages: list) -> int:
    """
    Token Telemetry Utility.
    Uses the GPT-4 encoding (cl100k_base) as a standard heuristic for estimating 
    message tokens during context compression cycles.
    """
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        # Fallback to base encoding if network/model identifier fails
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, 'content') and msg.content)