import uuid
import hashlib
from typing import List, Tuple
from loguru import logger
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from agentic_rag.config import settings

class DocumentChuncker:
    """
    DocumentChuncker implements a Multi-Vector Retrieval strategy (Parent-Child).
    
    Optimization Philosophy:
    1. Parent Chunking: Preserves structural context by splitting based on Markdown headers. 
       These represent coherent thematic blocks.
    2. Child Chunking: Highly granular sub-segments optimized for semantic similarity search.
    3. Retrieval Efficiency: By searching against child chunks but surface-answering with 
       parent context, we balance precise retrieval with broad conversational context.
    """
    
    def __init__(self):
        # Header structure preservation
        self.__headers_to_split_on = settings.HEADERS_TO_SPLIT_ON
        self.__header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.__headers_to_split_on
        )
        
        # Granular semantic splitting
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHILD_CHUNK_SIZE,
            chunk_overlap=settings.CHILD_CHUNK_OVERLAP
        )
        
        # Threshold for meaningful blocks
        self.__min_parent_size = settings.MIN_PARENT_SIZE
        
        logger.debug("DocumentChuncker stack initialized.")

    def create_chunks_single(self, file_path: str) -> Tuple[List, List]:
        """
        Pipeline: Ingest -> Header Split -> Granular Child Split -> ID Mapping.
        
        Args:
            file_path: Absolute path to the source Markdown file.
            
        Returns:
            Tuple containing pairs of (parent_id, document) and a list of child documents.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            source_name = str(file_path)
            
            # Step 1: Structural Analysis (Parent Generation)
            p_chunks = self.__header_splitter.split_text(text)
            
            parent_results = []
            child_results = []

            for i, p_chunk in enumerate(p_chunks):
                # Guardrail: Prevent fragmentation by ignoring/merging tiny, context-less blocks
                # In a more advanced implementation, logic for merging neighboring blocks would be added here.
                
                # Deterministic ID generation based on content hash for deduplication/tracking
                parent_id = hashlib.md5(f"{source_name}_{i}_{p_chunk.page_content[:100]}".encode()).hexdigest()
                
                p_chunk.metadata["parent_id"] = parent_id
                p_chunk.metadata["source"] = source_name
                parent_results.append((parent_id, p_chunk))

                # Step 2: Semantic Granularity (Child Generation)
                c_chunks = self.__child_splitter.split_documents([p_chunk])
                for c_chunk in c_chunks:
                    c_chunk.metadata["parent_id"] = parent_id
                    c_chunk.metadata["source"] = source_name
                    child_results.append(c_chunk)

            logger.info(f"Ingestion successful: Generated {len(parent_results)} parent blocks and {len(child_results)} child vectors for {file_path}")
            return parent_results, child_results
            
        except Exception as e:
            logger.error(f"Abort: Ingestion failure for {file_path}. Details: {e}")
            return [], []

    def create_chunks(self, path_dir: str = settings.MARKDOWN_DIR) -> Tuple[List, List]:
        """
        Batch processing entry point for directory-wide ingestion.
        
        Args:
            path_dir: Directory containing Markdown source files.
        """
        from pathlib import Path
        all_parents = []
        all_children = []
        
        # Iterate over structural source files
        for md_file in Path(path_dir).glob("*.md"):
            p, c = self.create_chunks_single(str(md_file))
            all_parents.extend(p)
            all_children.extend(c)
            
        return all_parents, all_children