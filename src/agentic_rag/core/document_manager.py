from pathlib import Path
import shutil
from typing import List, Tuple, Optional, Callable
from loguru import logger

from agentic_rag.config import settings
from agentic_rag.utils import pdfs_to_markdowns

class DocumentManager:
    """
    Manages the document ingestion pipeline, including conversion, chunking, and indexing.
    """

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.markdown_dir = Path(settings.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
    def add_documents(
        self, 
        document_paths: List[str] | str, 
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[int, int]:
        """
        Processes and adds documents to the RAG system.
        """
        if not document_paths:
            return 0, 0
            
        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        document_paths = [p for p in document_paths if p and Path(p).suffix.lower() in [".pdf", ".md"]]
        
        if not document_paths:
            logger.warning("No valid PDF or Markdown files provided for ingestion.")
            return 0, 0
            
        added = 0
        skipped = 0
        total = len(document_paths)
            
        for i, doc_path in enumerate(document_paths):
            if progress_callback:
                progress_callback((i + 1) / total, f"Processing {Path(doc_path).name}")
                
            doc_name = Path(doc_path).stem
            md_path = self.markdown_dir / f"{doc_name}.md"
            
            if md_path.exists():
                logger.info(f"Skipping {doc_name} as it already exists in the knowledge base.")
                skipped += 1
                continue
                
            try:            
                if Path(doc_path).suffix.lower() == ".md":
                    shutil.copy(doc_path, md_path)
                else:
                    pdfs_to_markdowns(str(doc_path), overwrite=False)            
                
                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)
                
                if not child_chunks:
                    logger.warning(f"No chunks generated for {doc_path}.")
                    skipped += 1
                    continue
                
                collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
                collection.add_documents(child_chunks)
                self.rag_system.parent_store.save_many(parent_chunks)
                
                logger.success(f"Successfully indexed: {doc_path}")
                added += 1
                
            except Exception as e:
                logger.error(f"Error processing {doc_path}: {e}")
                skipped += 1
            
        return added, skipped
    
    def get_markdown_files(self):
        if not self.markdown_dir.exists():
            return []
        return sorted([p.name.replace(".md", ".pdf") for p in self.markdown_dir.glob("*.md")])
    
    def clear_all(self):
        if self.markdown_dir.exists():
            shutil.rmtree(self.markdown_dir)
            self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
        self.rag_system.parent_store.clear_store()
        self.rag_system.vector_db.delete_collection(self.rag_system.collection_name)
        self.rag_system.vector_db.create_collection(self.rag_system.collection_name)