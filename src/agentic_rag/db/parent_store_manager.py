import re
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from agentic_rag.config import settings
from agentic_rag.interfaces import AbstractParentStore

class ParentStoreManager(AbstractParentStore):
    """
    Manages the Cold Storage for large-context parent documents.
    
    Architecture Design:
    1. Parent-Child Relationship: While RAG searches against 'children' (semantic vectors), 
       it generates answers using 'parents' (structural context).
    2. File-Based Persistence: Implements a simple, high-performance local filesystem storage 
       for parent chunks, ensuring easy introspection and backup.
    3. Retrieval Ordering: Preserves document flow during reconstruction of multiple parent chunks.
    """

    def __init__(self, store_path: str = settings.PARENT_STORE_PATH):
        """
        Initializes the persistence layer.
        
        Args:
            store_path: Directory path for document JSON storage.
        """
        self.__store_path = Path(store_path) 
        self.__store_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ParentStoreManager storage established at {self.__store_path}")

    def save(self, parent_id: str, content: str, metadata: Dict) -> None:
        """Persists a single parent document to storage."""
        file_path = self.__store_path / f"{parent_id}.json"
        file_path.write_text(
            json.dumps({"page_content": content,"metadata": metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def save_many(self, parents: List) -> None:
        """Batch persistence for multiple parent blocks."""
        logger.debug(f"Persisting {len(parents)} parent documents to cold storage.")
        for parent_id, doc in parents:
            self.save(parent_id, doc.page_content, doc.metadata)

    def load(self, parent_id: str) -> Dict:
        """Retrieves raw JSON payload for a given document index."""
        file_path = self.__store_path / (
            parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        )
        return json.loads(file_path.read_text(encoding="utf-8"))
    
    def load_content(self, parent_id: str) -> Dict:
        """Retrieves formatted content and metadata for LLM ingestion."""
        try:
            data = self.load(parent_id)
            return {
                    "content": data["page_content"],
                    "parent_id": parent_id,
                    "metadata": data["metadata"]
                }
        except Exception as e:
            logger.error(f"Failed to load parent chunk {parent_id}: {e}")
            return {}

    @staticmethod
    def _get_sort_key(id_str: str) -> int:
        """Heuristic for sorting chunks to preserve original document flow."""
        match = re.search(r'_parent_(\d+)$', id_str)
        return int(match.group(1)) if match else 0

    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        """Retrieves and logically ordered content blocks for multi-source synthesis."""
        unique_ids = set(parent_ids)
        return [self.load_content(pid) for pid in sorted(unique_ids, key=self._get_sort_key)]
    
    def clear_store(self) -> None:
        """Atomic purge of all parent storage. Used during knowledge base resets."""
        if self.__store_path.exists():
            logger.warning("Triggering absolute reset of Parent Cold Storage.")
            shutil.rmtree(self.__store_path)
        self.__store_path.mkdir(parents=True, exist_ok=True)