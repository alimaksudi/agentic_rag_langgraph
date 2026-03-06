import pytest
from typing import List, Any
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from agentic_rag.interfaces import AbstractVectorDB, AbstractParentStore

class MockVectorStore(VectorStore):
    """A mock VectorStore that returns dummy documents."""
    
    def add_texts(self, texts: List[str], metadatas: List[dict] = None, **kwargs: Any) -> List[str]:
        return ["mock_id_1", "mock_id_2"]
        
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        return [
            Document(page_content="Mock retrieved content", metadata={"source": "mock_file.pdf"})
        ]
        
    @classmethod
    def from_texts(cls, texts: List[str], embedding: Any, metadatas: List[dict] = None, **kwargs: Any) -> VectorStore:
        pass

class MockVectorDB(AbstractVectorDB):
    def create_collection(self, collection_name: str) -> None:
        pass

    def delete_collection(self, collection_name: str) -> None:
        pass

    def get_collection(self, collection_name: str) -> VectorStore:
        return MockVectorStore()

    def get_reranker(self) -> Any:
        return None

class MockParentStore(AbstractParentStore):
    def __init__(self):
        self.store = {}
        
    def save(self, parent_id: str, content: str, metadata: dict) -> None:
        self.store[parent_id] = {"parent_id": parent_id, "content": content, "metadata": metadata}
        
    def save_many(self, parents: List[Any]) -> None:
        for p in parents:
            self.save(p.metadata["parent_id"], p.page_content, p.metadata)
            
    def load_content_many(self, parent_ids: List[str]) -> List[dict]:
        return [self.store[pid] for pid in parent_ids if pid in self.store]
        
    def clear_store(self) -> None:
        self.store = {}

@pytest.fixture
def mock_vector_db():
    return MockVectorDB()

@pytest.fixture
def mock_parent_store():
    return MockParentStore()
