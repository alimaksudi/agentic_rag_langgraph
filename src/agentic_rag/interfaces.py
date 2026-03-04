"""
Abstract Interfaces for the Agentic RAG System.

This module defines the required contracts for infrastructure components,
allowing for seamless substitution of implementations (e.g., swapping Qdrant for Pinecone)
without modifying the core orchestration logic.
"""
from abc import ABC, abstractmethod
from typing import List, Any
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

class AbstractVectorDB(ABC):
    """
    Contract for a Vector Database connector.
    Any implementation must provide these methods to be compatible with the RAGSystem.
    """
    
    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        """Initialize or verify the existence of a vector collection."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Purge a collection and its data."""
        pass

    @abstractmethod
    def get_collection(self, collection_name: str) -> VectorStore:
        """Return a LangChain-compatible VectorStore interface."""
        pass

class AbstractParentStore(ABC):
    """
    Contract for the Parent Document Cold Storage.
    """
    
    @abstractmethod
    def save(self, parent_id: str, content: str, metadata: dict) -> None:
        """Persist a single parent document."""
        pass
    
    @abstractmethod
    def save_many(self, parents: List[Any]) -> None:
        """Persist multiple parent documents in batch."""
        pass
        
    @abstractmethod
    def load_content_many(self, parent_ids: List[str]) -> List[dict]:
        """Retrieve formatted content blocks for given parent IDs."""
        pass
    
    @abstractmethod
    def clear_store(self) -> None:
        """Purge all stored parent documents."""
        pass
