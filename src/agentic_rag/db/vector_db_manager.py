from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from loguru import logger
import os

from agentic_rag.config import settings
from agentic_rag.interfaces import AbstractVectorDB

class VectorDbManager(AbstractVectorDB):
    """
    Manages the Vector Intelligence Layer of the RAG system.
    
    Design Patterns:
    1. Hybrid Search: Integrates both Dense (Semantic) and Sparse (Keyword/BM25) 
       embeddings for robust retrieval across long-tail and technical queries.
    2. Local Persistency: Leverages Qdrant's embedded mode for zero-infrastructure 
       setup while maintaining enterprise-grade search features.
    """

    def __init__(self, db_path: str = settings.QDRANT_DB_PATH):
        """
        Initializes search engines and connection to the local vector storage.
        """
        logger.info(f"Initializing VectorDbManager at {db_path}")
        
        # Search Engine Layer
        self.dense_embeddings = HuggingFaceEmbeddings(model_name=settings.DENSE_MODEL)
        self.sparse_embeddings = FastEmbedSparse(model_name=settings.SPARSE_MODEL)
        
        # Storage Layer
        self.client = QdrantClient(path=db_path)
        self.__sparse_name = settings.SPARSE_VECTOR_NAME
        
        logger.debug("Embeddings engines loaded successfully.")

    def create_collection(self, collection_name: str) -> None:
        """
        Idempotently creates a collection with hybrid search vectors.
        Configures COSINE distance for semantic vectors and defaults for sparse BM25.
        """
        if not self.client.collection_exists(collection_name):
            logger.info(f"Establishing new collection: {collection_name}")
            
            # Determine vector dimensionality from model output
            sample_dim = len(self.dense_embeddings.embed_query("warmup"))
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=sample_dim,
                    distance=qmodels.Distance.COSINE
                ),
                sparse_vectors_config={
                    self.__sparse_name: qmodels.SparseVectorParams()
                },
            )
            logger.success(f"Collection '{collection_name}' ready.")
        else:
            logger.debug(f"Vector collection '{collection_name}' already exists.")

    def delete_collection(self, collection_name: str) -> None:
        """Purges a collection and its associated indices."""
        if self.client.collection_exists(collection_name):
            logger.warning(f"Purging vector collection: {collection_name}")
            self.client.delete_collection(collection_name)

    def get_collection(self, collection_name: str) -> QdrantVectorStore:
        """
        Returns a high-level LangChain-compatible interface for the collection.
        Configures the Hybrid retrieval mode (Dense + Sparse).
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name=self.__sparse_name
        )