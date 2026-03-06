import uuid
from typing import Dict, Any, Optional
from loguru import logger

from agentic_rag.config import settings
from agentic_rag.db.vector_db_manager import VectorDbManager
from agentic_rag.db.parent_store_manager import ParentStoreManager
from agentic_rag.document_chunker import DocumentChuncker
from agentic_rag.rag_agent.tools import ToolFactory
from agentic_rag.rag_agent.graph import create_agent_graph
from agentic_rag.exceptions import ConfigurationError, RAGSystemError

from agentic_rag.interfaces import AbstractVectorDB, AbstractParentStore

class RAGSystem:
    """
    RAGSystem serves as the central orchestration layer for the Agentic RAG pipeline.
    
    Architecture Design:
    1. Multi-Provider Strategy: Abstracts LLM initialization to support heterogeneous 
       compute environments (Local via Ollama, Cloud via OpenAI/Anthropic/Google).
    2. State Management: Employs persistent thread-based memory to maintain conversation 
       continuity and context safety.
    3. Operational Guardrails: Implements recursion limits and token-aware thresholds 
       to ensure system stability and cost-efficiency.
    4. Dependency Injection: Core components are injected to decouple infrastructure 
       and facilitate testing.
    """
    
    def __init__(
        self,
        vector_db: AbstractVectorDB,
        parent_store: AbstractParentStore,
        chunker: DocumentChuncker,
        collection_name: str = settings.CHILD_COLLECTION
    ):
        """
        Initializes the core infrastructure components via Dependency Injection.
        
        Args:
            vector_db: The vector storage engine (e.g., Qdrant).
            parent_store: The cold storage engine for parent chunks.
            chunker: The document parsing and splitting subsystem.
            collection_name: Unique identifier for the vector database collection.
        """
        self.collection_name = collection_name
        
        # Injected Infrastructure
        self.vector_db = vector_db
        self.parent_store = parent_store
        self.chunker = chunker
        
        # Execution Layer
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        
        # Safety/Guardrail Layer
        self.recursion_limit = 50
        
        logger.info(f"RAGSystem initialized with collection: {self.collection_name}")
        
    def _initialize_llm(self):
        """
        Dynamic LLM Factory. 
        Selects and configures the provider based on the environment-driven 'settings' object.
        Centralizes model-specific parameters like temperature and API keys.
        """
        provider = settings.ACTIVE_LLM_CONFIG.lower()
        model = settings.LLM_MODEL
        temp = settings.LLM_TEMPERATURE
        
        logger.info(f"Initializing LLM provider: {provider} (model: {model})")
        
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model, 
                temperature=temp, 
                base_url=settings.OLLAMA_BASE_URL
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=temp, api_key=settings.OPENAI_API_KEY)  # type: ignore
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model_name=model, temperature=temp, api_key=settings.ANTHROPIC_API_KEY)
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=temp, google_api_key=settings.GOOGLE_API_KEY)
        else:
            raise ConfigurationError(f"Unsupported LLM provider: {provider}")

    def initialize(self):
        """
        Bootstraps the RAG environment.
        Performs vector DB collection setup and compiles the LangGraph agent chain.
        Ensures high-level tools are bound to the execution engine.
        """
        try:
            # Idempotent collection creation
            self.vector_db.create_collection(self.collection_name)
            collection = self.vector_db.get_collection(self.collection_name)
            
            # Agent Compilation
            llm = self._initialize_llm()
            reranker = self.vector_db.get_reranker()
            tools = ToolFactory(collection, reranker).create_tools()
            self.agent_graph = create_agent_graph(llm, tools)
            
            logger.info("RAGSystem successfully initialized and graph compiled.")
        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize RAGSystem stack: {e}")
            raise RAGSystemError(f"System initialization failure: {e}") from e
        
    def get_config(self) -> Dict[str, Any]:
        """
        Generates the execution configuration for the LangGraph agent.
        Injects the thread safety identifier and recursion guardrails.
        """
        return {
            "configurable": {"thread_id": self.thread_id}, 
            "recursion_limit": self.recursion_limit
        }
    
    def reset_thread(self):
        """
        State Purging.
        Clears the checkpointer memory and rotates the thread ID to isolate 
        new sessions while preventing memory leaks.
        """
        try:
            if self.agent_graph and hasattr(self.agent_graph, "checkpointer"):
                # Clean persistent state
                self.agent_graph.checkpointer.delete_thread(self.thread_id)
            logger.info(f"Thread state {self.thread_id} reset.")
        except Exception as e:
            logger.warning(f"Handled error during thread state deletion: {e}")
            
        # Rotate thread ID to ensure fresh context for the user
        self.thread_id = str(uuid.uuid4())