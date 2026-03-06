from typing import List, Any
from langchain_core.tools import tool
from loguru import logger

from agentic_rag.db.parent_store_manager import ParentStoreManager

class ToolFactory:
    
    def __init__(self, collection: Any, reranker: Any = None):
        self.collection = collection
        self.reranker = reranker
        self.parent_store_manager = ParentStoreManager()
        logger.debug("ToolFactory initialized with re-ranker.")
    
    def _search_child_chunks(self, query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        try:
            # Stage 1: Broad Hybrid Search Extraction
            # We pull an expanded candidate pool (e.g. 20) to ensure high recall
            expanded_limit = limit * 4
            results = self.collection.similarity_search(query, k=expanded_limit, score_threshold=0.6)
            if not results:
                return "NO_RELEVANT_CHUNKS"

            # Stage 2: Cross-Encoder Precision Re-Ranking
            if self.reranker:
                # Pair the original query with each chunk's content for the cross-encoder
                sentence_pairs = [[query, doc.page_content] for doc in results]
                scores = self.reranker.predict(sentence_pairs)
                
                # Zip the scores with the documents and sort descending
                scored_docs = list(zip(scores, results))
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                
                # Extract the top K documents after re-ranking
                results = [doc for score, doc in scored_docs[:limit]]
            else:
                results = results[:limit]

            return "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])            

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_ids: List of parent chunk IDs to retrieve
        """
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"

            return "\n\n".join([
                f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {doc.get('content', '').strip()}"
                for doc in raw_parents
            ])            

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )          

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def create_tools(self) -> List:
        """Create and return the list of tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        return [search_tool, retrieve_tool]