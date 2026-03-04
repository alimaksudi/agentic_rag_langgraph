from typing import List, Any
from langchain_core.messages import HumanMessage
from loguru import logger

class ChatInterface:
    """
    Provides a simple interface for interacting with the RAG agent graph.
    """
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        
    def chat(self, message: str, history: List[Any]) -> str:
        """
        Sends a message to the RAG agent and returns the final response.
        """
        if not self.rag_system.agent_graph:
            logger.warning("Attempted to chat, but system is not initialized.")
            return "System not initialized!"
            
        try:
            logger.debug(f"Chat message received: {message}")
            result = self.rag_system.agent_graph.invoke(
                {"messages": [HumanMessage(content=message.strip())]},
                self.rag_system.get_config()
            )
            response = result["messages"][-1].content
            logger.info("Chat response generated successfully.")
            return response
            
        except Exception as e:
            from agentic_rag.exceptions import AgentExecutionError
            logger.error(f"Chat interaction failed: {e}")
            raise AgentExecutionError(f"Failed to generate response: {e}") from e
    
    def clear_session(self):
        self.rag_system.reset_thread()