from typing import List, Any
from langchain_core.messages import HumanMessage
from loguru import logger

class ChatInterface:
    """
    Provides a simple interface for interacting with the RAG agent graph.
    """
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        
    async def chat_stream(self, message: str, history: List[Any]):
        """
        Sends an asynchronous, streaming message to the RAG Agent.
        Yields progressive 'thought' steps followed by the final answer.
        """
        if not self.rag_system.agent_graph:
            logger.warning("Attempted to chat, but system is not initialized.")
            yield "System not initialized!"
            return
            
        try:
            logger.debug(f"Chat stream initiated: {message}")
            
            # The running string we will continuously yield to Gradio
            streamed_response = ""
            
            # Use LangGraph's async stream events (version V2 required)
            events = self.rag_system.agent_graph.astream_events(
                {"messages": [HumanMessage(content=message.strip())]},
                config=self.rag_system.get_config(),
                version="v2"
            )
            
            async for event in events:
                kind = event["event"]
                tags = event.get("tags", [])
                
                # Intercept Node Entry (Agent Thoughts)
                if kind == "on_chain_start" and event.get("name") in ["classify_intent", "rewrite_query", "orchestrator", "compress_context", "aggregate_answers", "fast_reply"]:
                    node_name = event["name"]
                    thought_maps = {
                        "classify_intent": "🧠 Analyzing intent...",
                        "rewrite_query": "✍️ Formulating optimal search queries...",
                        "orchestrator": "🔍 Searching semantic knowledge base...",
                        "compress_context": "🗜️ Compressing retrieved research...",
                        "aggregate_answers": "✨ Synthesizing final response...",
                        "fast_reply": "💬 Generating conversational reply..."
                    }
                    if node_name in thought_maps:
                        streamed_response += f"*{thought_maps[node_name]}*\n\n"
                        yield streamed_response
                
                # Intercept the Final LLM Output Tokens
                elif kind == "on_chat_model_stream":
                    if "aggregate_answers" in tags:
                        chunk = event["data"]["chunk"].content
                        if chunk:
                            streamed_response += chunk
                            yield streamed_response
                            
                # Intercept static outputs that don't call an LLM (Fast Path)
                elif kind == "on_chain_end" and event.get("name") == "fast_reply":
                    output = event.get("data", {}).get("output", {})
                    if "messages" in output and output["messages"]:
                        chunk = output["messages"][-1].content
                        streamed_response += chunk
                        yield streamed_response
            
            logger.info("Chat stream completed successfully.")
            
        except Exception as e:
            from agentic_rag.exceptions import AgentExecutionError
            logger.error(f"Chat interaction failed: {e}")
            yield f"**Service Error**: The agent encountered an issue processing your request."
    
    def clear_session(self):
        self.rag_system.reset_thread()