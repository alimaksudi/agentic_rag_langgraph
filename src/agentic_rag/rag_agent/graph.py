from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from functools import partial
from loguru import logger

from .graph_state import State, AgentState
from .nodes import orchestrator, compress_context, fallback_response, \
    should_compress_context, collect_answer, summarize_history, rewrite_query, \
    request_clarification, aggregate_answers
from .edges import route_after_orchestrator_call, route_after_rewrite

def create_agent_graph(llm, tools_list):
    """
    State-Machine Compiler for the Agentic RAG Pipeline.
    
    Architecture Design:
    1. Agent Subgraph (Recursive Research): 
       - Implements an autonomous reasoning loop ('orchestrator') that uses tools.
       - Integrated safety rails ('fallback_response') and memory management ('compress_context').
    2. Main Graph (Stateful Conversation):
       - Handles the pre-processing (Summary & Rewriting) and post-processing (Aggregation).
       - Uses 'Fan-out' (Map) pattern to parallelize sub-queries across agents.
       - Implements Human-in-the-Loop ('interrupt_before') for ambiguous queries.
    """
    
    # --- Subgraph Definition (Atomic Agentic Runner) ---
    
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    # Note: Using InMemorySaver for ephemeral session memory. 
    # For production persistency, swap with Postgres/Redis Checkpointer.
    checkpointer = InMemorySaver()

    logger.info("Compiling agentic logic loop...")
    agent_builder = StateGraph(AgentState)
    
    # Nodes: Operational Units
    agent_builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node("compress_context", partial(compress_context, llm=llm))
    agent_builder.add_node("fallback_response", partial(fallback_response, llm=llm))
    agent_builder.add_node(should_compress_context) 
    agent_builder.add_node(collect_answer)
    
    # Control Flow: Recursive Reasoning
    agent_builder.add_edge(START, "orchestrator")    
    agent_builder.add_conditional_edges(
        "orchestrator", 
        route_after_orchestrator_call, 
        {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"}
    )
    agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)
    
    agent_subgraph = agent_builder.compile()
    
    # --- Main Orchestration Graph (Conversation Manager) ---
    
    logger.info("Compiling main conversation coordinator...")
    graph_builder = StateGraph(State)
    
    # Nodes: Lifecycle Stages
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))
    graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    graph_builder.add_node(request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", partial(aggregate_answers, llm=llm))
    
    # Control Flow: Linear Transformation -> Parallel Fan-out -> Aggregation
    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)

    # Compilation with Human-in-the-Loop Interrupt Support
    agent_graph = graph_builder.compile(
        checkpointer=checkpointer, 
        interrupt_before=["request_clarification"]
    )

    logger.success("Multi-agent architecture successfully compiled and validated.")
    return agent_graph