from typing import Literal, List, Union
from langgraph.types import Send
from .graph_state import State, AgentState
from agentic_rag.config import settings

def route_after_orchestrator_call(state: AgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    """
    Execution Guardrail Router.
    
    Logic:
    1. Tool Check: If tool calls are present, branch to 'tools'.
    2. Threshold Enforcement: 
       - If tool_call_count exceeds budget -> Fallback.
       - If loop iteration limit reached -> Fallback.
    3. Terminal State: If no tools and under budget -> Collect Answer.
    """
    last_message = state["messages"][-1]
    
    # Check for requested tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Safety valve: detect infinite tool-calling loops
        if state.get("tool_call_count", 0) >= settings.MAX_TOOL_CALLS:
            return "fallback_response"
        return "tools"
    
    # Safety valve: prevent endless reasoning cycles
    if state.get("iteration_count", 0) >= settings.MAX_ITERATIONS:
        return "fallback_response"
        
    return "collect_answer"

def decide_to_generate(state: AgentState) -> Literal["collect_answer", "fail_response"]:
    """
    CRAG Decision Logic.
    
    If no relevant documents were found during the grading phase, 
    route to 'fail_response' to say "I don't know".
    """
    if not state.get("is_relevant", True):
        return "fail_response"
    return "collect_answer"

def route_after_intent(state: State) -> Literal["fast_reply", "summarize_history"]:
    """
    Zero-Shot Bypass Router.
    Routes conversational filler directly to the user, bypassing research.
    """
    if state.get("intent_type") == "chat":
        return "fast_reply"
    return "summarize_history"


def route_after_rewrite(state: State) -> Union[Literal["request_clarification"], List[Send]]:
    """
    Strategic Workflow Router.
    
    Logic:
    1. Clarity Check: If the question analysis deemed the input fuzzy -> Interruption (Request Clarification).
    2. Atomic Parallelization: If clear, fan-out the rewritten sub-queries 
       to 'agent' sub-graphs using LangGraph's Send API.
    """
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        # Fan-out: send each rewritten question to parallel processing
        # This implements the 'Map' stage of the Map-Reduce pattern
        return [
            Send("agent", {
                "question": query, 
                "question_index": idx, 
                "messages": [], 
                "tool_call_count": 0, 
                "iteration_count": 0
            })
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]