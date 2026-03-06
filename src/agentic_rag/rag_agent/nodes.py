from typing import Literal, Set, List, Any
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.types import Command
from agentic_rag.rag_agent.graph_state import State, AgentState
from agentic_rag.rag_agent.schemas import QueryAnalysis, IntentClassification, DocumentGrader
from agentic_rag.rag_agent.prompts import (
    get_intent_classifier_prompt,
    get_conversation_summary_prompt,
    get_rewrite_query_prompt,
    get_orchestrator_prompt,
    get_fallback_response_prompt,
    get_context_compression_prompt,
    get_aggregation_prompt,
    get_grader_prompt,
)
from agentic_rag.utils import estimate_context_tokens
from agentic_rag.config import settings

# --- Pre-processing Nodes ---

def classify_intent(state: State, llm):
    """
    Fast-Path zero-shot classifier.
    Bypasses deep research loops for simple pleasantries to save tokens and time.
    """
    last_message = state["messages"][-1].content
    
    llm_with_structure = llm.with_config(temperature=0.0).with_structured_output(IntentClassification)
    response = llm_with_structure.invoke([
        SystemMessage(content=get_intent_classifier_prompt()), 
        HumanMessage(content=last_message)
    ])
    
    return {
        "intent_type": response.intent_type,
        "fast_reply_msg": response.fast_reply
    }

def fast_reply(state: State):
    """
    Immediate return node for non-research queries.
    """
    reply = state.get("fast_reply_msg") or "Hello! I am ready to help you research your documents."
    return {"messages": [AIMessage(content=reply)]}

def summarize_history(state: State, llm):
    """
    Context Distillation: Summarizes the prior conversation turns.
    
    Philosophy: Reduces token noise while preserving intent. This ensures the 
    subsequent Query Rewriting stage has a compact but complete window into the user's needs.
    """
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}
    
    # Filter for meaningful exchange only
    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}
    
    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([
        SystemMessage(content=get_conversation_summary_prompt()), 
        HumanMessage(content=conversation)
    ])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

def rewrite_query(state: State, llm):
    """
    Intelligent Query Disambiguation & Rewriting.
    
    Strategy:
    1. Multi-Vector Expansion: Breaks complex/multi-part questions into individual atomic queries.
    2. De-referencing: Resolves pronouns or vague references using the conversation summary.
    3. Clarity Guard: Detects if the query is fundamentally unanswerable and triggers clarification.
    """
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"User Query:\n{last_message.content}\n"

    # Use structured output to force the LLM into a specific decision-making schema
    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([
        SystemMessage(content=get_rewrite_query_prompt()), 
        HumanMessage(content=context_section)
    ])

    if response.questions and response.is_clear:
        # State management: Clear intermediate messages to keep the graph graph clean
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {
            "questionIsClear": True, 
            "messages": delete_all, 
            "originalQuery": last_message.content, 
            "rewrittenQuestions": response.questions
        }

    # Fallback to human-in-the-loop if the intent is fuzzy
    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "I need more information to understand your question."
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}

def request_clarification(state: State):
    """Placeholder node for Human-in-the-Loop interruption."""
    return {}

# --- Agentic Retrieval Nodes (Parallelized via Send) ---

def orchestrator(state: AgentState, llm_with_tools):
    """
    The Thinking Brain of the Retrieval Loop.
    
    Control Logic:
    1. Memory Injection: Injects compressed summaries from prior "compression" cycles to prevent loops.
    2. Zero-Shot Forcing: On first run, explicitly mandates a search to prevent hollow hallucinations.
    3. Iterative Research: Decides whether to keep searching, retrieve full parent context, or end.
    """
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        # Guardrail: Force initial research to ground the response
        force_search = HumanMessage(content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION.")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {
            "messages": [human_msg, response], 
            "tool_call_count": len(response.tool_calls or []), 
            "iteration_count": 1
        }

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    
    return {
        "messages": [response], 
        "tool_call_count": len(tool_calls) if tool_calls else 0, 
        "iteration_count": 1
    }

def fallback_response(state: AgentState, llm):
    """
    Resilient Synthesis: Triggered when budget/limits are reached.
    
    Role: Instead of failing, it performs a 'best-effort' synthesis using whatever 
    shards of information were retrieved before the timeout/limit hit.
    """
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}

def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    """
    Dynamic Token Budget Manager.
    
    Design: Checks cumulative message tokens vs thresholds.
    - If below threshold: continues to orchestrator for more research.
    - If above threshold: triggers compression to prevent LLM context overflow.
    """
    messages = state["messages"]

    # Deduplication logic to avoid re-researching same items after compression
    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)
                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    # Token-Aware branching
    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = settings.BASE_TOKEN_THRESHOLD + int(current_token_summary * settings.TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)

def compress_context(state: AgentState, llm):
    """
    Semantic Compression Engine.
    
    Philosophy: Summarizes raw tool observations into a compact 'Research Context'.
    This effectively clears the message history while preserving the knowledge gained, 
    allowing for deep, recursive research without hitting context window limits.
    """
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([
        SystemMessage(content=get_context_compression_prompt()), 
        HumanMessage(content=conversation_text)
    ])
    new_summary = summary_response.content

    # Preserve already executed keys in the summary block for agent reference
    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    # Clean the slate: Remove intermediate messages to save tokens
    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

async def grade_documents(state: AgentState, llm):
    """
    Corrective RAG (CRAG) Relevance Assessment.
    
    Logic:
    1. Extracts all ToolMessages (search results).
    2. Uses a fast LLM pass to grade each chunk as 'yes' (relevant) or 'no' (irrelevant).
    3. If any chunks are relevant, synthesis proceeds.
    4. If ZERO chunks are relevant, marks search as a failure.
    """
    messages = state["messages"]
    question = state["question"]
    
    # Filter for tool result messages
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    
    if not tool_messages:
        return {"is_relevant": False}

    llm_with_structure = llm.with_config(temperature=0.0).with_structured_output(DocumentGrader)
    
    # Perform parallel grading requests
    grading_tasks = [
        llm_with_structure.ainvoke([
            SystemMessage(content=get_grader_prompt()),
            HumanMessage(content=f"Question: {question}\n\nDocument Chunk: {msg.content}")
        ])
        for msg in tool_messages
    ]
    
    import asyncio
    results = await asyncio.gather(*grading_tasks)
    
    significant_matches = [res for res in results if res.binary_score == "yes"]
    
    if not significant_matches:
        logger.warning(f"CRAG: No relevant documents found for query: {question}")
        return {"is_relevant": False}
    
    logger.info(f"CRAG: Found {len(significant_matches)} relevant chunks.")
    return {"is_relevant": True}

def fail_response(state: AgentState):
    """
    Graceful handling of irrelevant search results.
    """
    msg = f"I'm sorry, but I couldn't find any information in the available documents to accurately answer your question: '{state['question']}'."
    return {"messages": [AIMessage(content=msg)]}

def collect_answer(state: AgentState):
    """Post-Retrieval Data Unification."""
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    
    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}]
    }

# --- Post-processing Nodes ---

async def aggregate_answers(state: State, llm):
    """
    Final Synthesis Node (Map-Reduce Join).
    
    Role: Collects answers from parallel sub-query agents and fuses them into 
    a single cohesive narrative for the end-user.
    """
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = await llm.with_config(tags=["aggregate_answers"]).ainvoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    
    return {"messages": [AIMessage(content=synthesis_response.content)]}