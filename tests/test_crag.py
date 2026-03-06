import pytest
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from agentic_rag.rag_agent.nodes import grade_documents, fail_response
from agentic_rag.rag_agent.edges import decide_to_generate
from unittest.mock import MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_grade_documents_relevant():
    """Test that grade_documents identifies relevant content."""
    mock_llm = MagicMock()
    # Mock LLM response for grading
    mock_response = MagicMock()
    mock_response.binary_score = "yes"
    
    mock_llm.with_config.return_value.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_response)
    
    state = {
        "question": "What is the capital of France?",
        "messages": [
            ToolMessage(content="Paris is the capital of France.", tool_call_id="1")
        ]
    }
    
    result = await grade_documents(state, mock_llm)
    assert result["is_relevant"] is True

@pytest.mark.asyncio
async def test_grade_documents_irrelevant():
    """Test that grade_documents identifies irrelevant content."""
    mock_llm = MagicMock()
    # Mock LLM response for grading
    mock_response = MagicMock()
    mock_response.binary_score = "no"
    
    mock_llm.with_config.return_value.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_response)
    
    state = {
        "question": "What is the capital of France?",
        "messages": [
            ToolMessage(content="The Eiffel Tower is in Paris.", tool_call_id="1") 
        ]
    }
    
    result = await grade_documents(state, mock_llm)
    assert result["is_relevant"] is False

def test_decide_to_generate_logic():
    """Test the routing logic after grading."""
    state_ok = {"is_relevant": True}
    assert decide_to_generate(state_ok) == "collect_answer"
    
    state_fail = {"is_relevant": False}
    assert decide_to_generate(state_fail) == "fail_response"

def test_fail_response_content():
    """Test the fail response message content."""
    state = {"question": "What is X?"}
    result = fail_response(state)
    assert "couldn't find any information" in result["messages"][0].content
    assert "What is X?" in result["messages"][0].content
