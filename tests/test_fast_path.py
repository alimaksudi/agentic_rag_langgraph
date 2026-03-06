import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from agentic_rag.rag_agent.nodes import classify_intent, fast_reply
from agentic_rag.rag_agent.schemas import IntentClassification

def test_fast_reply_node():
    """Verify that fast_reply returns the correct message from state."""
    state = {"fast_reply_msg": "Hello there!"}
    result = fast_reply(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "Hello there!"

def test_fast_reply_default():
    """Verify default message if none in state."""
    state = {}
    result = fast_reply(state)
    assert "Hello!" in result["messages"][0].content

def test_classify_intent_logic():
    """Mock LLM to test classify_intent node."""
    mock_llm = MagicMock()
    mock_llm.with_config.return_value = mock_llm
    
    # Simulate structured output for a 'chat' intent
    mock_response = IntentClassification(intent_type="chat", fast_reply="Hi!")
    mock_llm.with_structured_output.return_value = mock_llm
    mock_llm.invoke.return_value = mock_response
    
    state = {"messages": [HumanMessage(content="Hello")]}
    result = classify_intent(state, mock_llm)
    
    assert result["intent_type"] == "chat"
    assert result["fast_reply_msg"] == "Hi!"
