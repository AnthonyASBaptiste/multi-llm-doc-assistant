import pytest
from pydantic import ValidationError
from app.models.llm import ChatMessage, ChatRequest, ChatResponse


def test_valid_chat_message():
    """Test creating a valid ChatMessage."""
    message = ChatMessage(role="user", content="Hello!")
    assert message.role == "user"
    assert message.content == "Hello!"

def test_invalid_chat_message_role():
    """Test that invalid roles are rejected."""
    with pytest.raises(ValidationError):
        ChatMessage(role="invalid_role", content="Hello!")

def test_valid_chat_request():
    """Test creating a valid ChatRequest."""
    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="Hello!")
        ]
    )
    assert len(request.messages) == 1
    assert request.model == "gpt-3.5-turbo"  # Check default value
    assert request.temperature == 0.7        # Check default value
    assert request.max_tokens is None        # Check default value

def test_chat_request_with_custom_values():
    """Test ChatRequest with custom values."""
    request = ChatRequest(
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hi!")
        ],
        model="gpt-4",
        temperature=0.5,
        max_tokens=100
    )
    assert len(request.messages) == 2
    assert request.model == "gpt-4"
    assert request.temperature == 0.5
    assert request.max_tokens == 100

def test_valid_chat_response():
    """Test creating a valid ChatResponse."""
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="Hello! How can I help?"),
        model="gpt-3.5-turbo",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    )
    assert response.message.role == "assistant"
    assert response.model == "gpt-3.5-turbo"
    assert response.usage["total_tokens"] == 18

def test_chat_request_empty_messages():
    """Test that ChatRequest requires at least one message."""
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])

def test_chat_request_invalid_temperature():
    """Test temperature validation."""
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[ChatMessage(role="user", content="Hello!")],
            temperature=2.0  # Temperature should be between 0 and 1
        )