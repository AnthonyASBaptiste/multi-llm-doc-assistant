from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal

class ChatMessage(BaseModel):
    """A single chat message."""
    role: Literal["user", "assistant", "system"] = Field(
        ...,  # ... means this field is required
        description="Role of the message sender"
    )
    content: str = Field(
        ...,
        description="Content of the message"
    )

class ChatRequest(BaseModel):
    """Request model for chat completions."""
    messages: List[ChatMessage] = Field(
        ...,
        description="List of chat messages"
    )
    model: str = Field(
        default="gpt-3.5-turbo",
        description="Model to use for completion"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature between 0 and 1"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens in response"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages_not_empty(cls, v):
        """Validate that messages list is not empty."""
        if not v:
            raise ValueError("messages list cannot be empty")
        return v

class ChatResponse(BaseModel):
    """Response model for chat completions."""
    message: ChatMessage
    model: str
    usage: dict  # Contains token usage information