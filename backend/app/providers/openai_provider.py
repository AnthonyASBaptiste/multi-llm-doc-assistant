from typing import List
import openai
from openai import AsyncOpenAI
from app.models.llm import ChatRequest, ChatResponse, ChatMessage
from app.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI implementation of the LLM provider interface."""

    def __init__(self, api_key: str):
        """
        Initialize the OpenAI provider.

        Args:
            api_key (str): OpenAI API key
        """
        self.api_key = api_key
        self.client = None
        self.available_models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo-preview"
        ]

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        self.client = AsyncOpenAI(api_key=self.api_key)
        # Verify the API key by making a test request
        try:
            await self.client.models.list()
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {str(e)}")

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion using OpenAI."""
        if not self.client:
            raise Exception("Provider not initialized. Call initialize() first.")

        try:
            # Convert our internal request format to OpenAI's format
            messages = [msg.model_dump() for msg in request.messages]

            # Make the API call
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )

            # Convert OpenAI's response to our internal format
            assistant_message = ChatMessage(
                role="assistant",
                content=response.choices[0].message.content
            )

            return ChatResponse(
                message=assistant_message,
                model=response.model,
                usage=response.usage.model_dump()
            )

        except Exception as e:
            raise Exception(f"OpenAI API request failed: {str(e)}")

    async def validate_models(self, models: List[str]) -> List[str]:
        """Validate that the requested models are available."""
        if not self.client:
            raise Exception("Provider not initialized. Call initialize() first.")

        valid_models = []
        for model in models:
            if model in self.available_models:
                valid_models.append(model)
            else:
                raise ValueError(f"Model {model} is not available")

        return valid_models