from abc import ABC, abstractmethod
from app.models.llm import ChatRequest, ChatResponse


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider with the necessary setup."""
        pass

    @abstractmethod
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """
        Generate a chat completion response.

        Args:
            request (ChatRequest): The chat completion request

        Returns:
            ChatResponse: The generated response

        Raises:
            Exception: If the request fails or the provider is not available
        """
        pass

    @abstractmethod
    async def validate_models(self, models: list[str]) -> list[str]:
        """
        Validate that the requested models are available.

        Args:
            models (list[str]): List of model names to validate

        Returns:
            list[str]: List of valid model names

        Raises:
            ValueError: If any of the models are not available
        """
        pass