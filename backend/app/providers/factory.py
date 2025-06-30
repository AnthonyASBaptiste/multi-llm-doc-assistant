from typing import Dict
from app.providers.base import LLMProvider
from app.providers.openai_provider import OpenAIProvider


class LLMProviderFactory:
    """Factory for creating and managing LLM providers."""

    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}

    async def initialize_provider(self, provider_name: str, api_key: str) -> LLMProvider:
        """
        Initialize and return a provider instance.

        Args:
            provider_name (str): Name of the provider to initialize
            api_key (str): API key for the provider

        Returns:
            LLMProvider: Initialized provider instance

        Raises:
            ValueError: If provider_name is not supported
        """
        if provider_name in self.providers:
            return self.providers[provider_name]

        if provider_name == "openai":
            provider = OpenAIProvider(api_key)
            await provider.initialize()
            self.providers[provider_name] = provider
            return provider

        raise ValueError(f"Unsupported provider: {provider_name}")

    def get_provider(self, provider_name: str) -> LLMProvider:
        """
        Get an initialized provider instance.

        Args:
            provider_name (str): Name of the provider

        Returns:
            LLMProvider: Provider instance

        Raises:
            ValueError: If provider is not initialized
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not initialized")
        return self.providers[provider_name]