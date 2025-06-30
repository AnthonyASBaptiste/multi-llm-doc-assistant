import pytest
import os
from unittest.mock import AsyncMock, Mock, patch
from app.models.llm import ChatMessage, ChatRequest
from app.providers import OpenAIProvider, LLMProviderFactory

class MockChatCompletionMessage:
    """Mock OpenAI ChatCompletionMessage."""
    def __init__(self, content: str, role: str):
        self.content = content
        self.role = role
        self.function_call = None
        self.tool_calls = None

class MockChoice:
    """Mock OpenAI Choice."""
    def __init__(self, message: MockChatCompletionMessage):
        self.finish_reason = "stop"
        self.index = 0
        self.message = message
        self.logprobs = None

class MockUsage:
    """Mock OpenAI Usage."""
    def __init__(self):
        self.completion_tokens = 10
        self.prompt_tokens = 8
        self.total_tokens = 18
    
    def model_dump(self):
        return {
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens
        }

class MockChatCompletion:
    """Mock OpenAI ChatCompletion."""
    def __init__(self):
        message = MockChatCompletionMessage(
            content="Hello! I'm an AI assistant.",
            role="assistant"
        )
        self.id = "chat-123"
        self.choices = [MockChoice(message)]
        self.created = 1677858242
        self.model = "gpt-3.5-turbo"
        self.object = "chat.completion"
        self.usage = MockUsage()

@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    return MockChatCompletion()

@pytest.fixture
def api_key():
    """Get API key from environment or use test key."""
    return os.getenv("OPENAI_API_KEY", "test-api-key")

@pytest.mark.asyncio
async def test_openai_provider_initialization(api_key):
    """Test OpenAI provider initialization."""
    provider = OpenAIProvider(api_key)
    
    # Mock the AsyncOpenAI constructor and client methods
    with patch('app.providers.openai_provider.AsyncOpenAI') as mock_openai:
        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock()
        mock_openai.return_value = mock_client
        
        await provider.initialize()
        
        # Verify AsyncOpenAI was called with the correct API key
        mock_openai.assert_called_once_with(api_key=api_key)
        # Verify models.list was called
        mock_client.models.list.assert_called_once()

@pytest.mark.asyncio
async def test_openai_provider_chat_completion(api_key, mock_openai_response):
    """Test OpenAI provider chat completion."""
    provider = OpenAIProvider(api_key)
    
    # Mock the client
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
    provider.client = mock_client
    
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Hello!")],
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    response = await provider.chat_completion(request)
    
    assert response.message.role == "assistant"
    assert response.message.content == "Hello! I'm an AI assistant."
    assert response.model == "gpt-3.5-turbo"
    assert response.usage["total_tokens"] == 18
    
    # Verify the API call was made with correct parameters
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0.7,
        max_tokens=None
    )

@pytest.mark.asyncio
async def test_openai_provider_validate_models(api_key):
    """Test model validation."""
    provider = OpenAIProvider(api_key)
    
    # Mock the client for initialization
    mock_client = AsyncMock()
    mock_client.models.list = AsyncMock()
    provider.client = mock_client
    
    # Test valid model
    valid_models = await provider.validate_models(["gpt-3.5-turbo"])
    assert valid_models == ["gpt-3.5-turbo"]
    
    # Test invalid model
    with pytest.raises(ValueError, match="Model invalid-model is not available"):
        await provider.validate_models(["invalid-model"])

@pytest.mark.asyncio
async def test_openai_provider_not_initialized(api_key):
    """Test error when using uninitialized provider."""
    provider = OpenAIProvider(api_key)
    
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Hello!")]
    )
    
    with pytest.raises(Exception, match="Provider not initialized"):
        await provider.chat_completion(request)

@pytest.mark.asyncio
async def test_provider_factory(api_key):
    """Test provider factory initialization and retrieval."""
    factory = LLMProviderFactory()
    
    # Mock the OpenAI provider initialization
    with patch('app.providers.openai_provider.AsyncOpenAI') as mock_openai:
        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock()
        mock_openai.return_value = mock_client
        
        provider = await factory.initialize_provider("openai", api_key)
        assert isinstance(provider, OpenAIProvider)
        mock_openai.assert_called_once_with(api_key=api_key)
        mock_client.models.list.assert_called_once()
    
    # Test getting initialized provider
    same_provider = factory.get_provider("openai")
    assert provider == same_provider
    
    # Test getting uninitialized provider
    with pytest.raises(ValueError, match="Provider nonexistent not initialized"):
        factory.get_provider("nonexistent")
    
    # Test initializing unsupported provider
    with pytest.raises(ValueError, match="Unsupported provider: unsupported"):
        await factory.initialize_provider("unsupported", api_key)

@pytest.mark.asyncio
async def test_openai_provider_api_error(api_key, mock_openai_response):
    """Test handling of OpenAI API errors."""
    provider = OpenAIProvider(api_key)
    
    # Mock the client to raise an exception
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
    provider.client = mock_client
    
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Hello!")]
    )
    
    with pytest.raises(Exception, match="OpenAI API request failed: API Error"):
        await provider.chat_completion(request)

@pytest.mark.asyncio
async def test_openai_provider_initialization_failure():
    """Test OpenAI provider initialization failure."""
    provider = OpenAIProvider("invalid-api-key")
    
    # Mock the client to raise an exception during initialization
    with patch('app.providers.openai_provider.AsyncOpenAI') as mock_openai:
        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(side_effect=Exception("Invalid API key"))
        mock_openai.return_value = mock_client
        
        with pytest.raises(Exception, match="Failed to initialize OpenAI client: Invalid API key"):
            await provider.initialize()