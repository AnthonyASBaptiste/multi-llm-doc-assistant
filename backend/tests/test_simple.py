
import pytest

def test_simple():
    """Simple synchronous test."""
    assert True

@pytest.mark.asyncio
async def test_async_simple():
    """Simple async test."""
    import asyncio
    await asyncio.sleep(0.01)
    assert True