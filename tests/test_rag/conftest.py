from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock


async def async_iter(items):
    """리스트를 async iterator로 변환한다."""
    for item in items:
        yield item


def mock_stream_manager(
    chunks: list[str],
    input_tokens: int = 100,
    output_tokens: int = 50,
):
    """anthropic.messages.stream()이 반환하는 async context manager를 모킹한다."""
    mock_stream = AsyncMock()
    mock_stream.text_stream = async_iter(chunks)
    mock_stream.get_final_message.return_value = MagicMock(
        usage=MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    )

    @asynccontextmanager
    async def stream_cm(**kwargs):
        yield mock_stream

    return stream_cm
