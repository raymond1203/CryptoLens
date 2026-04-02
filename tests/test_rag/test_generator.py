from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from src.rag.generator import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, generate_stream
from src.rag.prompt import SYSTEM_PROMPT


def _mock_stream_manager(chunks: list[str], input_tokens: int = 100, output_tokens: int = 50):
    """anthropic.messages.stream()이 반환하는 async context manager를 모킹한다."""
    mock_stream = AsyncMock()
    mock_stream.text_stream = _async_iter(chunks)
    mock_stream.get_final_message.return_value = MagicMock(
        usage=MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    )

    @asynccontextmanager
    async def stream_cm(**kwargs):
        yield mock_stream

    return stream_cm


class TestGenerateStream:
    async def test_yields_text_chunks(self):
        mock_anthropic = MagicMock()
        mock_anthropic.messages.stream = _mock_stream_manager(["안녕", "하세요"])

        messages = [{"role": "user", "content": "테스트"}]
        result = [text async for text in generate_stream(mock_anthropic, messages)]

        assert result == ["안녕", "하세요"]

    async def test_passes_system_prompt_and_defaults(self):
        mock_anthropic = MagicMock()
        call_kwargs_capture = {}

        original_cm = _mock_stream_manager(["ok"])

        @asynccontextmanager
        async def capturing_cm(**kwargs):
            call_kwargs_capture.update(kwargs)
            async with original_cm(**kwargs) as stream:
                yield stream

        mock_anthropic.messages.stream = capturing_cm

        messages = [{"role": "user", "content": "질문"}]
        _ = [text async for text in generate_stream(mock_anthropic, messages)]

        assert call_kwargs_capture["model"] == DEFAULT_MODEL
        assert call_kwargs_capture["max_tokens"] == DEFAULT_MAX_TOKENS
        assert call_kwargs_capture["system"] == SYSTEM_PROMPT
        assert call_kwargs_capture["messages"] == messages


async def _async_iter(items):
    for item in items:
        yield item
