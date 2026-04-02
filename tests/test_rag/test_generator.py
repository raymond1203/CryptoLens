from contextlib import asynccontextmanager
from unittest.mock import MagicMock

from src.rag.generator import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, generate_stream
from src.rag.prompt import SYSTEM_PROMPT
from tests.test_rag.conftest import mock_stream_manager


class TestGenerateStream:
    async def test_yields_text_chunks(self):
        mock_anthropic = MagicMock()
        mock_anthropic.messages.stream = mock_stream_manager(["안녕", "하세요"])

        messages = [{"role": "user", "content": "테스트"}]
        result = [text async for text in generate_stream(mock_anthropic, messages)]

        assert result == ["안녕", "하세요"]

    async def test_passes_system_prompt_and_defaults(self):
        mock_anthropic = MagicMock()
        call_kwargs_capture = {}

        original_cm = mock_stream_manager(["ok"])

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
