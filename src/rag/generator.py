import logging
from collections.abc import AsyncGenerator

from anthropic import AsyncAnthropic

from src.rag.prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 4096


async def generate_stream(
    anthropic: AsyncAnthropic,
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AsyncGenerator[str]:
    """Claude API로 스트리밍 응답을 생성한다.

    Args:
        messages: build_rag_prompt()이 반환한 메시지 리스트
        model: Claude 모델 ID
        max_tokens: 최대 출력 토큰 수

    Yields:
        토큰 단위 텍스트 청크
    """
    async with anthropic.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text

        final = await stream.get_final_message()
        logger.info(
            "응답 생성 완료 — input: %d tokens, output: %d tokens",
            final.usage.input_tokens,
            final.usage.output_tokens,
        )
