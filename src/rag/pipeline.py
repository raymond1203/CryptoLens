import logging
from collections.abc import AsyncGenerator
from datetime import datetime

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from src.rag.embedder import embed_dense, embed_sparse
from src.rag.generator import generate_stream
from src.rag.prompt import build_rag_prompt
from src.rag.retriever import hybrid_search

logger = logging.getLogger(__name__)


async def rag_query(
    query: str,
    *,
    qdrant: AsyncQdrantClient,
    openai: AsyncOpenAI,
    redis: Redis,
    anthropic: AsyncAnthropic,
    sources: list[str] | None = None,
    languages: list[str] | None = None,
    categories: list[str] | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    limit: int = 10,
) -> AsyncGenerator[str]:
    """RAG 파이프라인을 실행한다: 쿼리 → 임베딩 → 검색 → 프롬프트 → 스트리밍 응답.

    Yields:
        토큰 단위 텍스트 청크
    """
    # 1. 쿼리 임베딩
    dense_vectors = await embed_dense(openai, redis, [query])
    sparse_vectors = embed_sparse([query])

    # 2. 하이브리드 검색
    results = await hybrid_search(
        qdrant,
        dense_vectors[0],
        sparse_vectors[0],
        limit=limit,
        sources=sources,
        languages=languages,
        categories=categories,
        time_from=time_from,
        time_to=time_to,
    )

    if not results:
        yield "검색 결과가 없습니다. 다른 질문을 시도해 주세요."
        return

    logger.info("RAG 검색 완료: %d건", len(results))

    # 3. 프롬프트 조립
    messages = build_rag_prompt(query, results)

    # 4. 스트리밍 응답 생성
    async for text in generate_stream(anthropic, messages):
        yield text
