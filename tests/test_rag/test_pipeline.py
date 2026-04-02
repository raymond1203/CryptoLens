from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models

from src.rag.pipeline import rag_query
from tests.test_rag.conftest import mock_stream_manager


@pytest.fixture
def mock_deps():
    """RAG 파이프라인 의존성을 모킹한다."""
    qdrant = AsyncMock()
    openai = AsyncMock()
    redis = AsyncMock()
    anthropic = MagicMock()

    # embed_dense: 캐시 미스 → OpenAI 호출
    redis.get.return_value = None
    openai.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )

    # hybrid_search: 검색 결과 1건
    point = MagicMock(spec=models.ScoredPoint)
    point.payload = {
        "title": "Bitcoin News",
        "source": "coindesk",
        "url": "https://example.com",
        "text": "Bitcoin reached $100k.",
    }
    qdrant.query_points.return_value = MagicMock(points=[point])

    # generate_stream: 스트리밍 응답
    anthropic.messages.stream = mock_stream_manager(["비트코인이 ", "$100k를 돌파했습니다."])

    return {"qdrant": qdrant, "openai": openai, "redis": redis, "anthropic": anthropic}


class TestRagQuery:
    async def test_full_pipeline_streams_response(self, mock_deps):
        result = []
        async for text in rag_query("비트코인 뉴스", **mock_deps):
            result.append(text)

        assert "".join(result) == "비트코인이 $100k를 돌파했습니다."

    async def test_returns_no_results_message_when_empty(self, mock_deps):
        mock_deps["qdrant"].query_points.return_value = MagicMock(points=[])

        result = []
        async for text in rag_query("알 수 없는 질문", **mock_deps):
            result.append(text)

        assert "검색 결과가 없습니다" in "".join(result)

    async def test_passes_filters_to_search(self, mock_deps):
        _ = [
            text
            async for text in rag_query(
                "이더리움 거버넌스",
                **mock_deps,
                sources=["snapshot"],
                languages=["en"],
            )
        ]

        call_kwargs = mock_deps["qdrant"].query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is not None
