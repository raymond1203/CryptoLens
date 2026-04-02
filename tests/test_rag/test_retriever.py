from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models

from src.db.qdrant import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from src.rag.retriever import PREFETCH_LIMIT, _build_filter, hybrid_search


@pytest.fixture
def mock_qdrant():
    client = AsyncMock()
    client.query_points.return_value = MagicMock(points=[MagicMock(), MagicMock()])
    return client


class TestHybridSearch:
    async def test_calls_query_points_with_prefetch_and_rrf(self, mock_qdrant):
        dense = [0.1, 0.2]
        sparse = models.SparseVector(indices=[1, 2], values=[1.0, 1.0])

        results = await hybrid_search(mock_qdrant, dense, sparse)

        assert len(results) == 2
        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        assert call_kwargs["collection_name"] == COLLECTION_NAME
        assert len(call_kwargs["prefetch"]) == 2
        assert call_kwargs["prefetch"][0].using == DENSE_VECTOR_NAME
        assert call_kwargs["prefetch"][1].using == SPARSE_VECTOR_NAME
        assert call_kwargs["prefetch"][0].limit == PREFETCH_LIMIT
        assert isinstance(call_kwargs["query"], models.FusionQuery)

    async def test_passes_filter_when_provided(self, mock_qdrant):
        dense = [0.1]
        sparse = models.SparseVector(indices=[1], values=[1.0])

        await hybrid_search(mock_qdrant, dense, sparse, sources=["coindesk"])

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is not None
        assert len(call_kwargs["query_filter"].must) == 1

    async def test_no_filter_when_no_conditions(self, mock_qdrant):
        dense = [0.1]
        sparse = models.SparseVector(indices=[1], values=[1.0])

        await hybrid_search(mock_qdrant, dense, sparse)

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is None


class TestBuildFilter:
    def test_returns_none_when_no_conditions(self):
        assert _build_filter(None, None, None, None, None) is None

    def test_single_source_filter(self):
        f = _build_filter(
            sources=["coindesk"], languages=None, categories=None,
            time_from=None, time_to=None,
        )
        assert f is not None
        assert len(f.must) == 1

    def test_multiple_conditions(self):
        f = _build_filter(
            sources=["coindesk"],
            languages=["en", "ko"],
            categories=["news"],
            time_from=datetime(2026, 1, 1),
            time_to=None,
        )
        assert len(f.must) == 4  # source + language + category + timestamp

    def test_time_range_filter(self):
        f = _build_filter(
            sources=None,
            languages=None,
            categories=None,
            time_from=datetime(2026, 1, 1),
            time_to=datetime(2026, 12, 31),
        )
        assert len(f.must) == 1
        condition = f.must[0]
        assert condition.key == "timestamp"
