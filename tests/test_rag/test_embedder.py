import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models

from src.db.qdrant import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from src.rag.embedder import (
    BATCH_SIZE,
    _tokenize_to_sparse,
    embed_dense,
    embed_sparse,
    upsert_documents,
)


@pytest.fixture
def mock_openai():
    return AsyncMock()


@pytest.fixture
def mock_redis():
    return AsyncMock()


@pytest.fixture
def mock_qdrant():
    return AsyncMock()


def _make_embedding_response(embeddings: list[list[float]]):
    """OpenAI embeddings.create 응답을 모킹한다."""
    data = [MagicMock(embedding=emb) for emb in embeddings]
    return MagicMock(data=data)


class TestEmbedDense:
    async def test_all_cache_hit_skips_api_call(self, mock_openai, mock_redis):
        mock_redis.get.return_value = json.dumps([0.1, 0.2])

        result = await embed_dense(mock_openai, mock_redis, ["텍스트A", "텍스트B"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        mock_openai.embeddings.create.assert_not_called()

    async def test_all_cache_miss_calls_api(self, mock_openai, mock_redis):
        mock_redis.get.return_value = None
        mock_openai.embeddings.create.return_value = _make_embedding_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        result = await embed_dense(mock_openai, mock_redis, ["텍스트A", "텍스트B"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_openai.embeddings.create.assert_called_once()
        assert mock_redis.set.call_count == 2

    async def test_partial_cache_hit(self, mock_openai, mock_redis):
        """첫 번째는 캐시 히트, 두 번째는 미스."""
        mock_redis.get.side_effect = [json.dumps([0.1, 0.2]), None]
        mock_openai.embeddings.create.return_value = _make_embedding_response([[0.3, 0.4]])

        result = await embed_dense(mock_openai, mock_redis, ["캐시됨", "새로운"])

        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
        mock_openai.embeddings.create.assert_called_once()
        assert mock_redis.set.call_count == 1


class TestEmbedSparse:
    def test_returns_sparse_vectors(self):
        result = embed_sparse(["hello world", "test"])

        assert len(result) == 2
        assert all(isinstance(v, models.SparseVector) for v in result)

    def test_sparse_vector_has_indices_and_values(self):
        result = embed_sparse(["bitcoin ethereum bitcoin"])
        vec = result[0]

        assert len(vec.indices) == len(vec.values)
        assert len(vec.indices) == 2  # 고유 토큰 2개
        # "bitcoin"은 2회 등장 — _tokenize_to_sparse와 동일 해시 로직으로 검증
        ref = _tokenize_to_sparse("bitcoin")
        bitcoin_id = ref.indices[0]
        idx = list(vec.indices).index(bitcoin_id)
        assert vec.values[idx] == 2.0

    def test_indices_are_sorted(self):
        result = embed_sparse(["z a m b"])
        vec = result[0]

        assert list(vec.indices) == sorted(vec.indices)


class TestUpsertDocuments:
    async def test_upsert_creates_points_with_both_vectors(
        self, mock_qdrant, mock_openai, mock_redis
    ):
        mock_redis.get.return_value = None
        mock_openai.embeddings.create.return_value = _make_embedding_response([[0.1, 0.2]])

        payloads = [{"source": "coindesk", "language": "en", "category": "news"}]

        ids = await upsert_documents(
            mock_qdrant, mock_openai, mock_redis, ["Bitcoin hits ATH"], payloads
        )

        assert len(ids) == 1
        mock_qdrant.upsert.assert_called_once()
        call_kwargs = mock_qdrant.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == COLLECTION_NAME

        point = call_kwargs["points"][0]
        assert DENSE_VECTOR_NAME in point.vector
        assert SPARSE_VECTOR_NAME in point.vector

    async def test_upsert_uses_provided_ids(self, mock_qdrant, mock_openai, mock_redis):
        mock_redis.get.return_value = None
        mock_openai.embeddings.create.return_value = _make_embedding_response([[0.1]])

        custom_ids = ["my-id-1"]
        ids = await upsert_documents(
            mock_qdrant, mock_openai, mock_redis, ["text"], [{}], ids=custom_ids
        )

        assert ids == ["my-id-1"]

    async def test_upsert_batches_large_input(self, mock_qdrant, mock_openai, mock_redis):
        count = BATCH_SIZE + 10
        mock_redis.get.return_value = None
        mock_openai.embeddings.create.side_effect = [
            _make_embedding_response([[0.1]] * BATCH_SIZE),
            _make_embedding_response([[0.1]] * 10),
        ]

        ids = await upsert_documents(
            mock_qdrant,
            mock_openai,
            mock_redis,
            [f"text-{i}" for i in range(count)],
            [{} for _ in range(count)],
        )

        assert len(ids) == count
        assert mock_qdrant.upsert.call_count == 2
