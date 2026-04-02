import hashlib
import json
from unittest.mock import AsyncMock

import pytest

from src.db.redis import (
    DEFAULT_CACHE_TTL,
    EMBEDDING_PREFIX,
    EMBEDDING_TTL,
    cache_get,
    cache_set,
    get_cached_embedding,
    set_cached_embedding,
)


@pytest.fixture
def mock_redis():
    return AsyncMock()


class TestEmbeddingCache:
    async def test_get_cached_embedding_returns_none_on_miss(self, mock_redis):
        mock_redis.get.return_value = None

        result = await get_cached_embedding(mock_redis, "비트코인 가격")

        assert result is None

    async def test_get_cached_embedding_returns_vector_on_hit(self, mock_redis):
        embedding = [0.1, 0.2, 0.3]
        mock_redis.get.return_value = json.dumps(embedding)

        result = await get_cached_embedding(mock_redis, "비트코인 가격")

        assert result == embedding

    async def test_set_cached_embedding_stores_with_ttl(self, mock_redis):
        text = "이더리움 전망"
        embedding = [0.4, 0.5, 0.6]

        await set_cached_embedding(mock_redis, text, embedding)

        expected_key = f"{EMBEDDING_PREFIX}:{hashlib.sha256(text.encode()).hexdigest()}"
        mock_redis.set.assert_called_once_with(
            expected_key, json.dumps(embedding), ex=EMBEDDING_TTL
        )

    async def test_cache_roundtrip_uses_consistent_key(self, mock_redis):
        """동일 텍스트에 대해 get/set이 같은 키를 사용한다."""
        text = "솔라나 거버넌스"
        mock_redis.get.return_value = None

        await get_cached_embedding(mock_redis, text)
        await set_cached_embedding(mock_redis, text, [1.0])

        get_key = mock_redis.get.call_args[0][0]
        set_key = mock_redis.set.call_args[0][0]
        assert get_key == set_key


class TestGenericCache:
    async def test_cache_get_returns_value(self, mock_redis):
        mock_redis.get.return_value = '{"data": 1}'

        result = await cache_get(mock_redis, "my:key")

        assert result == '{"data": 1}'
        mock_redis.get.assert_called_once_with("my:key")

    async def test_cache_set_stores_with_custom_ttl(self, mock_redis):
        await cache_set(mock_redis, "my:key", "value", ttl=7200)

        mock_redis.set.assert_called_once_with("my:key", "value", ex=7200)

    async def test_cache_set_uses_default_ttl(self, mock_redis):
        await cache_set(mock_redis, "key", "val")

        mock_redis.set.assert_called_once_with("key", "val", ex=DEFAULT_CACHE_TTL)
