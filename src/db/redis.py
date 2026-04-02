import hashlib
import json
import logging

from redis.asyncio import Redis

logger = logging.getLogger(__name__)

EMBEDDING_PREFIX = "emb"
EMBEDDING_TTL = 86400  # 24시간
DEFAULT_CACHE_TTL = 3600  # 1시간


async def get_cached_embedding(redis: Redis, text: str) -> list[float] | None:
    """캐시된 임베딩을 조회한다. 없으면 None 반환."""
    key = _embedding_key(text)
    raw = await redis.get(key)
    if raw is None:
        return None
    logger.debug("임베딩 캐시 히트: %s", key)
    return json.loads(raw)


async def set_cached_embedding(redis: Redis, text: str, embedding: list[float]) -> None:
    """임베딩을 캐시에 저장한다 (24h TTL)."""
    key = _embedding_key(text)
    await redis.set(key, json.dumps(embedding), ex=EMBEDDING_TTL)


async def cache_get(redis: Redis, key: str) -> str | None:
    """범용 캐시 조회."""
    return await redis.get(key)


async def cache_set(redis: Redis, key: str, value: str, ttl: int = DEFAULT_CACHE_TTL) -> None:
    """범용 캐시 저장."""
    await redis.set(key, value, ex=ttl)


def _embedding_key(text: str) -> str:
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    return f"{EMBEDDING_PREFIX}:{text_hash}"
