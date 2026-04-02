import logging
import uuid

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models
from redis.asyncio import Redis

from src.db.qdrant import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from src.db.redis import get_cached_embedding, set_cached_embedding

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64


async def embed_dense(
    openai: AsyncOpenAI,
    redis: Redis,
    texts: list[str],
) -> list[list[float]]:
    """Dense 임베딩을 생성한다. Redis 캐시 히트 시 API 호출을 스킵한다."""
    results: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []

    for i, text in enumerate(texts):
        cached = await get_cached_embedding(redis, text)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        logger.info("전체 %d건 캐시 히트", len(texts))
        return results  # type: ignore[return-value]

    uncached_texts = [texts[i] for i in uncached_indices]
    embeddings = await _call_openai_embeddings(openai, uncached_texts)

    for idx, embedding in zip(uncached_indices, embeddings):
        results[idx] = embedding
        await set_cached_embedding(redis, texts[idx], embedding)

    hit = len(texts) - len(uncached_indices)
    logger.info("Dense 임베딩: %d건 캐시 히트, %d건 API 호출", hit, len(uncached_indices))
    return results  # type: ignore[return-value]


async def _call_openai_embeddings(
    openai: AsyncOpenAI,
    texts: list[str],
) -> list[list[float]]:
    """OpenAI 임베딩 API를 배치로 호출한다."""
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = await openai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def embed_sparse(texts: list[str]) -> list[models.SparseVector]:
    """BM25용 Sparse 벡터를 생성한다.

    Qdrant BM25 내장 인덱싱 사용: 클라이언트는 토큰 빈도만 전달하고,
    IDF 가중치는 서버 측에서 적용된다 (SparseVectorParams modifier=IDF).
    """
    return [_tokenize_to_sparse(text) for text in texts]


def _tokenize_to_sparse(text: str) -> models.SparseVector:
    """텍스트를 whitespace 토큰화하여 Sparse 벡터로 변환한다."""
    token_counts: dict[int, float] = {}
    for token in text.lower().split():
        token_id = hash(token) % (2**31)
        token_counts[token_id] = token_counts.get(token_id, 0.0) + 1.0

    indices = sorted(token_counts.keys())
    values = [token_counts[i] for i in indices]
    return models.SparseVector(indices=indices, values=values)


async def upsert_documents(
    qdrant: AsyncQdrantClient,
    openai: AsyncOpenAI,
    redis: Redis,
    texts: list[str],
    payloads: list[dict],
    ids: list[str] | None = None,
) -> list[str]:
    """문서를 Dense+Sparse 임베딩 후 Qdrant에 upsert한다.

    Args:
        texts: 임베딩할 텍스트 리스트
        payloads: 각 문서의 메타데이터 (source, language, category, timestamp, url, title)
        ids: 포인트 ID 리스트. None이면 UUID 자동 생성.

    Returns:
        upsert된 포인트 ID 리스트
    """
    point_ids = ids or [str(uuid.uuid4()) for _ in texts]

    dense_vectors = await embed_dense(openai, redis, texts)
    sparse_vectors = embed_sparse(texts)

    points = [
        models.PointStruct(
            id=pid,
            vector={
                DENSE_VECTOR_NAME: dense,
                SPARSE_VECTOR_NAME: sparse,
            },
            payload=payload,
        )
        for pid, dense, sparse, payload in zip(point_ids, dense_vectors, sparse_vectors, payloads)
    ]

    for start in range(0, len(points), BATCH_SIZE):
        batch = points[start : start + BATCH_SIZE]
        await qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)

    logger.info("Qdrant upsert 완료: %d건", len(points))
    return point_ids
