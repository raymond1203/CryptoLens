import logging
from datetime import datetime

from qdrant_client import AsyncQdrantClient, models

from src.db.qdrant import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

logger = logging.getLogger(__name__)

PREFETCH_LIMIT = 20
DEFAULT_LIMIT = 10


async def hybrid_search(
    qdrant: AsyncQdrantClient,
    dense_vector: list[float],
    sparse_vector: models.SparseVector,
    *,
    limit: int = DEFAULT_LIMIT,
    sources: list[str] | None = None,
    languages: list[str] | None = None,
    categories: list[str] | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
) -> list[models.ScoredPoint]:
    """Dense + Sparse Prefetch → RRF Fusion 하이브리드 검색을 수행한다.

    Args:
        dense_vector: Dense 임베딩 벡터
        sparse_vector: Sparse(BM25) 벡터
        limit: 최종 반환 개수
        sources: 출처 필터 (coindesk, upbit 등)
        languages: 언어 필터 (en, ko)
        categories: 카테고리 필터 (news, governance 등)
        time_from: 시간 범위 시작
        time_to: 시간 범위 끝

    Returns:
        RRF로 병합된 검색 결과
    """
    query_filter = _build_filter(sources, languages, categories, time_from, time_to)

    results = await qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=PREFETCH_LIMIT),
            models.Prefetch(query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=PREFETCH_LIMIT),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )

    logger.info("하이브리드 검색 완료: %d건 반환", len(results.points))
    return results.points


def _build_filter(
    sources: list[str] | None,
    languages: list[str] | None,
    categories: list[str] | None,
    time_from: datetime | None,
    time_to: datetime | None,
) -> models.Filter | None:
    """메타데이터 조건으로 Qdrant Filter를 구성한다."""
    conditions: list[models.Condition] = []

    if sources:
        conditions.append(
            models.FieldCondition(key="source", match=models.MatchAny(any=sources))
        )
    if languages:
        conditions.append(
            models.FieldCondition(key="language", match=models.MatchAny(any=languages))
        )
    if categories:
        conditions.append(
            models.FieldCondition(key="category", match=models.MatchAny(any=categories))
        )
    if time_from or time_to:
        range_params = {}
        if time_from:
            range_params["gte"] = time_from.isoformat()
        if time_to:
            range_params["lte"] = time_to.isoformat()
        conditions.append(
            models.FieldCondition(key="timestamp", range=models.DatetimeRange(**range_params))
        )

    return models.Filter(must=conditions) if conditions else None
