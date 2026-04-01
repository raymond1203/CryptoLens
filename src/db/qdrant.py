import logging

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

COLLECTION_NAME = "crypto_docs"
DENSE_VECTOR_SIZE = 1536  # OpenAI text-embedding-3-small
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


async def ensure_collection(client: AsyncQdrantClient) -> None:
    """컬렉션이 없으면 생성하고, Payload 인덱스는 항상 reconciliation한다."""
    collections = await client.get_collections()
    existing = [c.name for c in collections.collections]

    if COLLECTION_NAME not in existing:
        await _create_collection(client)

    await _ensure_payload_indexes(client)


async def _create_collection(client: AsyncQdrantClient) -> None:
    await client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
            ),
        ),
    )
    logger.info("컬렉션 '%s' 생성 완료", COLLECTION_NAME)


async def _ensure_payload_indexes(client: AsyncQdrantClient) -> None:
    indexes = [
        ("source", models.PayloadSchemaType.KEYWORD),
        ("language", models.PayloadSchemaType.KEYWORD),
        ("category", models.PayloadSchemaType.KEYWORD),
        ("timestamp", models.PayloadSchemaType.DATETIME),
    ]
    for field_name, field_schema in indexes:
        try:
            await client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
        except UnexpectedResponse as e:
            if e.status_code == 400:
                logger.debug("인덱스 '%s' 이미 존재, 스킵", field_name)
            else:
                raise
    logger.info("Payload 인덱스 확인 완료")
