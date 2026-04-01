import logging

from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger(__name__)

COLLECTION_NAME = "crypto_docs"
DENSE_VECTOR_SIZE = 1536  # OpenAI text-embedding-3-small
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


async def ensure_collection(client: AsyncQdrantClient) -> None:
    """컬렉션이 없으면 생성한다. 있으면 스킵."""
    collections = await client.get_collections()
    existing = [c.name for c in collections.collections]

    if COLLECTION_NAME in existing:
        logger.info("컬렉션 '%s' 이미 존재, 스킵", COLLECTION_NAME)
        return

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

    # Payload 인덱스 생성
    for field in ["source", "language", "category"]:
        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    await client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="timestamp",
        field_schema=models.PayloadSchemaType.DATETIME,
    )
    logger.info("Payload 인덱스 생성 완료 (source, language, category, timestamp)")
