from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db.qdrant import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
)


@pytest.fixture
def mock_client():
    return AsyncMock()


async def test_ensure_collection_creates_when_not_exists(mock_client):
    mock_client.get_collections.return_value = MagicMock(collections=[])

    await ensure_collection(mock_client)

    mock_client.create_collection.assert_called_once()
    call_kwargs = mock_client.create_collection.call_args.kwargs
    assert call_kwargs["collection_name"] == COLLECTION_NAME
    assert DENSE_VECTOR_NAME in call_kwargs["vectors_config"]
    assert SPARSE_VECTOR_NAME in call_kwargs["sparse_vectors_config"]

    assert mock_client.create_payload_index.call_count == 4


async def test_ensure_collection_skips_when_exists(mock_client):
    existing = MagicMock()
    existing.name = COLLECTION_NAME
    mock_client.get_collections.return_value = MagicMock(collections=[existing])

    await ensure_collection(mock_client)

    mock_client.create_collection.assert_not_called()
