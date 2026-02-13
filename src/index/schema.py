"""
src/index/schema.py — Qdrant collection schema definition.

Defines the vector + payload schema for the RAG-OS Qdrant collection.
"""

from __future__ import annotations

from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
)


def get_vector_params(dimension: int) -> VectorParams:
    """
    Return VectorParams for the Qdrant collection.

    Uses cosine distance — BGE models produce L2-normalized vectors,
    so cosine and dot-product are equivalent.
    """
    return VectorParams(
        size=dimension,
        distance=Distance.COSINE,
    )


# Payload field schemas for metadata filtering
PAYLOAD_SCHEMA = {
    "doc_id": PayloadSchemaType.KEYWORD,
    "source": PayloadSchemaType.KEYWORD,
    "doc_type": PayloadSchemaType.KEYWORD,
    "section_path": PayloadSchemaType.TEXT,
    "page_start": PayloadSchemaType.INTEGER,
    "page_end": PayloadSchemaType.INTEGER,
    "chunk_index": PayloadSchemaType.INTEGER,
    "chunk_type": PayloadSchemaType.KEYWORD,
}
