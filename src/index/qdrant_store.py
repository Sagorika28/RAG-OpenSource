"""
src/index/qdrant_store.py — Qdrant vector store wrapper.

Supports two modes (controlled by config):
  - local:  QdrantClient(path="./qdrant_data")  — embedded, no Docker
  - server: QdrantClient(url="http://localhost:6333") — Docker/remote

Provides CRUD operations: create_collection, upsert_chunks, search,
delete_collection, and get_stats.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    PayloadSchemaType,
)

from src.core.types import Chunk, Hit
from src.index.schema import PAYLOAD_SCHEMA, get_vector_params

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant vector store for chunk storage and retrieval."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Qdrant client based on config.

        Config keys:
            mode: "local" or "server"
            path: directory for embedded mode (mode=local)
            url:  Qdrant server URL (mode=server)
            collection_name: name of the collection
        """
        self.config = config
        self.collection_name = config.get("collection_name", "rag_os_chunks")
        mode = config.get("mode", "local")

        if mode == "server":
            url = config.get("url", "http://localhost:6333")
            logger.info(f"Connecting to Qdrant server at {url}")
            self.client = QdrantClient(url=url)
        else:
            path = config.get("path", "./qdrant_data")
            logger.info(f"Using Qdrant local storage at {path}")
            self.client = QdrantClient(path=path)

    # ------------------------------------------------------------------ #
    #  Collection management                                              #
    # ------------------------------------------------------------------ #

    def create_collection(self, dimension: int, recreate: bool = False) -> None:
        """
        Create the Qdrant collection if it doesn't exist.

        Args:
            dimension: Vector dimensionality (e.g. 384 for BGE-small).
            recreate: If True, delete and recreate the collection.
        """
        exists = self.client.collection_exists(self.collection_name)

        if exists and recreate:
            logger.warning(f"Recreating collection '{self.collection_name}'")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=get_vector_params(dimension),
            )
            # Create payload indexes for filtering
            for field_name, schema_type in PAYLOAD_SCHEMA.items():
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            logger.info(
                f"Created collection '{self.collection_name}' "
                f"(dim={dimension})"
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def delete_collection(self) -> None:
        """Delete the collection entirely."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")

    # ------------------------------------------------------------------ #
    #  Upsert                                                             #
    # ------------------------------------------------------------------ #

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        vectors: np.ndarray,
        batch_size: int = 100,
    ) -> int:
        """
        Upsert chunks with their vectors into Qdrant.

        Args:
            chunks: List of Chunk objects.
            vectors: np.ndarray of shape (len(chunks), dim).
            batch_size: Number of points per upsert batch.

        Returns:
            Number of points upserted.
        """
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {vectors.shape[0]} vectors"
            )

        points: List[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "raw_text": chunk.raw_text,
                "context_header": chunk.context_header,
                **chunk.metadata,
            }
            points.append(PointStruct(
                id=chunk.chunk_id,
                vector=vector.tolist(),
                payload=payload,
            ))

        # Batch upsert
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            total += len(batch)

        logger.info(f"Upserted {total} chunks into '{self.collection_name}'")
        return total

    # ------------------------------------------------------------------ #
    #  Search                                                             #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        doc_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Hit]:
        """
        Search for similar chunks.

        Args:
            query_vector: Query embedding of shape (dim,) or (1, dim).
            top_k: Number of results to return.
            doc_type: Optional filter by document type.
            source: Optional filter by source filename.

        Returns:
            List of Hit objects sorted by score (descending).
        """
        # Flatten to 1-D
        if query_vector.ndim > 1:
            query_vector = query_vector.squeeze()

        # Build optional filters
        conditions = []
        if doc_type:
            conditions.append(
                FieldCondition(
                    key="doc_type", match=MatchValue(value=doc_type)
                )
            )
        if source:
            conditions.append(
                FieldCondition(
                    key="source", match=MatchValue(value=source)
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        hits: List[Hit] = []
        for point in response.points:
            payload = point.payload or {}
            chunk = Chunk(
                chunk_id=payload.get("chunk_id", str(point.id)),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                raw_text=payload.get("raw_text", ""),
                context_header=payload.get("context_header", ""),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k not in ("text", "raw_text", "context_header",
                                 "chunk_id", "doc_id")
                },
            )
            hits.append(Hit(chunk=chunk, score=point.score))

        return hits

    # ------------------------------------------------------------------ #
    #  Stats                                                              #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            stats: Dict[str, Any] = {
                "collection": self.collection_name,
                "points_count": getattr(info, "points_count", None),
                "status": str(getattr(info, "status", "unknown")),
            }
            # vectors_count was removed in newer qdrant-client versions
            if hasattr(info, "vectors_count"):
                stats["vectors_count"] = info.vectors_count
            # Safely get vector size from config
            try:
                vectors_cfg = info.config.params.vectors
                if hasattr(vectors_cfg, "size"):
                    stats["vector_size"] = vectors_cfg.size
                elif isinstance(vectors_cfg, dict):
                    # Named vectors
                    for name, cfg in vectors_cfg.items():
                        stats["vector_size"] = getattr(cfg, "size", None)
                        break
            except Exception:
                stats["vector_size"] = None
            return stats
        except Exception as e:
            logger.warning(f"Could not get stats: {e}")
            return {"error": str(e)}
