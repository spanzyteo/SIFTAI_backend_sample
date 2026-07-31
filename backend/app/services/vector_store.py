from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class VectorStoreProtocol(Protocol):
    async def initialize(self) -> None: ...

    async def upsert_chunks(self, chunks: list[str], metadata: list[dict[str, Any]]) -> None: ...

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...

    async def delete_document(self, document_id: str) -> None: ...


@dataclass
class VectorStoreService:
    """A lightweight local vector-store shim for fallback development."""

    _entries: list[dict[str, Any]] = field(default_factory=list)

    async def initialize(self) -> None:
        return

    async def upsert_chunks(self, chunks: list[str], metadata: list[dict[str, Any]]) -> None:
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata must be the same length")

        for chunk, item_metadata in zip(chunks, metadata, strict=True):
            self._entries.append({"text": chunk, "metadata": item_metadata, "score": 0.0})

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_tokens = set(query.lower().split())
        if not query_tokens:
            return []

        scored_entries = []
        for entry in self._entries:
            entry_tokens = set(entry["text"].lower().split())
            overlap = len(query_tokens & entry_tokens)
            if overlap:
                scored_entries.append({
                    "text": entry["text"],
                    "metadata": entry["metadata"],
                    "score": float(overlap),
                })

        scored_entries.sort(key=lambda item: item["score"], reverse=True)
        return scored_entries[:top_k]

    async def delete_document(self, document_id: str) -> None:
        self._entries = [entry for entry in self._entries if entry["metadata"].get("document_id") != document_id]


def _to_ahnlich_metadata_value(val: Any, metadata_module: Any) -> Any:
    """Utility to map standard Python types to Ahnlich MetadataValue protobuf wrappers."""
    if isinstance(val, bool):
        return metadata_module.MetadataValue(raw_boolean=val)
    elif isinstance(val, int):
        return metadata_module.MetadataValue(raw_integer=val)
    elif isinstance(val, float):
        return metadata_module.MetadataValue(raw_float=val)
    else:
        return metadata_module.MetadataValue(raw_string=str(val))


def _extract_metadata_value(value: Any) -> Any:
    """Extract primitive Python values safely from Ahnlich metadata wrappers."""
    if value is None:
        return None

    # Handle betterproto / protobuf oneof inspection safely
    if hasattr(value, "raw_string") and value.raw_string:
        return value.raw_string
    if hasattr(value, "raw_integer") and value.raw_integer is not None and value.raw_integer != 0:
        return value.raw_integer
    if hasattr(value, "raw_boolean") and value.raw_boolean is not None:
        return value.raw_boolean
    if hasattr(value, "raw_float") and value.raw_float is not None and value.raw_float != 0.0:
        return value.raw_float

    if hasattr(value, "value"):
        inner = getattr(value, "value")
        if isinstance(inner, dict):
            return {k: _extract_metadata_value(v) for k, v in inner.items()}
        return _extract_metadata_value(inner)

    return str(value)


@dataclass
class AhnlichVectorStoreService:
    """Ahnlich AI-backed vector store implementation using gRPC client."""

    _endpoint: str | None = None
    _host: str | None = None
    _port: int = 1370
    _store_name: str | None = None
    _fallback: VectorStoreService = field(default_factory=VectorStoreService)
    _last_error: str | None = None

    def __post_init__(self) -> None:
        self._endpoint = self._endpoint or os.getenv("AHNLICH_ENDPOINT")
        self._host = self._host or os.getenv("AHNLICH_HOST") or self._resolved_host_from_endpoint()
        self._port = int(os.getenv("AHNLICH_PORT", str(self._port)))
        self._store_name = self._store_name or os.getenv("AHNLICH_STORE_NAME", "legal_docs")

    def _resolved_host_from_endpoint(self) -> str:
        if not self._endpoint:
            return "127.0.0.1"

        parsed = urlparse(self._endpoint)
        return parsed.hostname or "127.0.0.1"

    def _connection_settings(self) -> tuple[str, int]:
        if self._endpoint:
            parsed = urlparse(self._endpoint)
            if parsed.hostname:
                return parsed.hostname, parsed.port or self._port

        return self._host or "127.0.0.1", self._port

    def _has_connection_target(self) -> bool:
        return bool(self._endpoint or self._host or os.getenv("AHNLICH_HOST"))

    def _set_last_error(self, error: Exception | None) -> None:
        if error is None:
            self._last_error = None
            return
        self._last_error = f"{type(error).__name__}: {error}"
        logger.error(f"AhnlichVectorStore Error: {self._last_error}")

    def _clear_last_error(self) -> None:
        self._last_error = None

    async def initialize(self) -> None:
        if not self._has_connection_target():
            await self._fallback.initialize()
            return

        self._clear_last_error()

        try:
            from grpclib.client import Channel
            from ahnlich_client_py.grpc.services.ai_service import AiServiceStub
            from ahnlich_client_py.grpc.ai import query as ai_query
            from ahnlich_client_py.grpc.ai.models import AiModel
        except ImportError as exc:
            self._set_last_error(exc)
            await self._fallback.initialize()
            return

        host, port = self._connection_settings()
        try:
            async with Channel(host=host, port=port) as channel:
                client = AiServiceStub(channel)
                response = await client.list_stores(ai_query.ListStores())
                existing_stores = {store.name for store in response.stores}

                if self._store_name not in existing_stores:
                    await client.create_store(
                        ai_query.CreateStore(
                            store=self._store_name,
                            index_model=AiModel.ALL_MINI_LM_L6_V2,
                            query_model=AiModel.ALL_MINI_LM_L6_V2,
                            predicates=["document_id", "user_id", "page_number", "chunk_id"],
                            error_if_exists=False,
                            store_original=True,
                        )
                    )
        except Exception as exc:
            self._set_last_error(exc)
            await self._fallback.initialize()

    async def upsert_chunks(self, chunks: list[str], metadata: list[dict[str, Any]]) -> None:
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata must be the same length")

        if not self._has_connection_target():
            await self._fallback.upsert_chunks(chunks, metadata)
            return

        try:
            from grpclib.client import Channel
            from ahnlich_client_py.grpc.services.ai_service import AiServiceStub
            from ahnlich_client_py.grpc.ai import query as ai_query
            from ahnlich_client_py.grpc.ai.preprocess import PreprocessAction
            from ahnlich_client_py.grpc import keyval, metadata as metadata_module
        except ImportError as exc:
            self._set_last_error(exc)
            await self._fallback.upsert_chunks(chunks, metadata)
            return

        # Ensure store exists before upserting
        await self.initialize()

        host, port = self._connection_settings()
        try:
            async with Channel(host=host, port=port) as channel:
                client = AiServiceStub(channel)
                inputs = []
                for chunk_text, item_metadata in zip(chunks, metadata, strict=True):
                    metadata_value = {
                        key: _to_ahnlich_metadata_value(value, metadata_module)
                        for key, value in item_metadata.items()
                    }
                    inputs.append(
                        keyval.AiStoreEntry(
                            key=keyval.StoreInput(raw_string=chunk_text),
                            value=keyval.StoreValue(value=metadata_value),
                        )
                    )
                
                await client.set(
                    ai_query.Set(
                        store=self._store_name,
                        inputs=inputs,
                        preprocess_action=PreprocessAction.ModelPreprocessing,
                    )
                )
                self._clear_last_error()
        except Exception as exc:
            self._set_last_error(exc)
            await self._fallback.upsert_chunks(chunks, metadata)

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._has_connection_target():
            return await self._fallback.search(query=query, top_k=top_k)

        try:
            from grpclib.client import Channel
            from ahnlich_client_py.grpc.services.ai_service import AiServiceStub
            from ahnlich_client_py.grpc.ai import query as ai_query
            from ahnlich_client_py.grpc.ai.preprocess import PreprocessAction
            from ahnlich_client_py.grpc.algorithm.algorithms import Algorithm
            from ahnlich_client_py.grpc import keyval
        except ImportError as exc:
            self._set_last_error(exc)
            return await self._fallback.search(query=query, top_k=top_k)

        host, port = self._connection_settings()
        try:
            async with Channel(host=host, port=port) as channel:
                client = AiServiceStub(channel)
                response = await client.get_sim_n(
                    ai_query.GetSimN(
                        store=self._store_name,
                        search_input=keyval.StoreInput(raw_string=query),
                        closest_n=top_k,
                        algorithm=Algorithm.CosineSimilarity,
                        preprocess_action=PreprocessAction.ModelPreprocessing,
                    )
                )
                self._clear_last_error()

                results = []
                for entry in response.entries:
                    extracted_meta = _extract_metadata_value(entry.value)
                    
                    # Resolve raw string from StoreInput
                    chunk_text = ""
                    if hasattr(entry.key, "raw_string") and entry.key.raw_string:
                        chunk_text = entry.key.raw_string
                    elif hasattr(entry, "key") and isinstance(entry.key, str):
                        chunk_text = entry.key

                    results.append({
                        "text": chunk_text,
                        "metadata": extracted_meta if isinstance(extracted_meta, dict) else {},
                        "score": float(entry.score) if hasattr(entry, "score") else 0.0,
                    })

                return results
        except Exception as exc:
            self._set_last_error(exc)
            return await self._fallback.search(query=query, top_k=top_k)

    async def delete_document(self, document_id: str) -> None:
        if not self._has_connection_target():
            await self._fallback.delete_document(document_id)
            return

        try:
            from grpclib.client import Channel
            from ahnlich_client_py.grpc.services.ai_service import AiServiceStub
            from ahnlich_client_py.grpc.ai import query as ai_query
            from ahnlich_client_py.grpc import predicates, metadata as metadata_module
        except ImportError:
            await self._fallback.delete_document(document_id)
            return

        condition = predicates.PredicateCondition(
            value=predicates.Predicate(
                equals=predicates.Equals(
                    key="document_id",
                    value=metadata_module.MetadataValue(raw_string=document_id),
                )
            )
        )

        host, port = self._connection_settings()
        try:
            async with Channel(host=host, port=port) as channel:
                client = AiServiceStub(channel)
                response = await client.get_pred(
                    ai_query.GetPred(
                        store=self._store_name,
                        condition=condition,
                    )
                )
                if response.entries:
                    keys_to_delete = [entry.key for entry in response.entries]
                    await client.del_key(ai_query.DelKey(store=self._store_name, keys=keys_to_delete))
        except Exception as exc:
            self._set_last_error(exc)
            await self._fallback.delete_document(document_id)


def create_vector_store_service() -> VectorStoreProtocol:
    use_ahnlich = os.getenv("USE_AHNLICH", "true").lower() in {"1", "true", "yes", "on"}
    if use_ahnlich:
        return AhnlichVectorStoreService()
    return VectorStoreService()