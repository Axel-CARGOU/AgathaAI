import asyncio
import hashlib
import logging
import math
import re
import unicodedata
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

from src.config.config import settings


logger = logging.getLogger("app")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
META_POINT_ID = "00000000-0000-0000-0000-000000000000"


@dataclass(slots=True)
class MemoryEntry:
    id: str
    created_at: str
    source: str
    user_id: str | None
    user_name: str
    user_prompt: str
    llm_response: str
    has_image: bool = False
    channel_id: str | None = None
    guild_id: str | None = None
    message_id: str | None = None
    is_dm: bool | None = None

    @classmethod
    def from_interaction(
        cls,
        user_prompt: str,
        llm_response: str,
        *,
        source: str,
        user_name: str,
        user_id: str | None,
        has_image: bool,
        metadata: dict[str, Any] | None,
    ) -> "MemoryEntry":
        metadata = metadata or {}
        return cls(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=_safe_str(source, "unknown"),
            user_id=_optional_str(user_id),
            user_name=_safe_str(user_name, "User"),
            user_prompt=_clean_text(user_prompt, settings.RAG_MAX_STORED_CHARS),
            llm_response=_clean_text(llm_response, settings.RAG_MAX_STORED_CHARS),
            has_image=has_image,
            channel_id=_optional_str(metadata.get("channel_id")),
            guild_id=_optional_str(metadata.get("guild_id")),
            message_id=_optional_str(metadata.get("message_id")),
            is_dm=metadata.get("is_dm") if isinstance(metadata.get("is_dm"), bool) else None,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=_safe_str(payload.get("id"), str(uuid.uuid4())),
            created_at=_safe_str(payload.get("created_at"), ""),
            source=_safe_str(payload.get("source"), "unknown"),
            user_id=_optional_str(payload.get("user_id")),
            user_name=_safe_str(payload.get("user_name"), "User"),
            user_prompt=_safe_str(payload.get("user_prompt"), ""),
            llm_response=_safe_str(payload.get("llm_response"), ""),
            has_image=bool(payload.get("has_image")),
            channel_id=_optional_str(payload.get("channel_id")),
            guild_id=_optional_str(payload.get("guild_id")),
            message_id=_optional_str(payload.get("message_id")),
            is_dm=payload.get("is_dm") if isinstance(payload.get("is_dm"), bool) else None,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["memory_text"] = self.searchable_text()
        return payload

    def searchable_text(self) -> str:
        image_marker = " [image attached but not stored]" if self.has_image else ""
        return (
            f"User {self.user_name}{image_marker}: {self.user_prompt}\n"
            f"AgathaAI: {self.llm_response}"
        ).strip()


class _HashEmbedder:
    """Small deterministic fallback when FastEmbed is unavailable."""

    def __init__(self, size: int):
        self.size = max(64, int(size or 384))
        self.identity = f"hash:{self.size}"

    def embed(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        features = tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        vector = [0.0] * self.size

        for feature in features or ["empty"]:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.size
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = 1.35 if " " in feature else 1.0
            vector[index] += sign * weight

        return _normalize(vector)


"""class _FastEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        probe = self.embed("dimension probe")
        self.size = len(probe)
        self.identity = f"fastembed:{self.model_name}:{self.size}"

    def embed(self, text: str) -> list[float]:
        embedding = next(self._model.embed([text or " "]))
        return [float(value) for value in embedding]"""
        

class _FastEmbedder:                                                                                       
    def __init__(self, model_name: str):
        self.model_name = model_name

        if model_name == "intfloat/multilingual-e5-small":
            try:
                TextEmbedding.add_custom_model(
                    model="intfloat/multilingual-e5-small",
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf="intfloat/multilingual-e5-small"),
                    dim=384,
                    model_file="onnx/model.onnx",
                )
            except ValueError as exc:
                if "already registered" not in str(exc).lower():
                    raise

        self._model = TextEmbedding(model_name=model_name)
        probe = self.embed("dimension probe")
        self.size = len(probe)
        self.identity = f"fastembed:{self.model_name}:{self.size}"

    def embed(self, text: str) -> list[float]:
        embedding = next(self._model.embed([text or " "]))
        return [float(value) for value in embedding]


class RAGMemory:
    def __init__(self) -> None:
        self.short_term_limit = _positive_int(settings.RAG_SHORT_TERM_LIMIT, 10)
        self.long_term_context_limit = _positive_int(settings.RAG_LONG_TERM_CONTEXT_LIMIT, 5)
        self.max_context_chars = _positive_int(settings.RAG_MAX_CONTEXT_CHARS, 4000)
        self.max_entry_chars = _positive_int(settings.RAG_MAX_ENTRY_CHARS, 650)
        self.collection_name = _safe_str(
            settings.RAG_COLLECTION_NAME,
            "agathaai_long_term_memory",
        )

        self._short_term: deque[MemoryEntry] = deque()
        self._memory_lock = asyncio.Lock()
        self._client_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._long_term_ready = False
        self._client = None
        self._models = None
        self._embedder = None
        self._long_term_warning_logged = False

    async def add_interaction(
        self,
        user_prompt: str,
        llm_response: str,
        *,
        source: str = "unknown",
        user_name: str = "User",
        user_id: str | None = None,
        has_image: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not settings.RAG_ENABLED:
            return

        if not _clean_text(user_prompt, 256) and not _clean_text(llm_response, 256):
            return

        entry = MemoryEntry.from_interaction(
            user_prompt,
            llm_response,
            source=source,
            user_name=user_name,
            user_id=user_id,
            has_image=has_image,
            metadata=metadata,
        )

        overflow: list[MemoryEntry] = []
        async with self._memory_lock:
            self._short_term.append(entry)
            while len(self._short_term) > self.short_term_limit:
                overflow.append(self._short_term.popleft())

        if overflow:
            stored = await self._store_long_term(overflow, wait=False)
            if stored != len(overflow) and not self._long_term_warning_logged:
                self._long_term_warning_logged = True
                logger.warning(
                    "\nRAG long-term memory unavailable; %s old short-term entries were not persisted.",
                    len(overflow) - stored,
                )

    async def build_context(
        self,
        query: str,
        *,
        language: str = "fr",
    ) -> str:
        if not settings.RAG_ENABLED:
            return ""

        async with self._memory_lock:
            recent_entries = list(self._short_term)

        long_term_entries = await self._search_long_term(query, self.long_term_context_limit)

        return self._format_context(
            recent_entries=recent_entries,
            long_term_entries=long_term_entries,
            language=language,
        )

    async def flush_short_term(self) -> int:
        if not settings.RAG_ENABLED:
            return 0

        async with self._memory_lock:
            entries = list(self._short_term)

        if not entries:
            return 0

        stored = await self._store_long_term(entries, wait=True)

        if stored == len(entries):
            async with self._memory_lock:
                for entry in entries:
                    try:
                        self._short_term.remove(entry)
                    except ValueError:
                        pass

        return stored

    def short_term_snapshot(self) -> list[MemoryEntry]:
        return list(self._short_term)

    def reset_long_term(self) -> None:
        client = self._client
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

        self._client = None
        self._models = None
        self._embedder = None
        self._initialized = False
        self._long_term_ready = False
        self._long_term_warning_logged = False

    async def list_entries(self, query: str = "", limit: int = 100) -> list[MemoryEntry]:
        limit = max(1, int(limit or 100))
        query = _clean_text(query, settings.RAG_MAX_STORED_CHARS)

        async with self._memory_lock:
            short_entries = list(self._short_term)

        if query:
            long_entries = await self._search_long_term(query, max(0, limit - len(short_entries)))
            entries = [*short_entries, *long_entries]
        else:
            long_entries = await self._list_long_term(limit=max(0, limit - len(short_entries)))
            entries = [*short_entries, *long_entries]

        return entries[:limit]

    async def add_manual_entry(
        self,
        *,
        user_prompt: str,
        llm_response: str = "",
        user_name: str = "Control Panel",
        source: str = "control_panel",
    ) -> MemoryEntry:
        entry = MemoryEntry.from_interaction(
            user_prompt,
            llm_response,
            source=source,
            user_name=user_name,
            user_id=None,
            has_image=False,
            metadata={},
        )

        async with self._memory_lock:
            self._short_term.append(entry)

        return entry

    async def update_entry(self, memory_id: str, payload: dict[str, Any]) -> MemoryEntry | None:
        memory_id = _safe_str(memory_id, "")
        if not memory_id:
            return None

        async with self._memory_lock:
            for index, entry in enumerate(self._short_term):
                if entry.id != memory_id:
                    continue

                updated = _entry_with_updates(entry, payload)
                self._short_term[index] = updated
                return updated

        long_entry = await self._get_long_term_by_id(memory_id)
        if long_entry is None:
            return None

        updated = _entry_with_updates(long_entry, payload)
        await self._replace_long_term(updated)
        return updated

    async def delete_entry(self, memory_id: str) -> bool:
        memory_id = _safe_str(memory_id, "")
        if not memory_id:
            return False

        async with self._memory_lock:
            for entry in list(self._short_term):
                if entry.id == memory_id:
                    self._short_term.remove(entry)
                    return True

        return await self._delete_long_term(memory_id)

    async def clear_short_term(self) -> int:
        async with self._memory_lock:
            count = len(self._short_term)
            self._short_term.clear()
        return count

    async def _store_long_term(self, entries: list[MemoryEntry], *, wait: bool) -> int:
        if not entries:
            return 0
        if not await self._ensure_long_term_ready():
            return 0

        try:
            async with self._client_lock:
                await asyncio.to_thread(self._store_long_term_sync, entries, wait)
            self._long_term_warning_logged = False
            return len(entries)
        except Exception as exc:
            self._long_term_ready = False
            logger.exception("\nUnable to write RAG long-term memory: %s", exc)
            return 0

    async def _search_long_term(self, query: str, limit: int) -> list[MemoryEntry]:
        query = _clean_text(query, settings.RAG_MAX_STORED_CHARS)
        if limit <= 0 or len(query) < 2:
            return []
        if not await self._ensure_long_term_ready():
            return []

        try:
            async with self._client_lock:
                return await asyncio.to_thread(self._search_long_term_sync, query, limit)
        except Exception as exc:
            self._long_term_ready = False
            logger.exception("\nUnable to query RAG long-term memory: %s", exc)
            return []

    async def _list_long_term(self, limit: int) -> list[MemoryEntry]:
        if limit <= 0:
            return []
        if not await self._ensure_long_term_ready():
            return []

        try:
            async with self._client_lock:
                return await asyncio.to_thread(self._list_long_term_sync, limit)
        except Exception as exc:
            self._long_term_ready = False
            logger.exception("\nUnable to list RAG long-term memory: %s", exc)
            return []

    async def _get_long_term_by_id(self, memory_id: str) -> MemoryEntry | None:
        if not await self._ensure_long_term_ready():
            return None

        try:
            async with self._client_lock:
                return await asyncio.to_thread(self._get_long_term_by_id_sync, memory_id)
        except Exception as exc:
            self._long_term_ready = False
            logger.exception("\nUnable to retrieve RAG long-term memory: %s", exc)
            return None

    async def _replace_long_term(self, entry: MemoryEntry) -> None:
        if not await self._ensure_long_term_ready():
            return

        try:
            async with self._client_lock:
                await asyncio.to_thread(self._replace_long_term_sync, entry)
        except Exception as exc:
            self._long_term_ready = False
            logger.exception("\nUnable to update RAG long-term memory: %s", exc)

    async def _delete_long_term(self, memory_id: str) -> bool:
        if not await self._ensure_long_term_ready():
            return False

        try:
            async with self._client_lock:
                await asyncio.to_thread(self._delete_long_term_sync, memory_id)
            return True
        except Exception as exc:
            self._long_term_ready = False
            logger.exception("\nUnable to delete RAG long-term memory: %s", exc)
            return False

    async def _ensure_long_term_ready(self) -> bool:
        if not settings.RAG_ENABLED:
            return False
        if self._initialized:
            return self._long_term_ready

        async with self._init_lock:
            if self._initialized:
                return self._long_term_ready

            try:
                await asyncio.to_thread(self._init_long_term_sync)
                self._long_term_ready = True
                logger.info(
                    "\nRAG long-term memory ready: collection=%s, short_term_limit=%s, long_term_context_limit=%s",
                    self.collection_name,
                    self.short_term_limit,
                    self.long_term_context_limit,
                )
            except ModuleNotFoundError as exc:
                self._long_term_ready = False
                logger.warning(
                    "\nRAG long-term memory disabled because %s is missing. "
                    "Install with: pip install \"qdrant-client[fastembed]>=1.14.2\". "
                    "Short-term memory still works.",
                    exc.name,
                )
            except Exception as exc:
                self._long_term_ready = False
                logger.exception("\nRAG long-term memory disabled: %s", exc)
            finally:
                self._initialized = True

        return self._long_term_ready

    def _init_long_term_sync(self) -> None:
        from qdrant_client import QdrantClient, models

        self._models = models
        qdrant_path = _resolve_path(settings.RAG_QDRANT_PATH)
        qdrant_path.mkdir(parents=True, exist_ok=True)

        self._client = QdrantClient(path=str(qdrant_path))
        self._embedder = self._make_embedder()
        self._ensure_collection()

    def _make_embedder(self):
        model_name = _safe_str(settings.RAG_EMBEDDING_MODEL, "")
        if model_name:
            try:
                embedder = _FastEmbedder(model_name)
                logger.info("\nRAG embeddings using FastEmbed model: %s", model_name)
                return embedder
            except ModuleNotFoundError as exc:
                logger.warning(
                    "\nFastEmbed dependency %s is not installed. Falling back to deterministic hash embeddings for RAG.",
                    exc.name,
                )
            except Exception as exc:
                logger.warning(
                    "\nUnable to load FastEmbed model %s for RAG (%s). Falling back to hash embeddings.",
                    model_name,
                    exc,
                )

        logger.info(
            "\nRAG embeddings using deterministic hash fallback (%s dimensions).",
            settings.RAG_HASH_EMBEDDING_SIZE,
        )
        return _HashEmbedder(settings.RAG_HASH_EMBEDDING_SIZE)

    def _ensure_collection(self) -> None:
        exists = self._collection_exists()
        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._models.VectorParams(
                    size=self._embedder.size,
                    distance=self._models.Distance.COSINE,
                ),
            )
            self._upsert_collection_meta()
            return

        collection_size = self._collection_vector_size()
        if collection_size is not None and collection_size != self._embedder.size:
            raise RuntimeError(
                f"Qdrant collection {self.collection_name!r} has vector size "
                f"{collection_size}, but current RAG embedder uses {self._embedder.size}. "
                "Change RAG_COLLECTION_NAME or migrate the old collection."
            )

        stored_embedder = self._collection_embedder_identity()
        if stored_embedder is None:
            logger.warning(
                "\nRAG collection %s has no embedding metadata. Marking it as %s.",
                self.collection_name,
                self._embedder.identity,
            )
            self._upsert_collection_meta()
        elif stored_embedder != self._embedder.identity:
            raise RuntimeError(
                f"Qdrant collection {self.collection_name!r} was indexed with "
                f"{stored_embedder!r}, but current RAG embedder is {self._embedder.identity!r}. "
                "Change RAG_COLLECTION_NAME or migrate/rebuild the collection."
            )

    def _collection_exists(self) -> bool:
        if hasattr(self._client, "collection_exists"):
            return bool(self._client.collection_exists(self.collection_name))

        try:
            self._client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def _collection_vector_size(self) -> int | None:
        info = self._client.get_collection(self.collection_name)
        params = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)

        if hasattr(params, "size"):
            return int(params.size)

        if isinstance(params, dict) and params:
            first = next(iter(params.values()))
            if hasattr(first, "size"):
                return int(first.size)

        return None

    def _collection_embedder_identity(self) -> str | None:
        if not hasattr(self._client, "retrieve"):
            return None

        records = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[META_POINT_ID],
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            return None

        payload = getattr(records[0], "payload", None) or {}
        return payload.get("embedding_provider")

    def _upsert_collection_meta(self) -> None:
        vector = [0.0] * self._embedder.size
        vector[0] = 1.0
        self._client.upsert(
            collection_name=self.collection_name,
            points=[
                self._models.PointStruct(
                    id=META_POINT_ID,
                    vector=vector,
                    payload={
                        "kind": "rag_meta",
                        "embedding_provider": self._embedder.identity,
                        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                )
            ],
            wait=True,
        )

    def _store_long_term_sync(self, entries: list[MemoryEntry], wait: bool) -> None:
        points = []
        for entry in entries:
            vector = self._embedder.embed(entry.searchable_text())
            payload = entry.to_payload()
            payload["kind"] = "memory"
            payload["embedding_provider"] = self._embedder.identity
            points.append(
                self._models.PointStruct(
                    id=entry.id,
                    vector=vector,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=wait,
        )

    def _search_long_term_sync(self, query: str, limit: int) -> list[MemoryEntry]:
        vector = self._embedder.embed(query)

        if hasattr(self._client, "query_points"):
            result = self._client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=self._memory_filter(),
                with_payload=True,
                with_vectors=False,
                limit=limit + 3,
            )
            points = getattr(result, "points", result)
        else:
            points = self._client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=self._memory_filter(),
                with_payload=True,
                with_vectors=False,
                limit=limit + 3,
            )

        memories: list[MemoryEntry] = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            if payload and payload.get("kind") == "memory":
                memories.append(MemoryEntry.from_payload(payload))
            if len(memories) >= limit:
                break

        return memories

    def _list_long_term_sync(self, limit: int) -> list[MemoryEntry]:
        if not hasattr(self._client, "scroll"):
            return []

        points, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._memory_filter(),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        memories: list[MemoryEntry] = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            if payload and payload.get("kind") == "memory":
                memories.append(MemoryEntry.from_payload(payload))

        return memories

    def _get_long_term_by_id_sync(self, memory_id: str) -> MemoryEntry | None:
        if not hasattr(self._client, "retrieve"):
            return None

        records = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[memory_id],
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            return None

        payload = getattr(records[0], "payload", None) or {}
        if not payload or payload.get("kind") != "memory":
            return None

        return MemoryEntry.from_payload(payload)

    def _replace_long_term_sync(self, entry: MemoryEntry) -> None:
        self._delete_long_term_sync(entry.id)
        self._store_long_term_sync([entry], wait=True)

    def _delete_long_term_sync(self, memory_id: str) -> None:
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=self._models.PointIdsList(points=[memory_id]),
            wait=True,
        )

    def _memory_filter(self):
        return self._models.Filter(
            must=[
                self._models.FieldCondition(
                    key="kind",
                    match=self._models.MatchValue(value="memory"),
                )
            ]
        )

    def _format_context(
        self,
        *,
        recent_entries: list[MemoryEntry],
        long_term_entries: list[MemoryEntry],
        language: str,
    ) -> str:
        if not recent_entries and not long_term_entries:
            return ""

        labels = _labels(language)
        budget = self.max_context_chars
        recent_budget = max(800, int(budget * 0.65))
        long_term_budget = max(500, budget - recent_budget)

        recent_blocks = self._format_recent_blocks(recent_entries, recent_budget, labels)
        long_term_blocks = self._format_blocks(long_term_entries, long_term_budget, labels)

        lines = [labels["title"], labels["rule"]]
        if recent_blocks:
                                           
            lines.append(f"{labels['recent']} (du plus ancien au plus recent, le dernier bloc est le plus recent):")
            lines.extend(recent_blocks)
        if long_term_blocks:
            lines.append(labels["long_term"])
            lines.extend(long_term_blocks)
        lines.append(labels["end"])

        context = "\n".join(lines)
        if len(context) > budget:
            context = context[: budget - 12].rstrip() + "\n[...]\n" + labels["end"]
        return context

    def _format_recent_blocks(
        self,
        entries: list[MemoryEntry],
        budget: int,
        labels: dict[str, str],
    ) -> list[str]:
        selected: list[str] = []
        used = 0

        for entry in reversed(entries[-self.short_term_limit :]):
            block = self._format_entry(entry, labels)
            block_len = len(block) + 1
            if used and used + block_len > budget:
                break
            selected.append(block)
            used += block_len

        return list(reversed(selected))

    def _format_blocks(
        self,
        entries: list[MemoryEntry],
        budget: int,
        labels: dict[str, str],
    ) -> list[str]:
        blocks: list[str] = []
        used = 0

        for entry in entries:
            block = self._format_entry(entry, labels)
            block_len = len(block) + 1
            if used and used + block_len > budget:
                break
            blocks.append(block)
            used += block_len

        return blocks

    def _format_entry(self, entry: MemoryEntry, labels: dict[str, str]) -> str:
        prompt = entry.user_prompt or labels["empty_prompt"]
        if entry.has_image:
            prompt = f"{prompt} {labels['image_not_stored']}".strip()

        prompt = _clip(prompt, max(80, int(self.max_entry_chars * 0.4)))
        response = _clip(entry.llm_response, max(120, int(self.max_entry_chars * 0.6)))
        timestamp = _short_time(entry.created_at)

        """return (
            f"- {timestamp} | {entry.user_name}: {prompt}\n"
            f"  AgathaAI: {response}"
        )"""
        return (
            f"- {timestamp}\n"
            f"  USER_NAME: {entry.user_name}\n"
            f"  USER_MESSAGE: {prompt}\n"
            f"  AGATHA_RESPONSE: {response}"
        )


async def build_rag_context(query: str, *, language: str = "fr") -> str:
    return await rag_memory.build_context(query, language=language)


async def init_rag_memory() -> bool:
    return await rag_memory._ensure_long_term_ready()


async def reset_rag_memory() -> None:
    async with rag_memory._client_lock:
        await asyncio.to_thread(rag_memory.reset_long_term)


async def remember_interaction(
    user_prompt: str,
    llm_response: str,
    *,
    source: str = "unknown",
    user_name: str = "User",
    user_id: str | None = None,
    has_image: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    await rag_memory.add_interaction(
        user_prompt,
        llm_response,
        source=source,
        user_name=user_name,
        user_id=user_id,
        has_image=has_image,
        metadata=metadata,
    )


async def flush_short_term_memory() -> int:
    return await rag_memory.flush_short_term()


async def list_memory_entries(query: str = "", limit: int = 100) -> list[dict[str, Any]]:
    entries = await rag_memory.list_entries(query=query, limit=limit)
    return [entry.to_payload() for entry in entries]


async def add_manual_memory(
    user_prompt: str,
    llm_response: str = "",
    *,
    user_name: str = "Control Panel",
) -> dict[str, Any]:
    entry = await rag_memory.add_manual_entry(
        user_prompt=user_prompt,
        llm_response=llm_response,
        user_name=user_name,
    )
    return entry.to_payload()


async def update_memory_entry(memory_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    entry = await rag_memory.update_entry(memory_id, payload)
    return entry.to_payload() if entry else None


async def delete_memory_entry(memory_id: str) -> bool:
    return await rag_memory.delete_entry(memory_id)


async def clear_short_term_entries() -> int:
    return await rag_memory.clear_short_term()


def _resolve_path(raw_path: str) -> Path:
    path = Path(_safe_str(raw_path, "qdrant_data")).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return TOKEN_RE.findall(normalized)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _clean_text(value: Any, max_chars: int) -> str:
    text = _safe_str(value, "")
    text = re.sub(r"\s+", " ", text).strip()
    return _clip(text, max_chars)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _safe_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _entry_with_updates(entry: MemoryEntry, payload: dict[str, Any]) -> MemoryEntry:
    data = entry.to_payload()

    for key in (
        "source",
        "user_id",
        "user_name",
        "user_prompt",
        "llm_response",
        "has_image",
        "channel_id",
        "guild_id",
        "message_id",
        "is_dm",
    ):
        if key in payload:
            data[key] = payload[key]

    data["user_prompt"] = _clean_text(data.get("user_prompt"), settings.RAG_MAX_STORED_CHARS)
    data["llm_response"] = _clean_text(data.get("llm_response"), settings.RAG_MAX_STORED_CHARS)

    return MemoryEntry.from_payload(data)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _short_time(value: str) -> str:
    if not value:
        return "unknown time"
    return value.replace("T", " ").replace("+00:00", " UTC")


def _labels(language: str) -> dict[str, str]:
    if language == "fr":
        return {
            "title": "[MEMOIRE D'AGATHAAI]",
                                                                                                                                    
            "rule": (
                    "Les blocs ci-dessous sont un historique exact de conversations passées, "
                    "du plus ancien au plus récent. Si l'utilisateur demande un code, un nombre, "
                    "une dernière réponse, ou une information exacte présente ici, réponds avec "
                    "la valeur exacte, sans l'inventer ni la reformuler."
                ),
            "recent": "Memoire recente:",
            "long_term": "Souvenirs long-terme pertinents:",
            "end": "[FIN MEMOIRE]",
            "empty_prompt": "[message image ou vide]",
            "image_not_stored": "[image recue, non stockee]",
        }

    return {
        "title": "[AGATHAAI MEMORY]",
                                                                                                                 
        "rule": (
            "The blocks below are an exact history of past conversations, "
            "from oldest to newest. If the user asks for a code, a number, "
            "the last response, or any exact information present here, respond with "
            "the exact value, without inventing or rephrasing it."
        ),
        "recent": "Recent memory:",
        "long_term": "Relevant long-term memories:",
        "end": "[END MEMORY]",
        "empty_prompt": "[image-only or empty message]",
        "image_not_stored": "[image received, not stored]",
    }


rag_memory = RAGMemory()
