import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Optional
import asyncio

from ai_companion.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


@dataclass
class Memory:
    """Represents a memory entry in the vector store."""

    text: str
    metadata: dict
    score: Optional[float] = None

    @property
    def id(self) -> Optional[str]:
        return self.metadata.get("id")

    @property
    def timestamp(self) -> Optional[datetime]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None


class VectorStore:
    """A class to handle vector storage operations using Qdrant."""

    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "long_term_memory"
    SIMILARITY_THRESHOLD = 0.9  # Threshold for considering memories as similar

    _instance: Optional["VectorStore"] = None
    _initialized: bool = False

    def __new__(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self._validate_env_vars()
            # Defer loading the model and client until needed to avoid blocking on initialization
            self._model = None
            self._client = None
            self._initialized = True

    @property
    def model(self):
        """Get the embedding model synchronously. Will raise an error if used in an async context."""
        if self._model is None:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    raise RuntimeError(
                        "Cannot access model synchronously in async context. Use 'await get_model()' instead."
                    )
            except RuntimeError:
                # No running event loop, safe to continue synchronously
                pass

            self._model = SentenceTransformer(self.EMBEDDING_MODEL)
        return self._model

    async def get_model(self):
        """Get the embedding model asynchronously."""
        if self._model is None:
            self._model = await asyncio.to_thread(SentenceTransformer, self.EMBEDDING_MODEL)
        return self._model

    @property
    def client(self):
        """Get the Qdrant client synchronously. Will raise an error if used in an async context."""
        if self._client is None:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    raise RuntimeError(
                        "Cannot access client synchronously in async context. Use 'await get_client()' instead."
                    )
            except RuntimeError:
                # No running event loop, safe to continue synchronously
                pass

            self._client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        return self._client

    async def get_client(self):
        """Get the Qdrant client asynchronously."""
        if self._client is None:
            self._client = await asyncio.to_thread(
                QdrantClient, url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
            )
        return self._client

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _collection_exists(self) -> bool:
        """Check if the memory collection exists synchronously."""
        collections = self.client.get_collections().collections
        return any(col.name == self.COLLECTION_NAME for col in collections)

    async def _collection_exists_async(self) -> bool:
        """Check if the memory collection exists asynchronously."""
        client = await self.get_client()
        collections = await asyncio.to_thread(client.get_collections)
        return any(col.name == self.COLLECTION_NAME for col in collections.collections)

    def _create_collection(self) -> None:
        """Create a new collection for storing memories synchronously."""
        sample_embedding = self.model.encode("sample text")
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(sample_embedding),
                distance=Distance.COSINE,
            ),
        )

    async def _create_collection_async(self) -> None:
        """Create a new collection for storing memories asynchronously."""
        model = await self.get_model()
        sample_embedding = await asyncio.to_thread(model.encode, "sample text")
        client = await self.get_client()

        await asyncio.to_thread(
            client.create_collection,
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(sample_embedding),
                distance=Distance.COSINE,
            ),
        )

    def find_similar_memory(self, text: str) -> Optional[Memory]:
        """Find if a similar memory already exists synchronously.

        Args:
            text: The text to search for

        Returns:
            Optional Memory if a similar one is found
        """
        results = self.search_memories(text, k=1)
        if (
            results
            and results[0].score is not None
            and results[0].score >= self.SIMILARITY_THRESHOLD
        ):
            return results[0]
        return None

    async def find_similar_memory_async(self, text: str) -> Optional[Memory]:
        """Find if a similar memory already exists asynchronously.

        Args:
            text: The text to search for

        Returns:
            Optional Memory if a similar one is found
        """
        results = await self.search_memories_async(text, k=1)
        if (
            results
            and results[0].score is not None
            and results[0].score >= self.SIMILARITY_THRESHOLD
        ):
            return results[0]
        return None

    def store_memory(self, text: str, metadata: dict) -> None:
        """Store a new memory in the vector store or update if similar exists synchronously.

        Args:
            text: The text content of the memory
            metadata: Additional information about the memory (timestamp, type, etc.)
        """
        if not self._collection_exists():
            self._create_collection()

        # Check if similar memory exists
        similar_memory = self.find_similar_memory(text)
        if similar_memory and similar_memory.id:
            metadata["id"] = similar_memory.id  # Keep same ID for update

        embedding = self.model.encode(text)
        point = PointStruct(
            id=metadata.get("id", hash(text)),
            vector=embedding.tolist(),
            payload={
                "text": text,
                **metadata,
            },
        )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )

    async def store_memory_async(self, text: str, metadata: dict) -> None:
        """Store a new memory in the vector store or update if similar exists asynchronously.

        Args:
            text: The text content of the memory
            metadata: Additional information about the memory (timestamp, type, etc.)
        """
        if not await self._collection_exists_async():
            await self._create_collection_async()

        # Check if similar memory exists
        similar_memory = await self.find_similar_memory_async(text)
        if similar_memory and similar_memory.id:
            metadata["id"] = similar_memory.id  # Keep same ID for update

        model = await self.get_model()
        embedding = await asyncio.to_thread(model.encode, text)

        point = PointStruct(
            id=metadata.get("id", hash(text)),
            vector=embedding.tolist(),
            payload={
                "text": text,
                **metadata,
            },
        )

        client = await self.get_client()
        await asyncio.to_thread(
            client.upsert,
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )

    def search_memories(self, query: str, k: int = 5) -> List[Memory]:
        """Search for similar memories in the vector store synchronously.

        Args:
            query: Text to search for
            k: Number of results to return

        Returns:
            List of Memory objects
        """
        if not self._collection_exists():
            return []

        query_embedding = self.model.encode(query)
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=k,
        )

        return [
            Memory(
                text=hit.payload.get("text", "") if hit.payload else "",
                metadata={k: v for k, v in hit.payload.items() if k != "text"}
                if hit.payload
                else {},
                score=hit.score,
            )
            for hit in results
        ]

    async def search_memories_async(self, query: str, k: int = 5) -> List[Memory]:
        """Search for similar memories in the vector store asynchronously.

        Args:
            query: Text to search for
            k: Number of results to return

        Returns:
            List of Memory objects
        """
        if not await self._collection_exists_async():
            return []

        model = await self.get_model()
        query_embedding = await asyncio.to_thread(model.encode, query)

        client = await self.get_client()
        results = await asyncio.to_thread(
            client.search,
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=k,
        )

        return [
            Memory(
                text=hit.payload.get("text", "") if hit.payload else "",
                metadata={k: v for k, v in hit.payload.items() if k != "text"}
                if hit.payload
                else {},
                score=hit.score,
            )
            for hit in results
        ]


@lru_cache
def get_vector_store() -> VectorStore:
    """Get or create the VectorStore singleton instance."""
    return VectorStore()


async def get_vector_store_async() -> VectorStore:
    """Get or create the VectorStore singleton instance asynchronously."""
    store = get_vector_store()
    # Initialize the model and client asynchronously
    await store.get_model()
    await store.get_client()
    return store
