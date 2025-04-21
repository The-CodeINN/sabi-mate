import logging
import uuid
from datetime import datetime
from typing import List, Optional

from ai_companion.core.prompts import MEMORY_ANALYSIS_PROMPT
from ai_companion.modules.memory.long_term.vector_store import (
    get_vector_store,
    get_vector_store_async,
)
from ai_companion.settings import settings
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, SecretStr


class MemoryAnalysis(BaseModel):
    """Result of analyzing a message for memory-worthy content."""

    is_important: bool = Field(
        ...,
        description="Whether the message is important enough to be stored as a memory",
    )
    formatted_memory: Optional[str] = Field(..., description="The formatted memory to be stored")


class MemoryManager:
    """Manager class for handling long-term memory operations."""

    def __init__(self):
        self.vector_store = get_vector_store()
        self.logger = logging.getLogger(__name__)
        self.llm = ChatGroq(
            model=settings.SMALL_TEXT_MODEL_NAME,
            api_key=SecretStr(settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None,
            temperature=0.1,
            max_retries=2,
        ).with_structured_output(MemoryAnalysis)

    async def _analyze_memory(self, message: str) -> MemoryAnalysis:
        """Analyze a message to determine importance and format if needed."""
        prompt = MEMORY_ANALYSIS_PROMPT.format(message=message)
        result = await self.llm.ainvoke(prompt)
        if isinstance(result, dict):
            return MemoryAnalysis(**result)
        if not isinstance(result, MemoryAnalysis):
            # Convert BaseModel to MemoryAnalysis if needed
            return MemoryAnalysis(**result.dict())
        return result

    async def extract_and_store_memories(self, message: BaseMessage) -> None:
        """Extract important information from a message and store in vector store."""
        if message.type != "human":
            return

        # Ensure we have a string content to analyze
        content = message.content
        if isinstance(content, list):
            # Convert list content to string by taking text parts
            content = " ".join([item for item in content if isinstance(item, str)])
        elif not isinstance(content, str):
            # If not a string or list, try to convert to string
            content = str(content)

        # Analyze the message for importance and formatting
        analysis = await self._analyze_memory(content)
        if analysis.is_important and analysis.formatted_memory:
            # Get async version of vector store for operations in async context
            vector_store = await get_vector_store_async()

            # Check if similar memory exists
            similar = await vector_store.find_similar_memory_async(analysis.formatted_memory)
            if similar:
                # Skip storage if we already have a similar memory
                self.logger.info(f"Similar memory already exists: '{analysis.formatted_memory}'")
                return

            # Store new memory
            self.logger.info(f"Storing new memory: '{analysis.formatted_memory}'")
            await vector_store.store_memory_async(
                text=analysis.formatted_memory,
                metadata={
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                },
            )

    async def get_relevant_memories_async(self, context: str) -> List[str]:
        """Retrieve relevant memories based on the current context asynchronously."""
        vector_store = await get_vector_store_async()
        memories = await vector_store.search_memories_async(context, k=settings.MEMORY_TOP_K)
        if memories:
            for memory in memories:
                self.logger.debug(f"Memory: '{memory.text}' (score: {memory.score:.2f})")
        return [memory.text for memory in memories]

    def get_relevant_memories(self, context: str) -> List[str]:
        """Retrieve relevant memories based on the current context synchronously.
        Note: This method is blocking and should not be used in async contexts.
        Use get_relevant_memories_async instead when in an async context.
        """
        memories = self.vector_store.search_memories(context, k=settings.MEMORY_TOP_K)
        if memories:
            for memory in memories:
                self.logger.debug(f"Memory: '{memory.text}' (score: {memory.score:.2f})")
        return [memory.text for memory in memories]

    def format_memories_for_prompt(self, memories: List[str]) -> str:
        """Format retrieved memories as bullet points."""
        if not memories:
            return ""
        return "\n".join(f"- {memory}" for memory in memories)


def get_memory_manager() -> MemoryManager:
    """Get a MemoryManager instance."""
    return MemoryManager()


async def get_memory_manager_async() -> MemoryManager:
    """Get a MemoryManager instance with async initialization."""
    manager = MemoryManager()
    # Ensure vector store is initialized asynchronously
    await get_vector_store_async()
    return manager
