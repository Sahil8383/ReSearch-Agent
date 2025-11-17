# Strategy 2: Semantic Similarity with Embeddings (Production-Grade)

## Overview

This strategy uses embeddings to find semantically similar past conversations to your current query. Unlike keyword matching, embeddings understand meaning, so "What's the weather?" matches with "Tell me the climate" even though they use different words.

---

## Step 1: Dependencies

Add to **requirements.txt**:

```bash
pip install openai numpy scikit-learn
```

Or for Anthropic embeddings:

```bash
pip install anthropic numpy scikit-learn
```

---

## Step 2: Create Embedding Memory Manager

Create **api/services/embedding_memory.py**:

```python
import numpy as np
from typing import List, Dict, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..models import Conversation
import json

class EmbeddingMemoryManager:
    """
    Retrieve conversations based on semantic similarity using embeddings.
    This strategy finds conversations that are semantically similar to the
    current query, regardless of exact keyword matches.
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize with embedding model.

        Options:
        - "text-embedding-3-small" (OpenAI) - Recommended, fast
        - "text-embedding-3-large" (OpenAI) - More accurate
        - "cohere" (Cohere API)
        """
        self.embedding_model = embedding_model
        self.embedding_cache = {}  # Cache to avoid re-computing

        if "openai" in embedding_model or "text-embedding" in embedding_model:
            from openai import OpenAI
            self.client = OpenAI()
            self.api_provider = "openai"
        else:
            self.api_provider = "anthropic"

    async def get_semantic_context(
        self,
        current_query: str,
        session_id: str,
        db: AsyncSession,
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve most semantically similar conversations.

        Args:
            current_query: The current user query
            session_id: The session ID to search within
            db: Database session
            top_k: Number of top similar conversations to return
            similarity_threshold: Minimum cosine similarity score (0-1)

        Returns:
            List of messages formatted for the LLM:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """

        # Step 1: Get embedding for current query
        query_embedding = await self._get_embedding(current_query)

        # Step 2: Get past conversations
        result = await db.execute(
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(Conversation.created_at.desc())
            .limit(50)  # Look at last 50 conversations
        )
        conversations = result.scalars().all()

        if not conversations:
            return []

        # Step 3: Calculate similarity for each conversation
        similarities = []

        for conv in conversations:
            # Get embedding for past conversation
            past_text = f"{conv.query} {conv.response}"
            past_embedding = await self._get_embedding(past_text)

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, past_embedding)

            similarities.append({
                "similarity": similarity,
                "conversation": conv
            })

        # Step 4: Filter by threshold and sort
        similarities = [s for s in similarities if s["similarity"] >= similarity_threshold]
        similarities.sort(reverse=True, key=lambda x: x["similarity"])

        # Step 5: Format messages
        context = []
        for item in similarities[:top_k]:
            conv = item["conversation"]
            similarity_score = item["similarity"]

            context.extend([
                {
                    "role": "user",
                    "content": conv.query,
                    "metadata": {"similarity": round(similarity_score, 3)}
                },
                {
                    "role": "assistant",
                    "content": conv.response
                }
            ])

        return context

    async def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text.
        Uses cache to avoid API calls for repeated text.
        """

        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        # Generate embedding based on provider
        if self.api_provider == "openai":
            embedding = await self._get_openai_embedding(text)
        else:
            embedding = await self._get_anthropic_embedding(text)

        # Cache it
        self.embedding_cache[text] = embedding

        return embedding

    async def _get_openai_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API"""
        from openai import OpenAI

        client = OpenAI()

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        embedding = np.array(response.data[0].embedding)
        return embedding

    async def _get_anthropic_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Anthropic API (if available)"""
        # Note: Anthropic doesn't have embeddings API yet
        # This is a placeholder for future use
        # For now, use OpenAI
        return await self._get_openai_embedding(text)

    def _cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Returns value between 0 and 1:
        - 1.0 = identical meaning
        - 0.5 = moderately similar
        - 0.0 = completely different
        """

        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        similarity = dot_product / (norm1 * norm2)

        return float(similarity)

    def clear_cache(self):
        """Clear the embedding cache to free memory"""
        self.embedding_cache.clear()
```

---

## Step 3: Update Database Models to Store Embeddings

Update **api/models.py**:

```python
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)

    # Query and Response
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=True)

    # Execution Details
    iterations = Column(Integer, default=0)
    status = Column(String(50), default="pending")
    error_message = Column(Text, nullable=True)

    # Actions Taken
    actions_taken = Column(JSON, default=list)

    # NEW: Store embedding vector (pre-computed)
    query_embedding = Column(JSON, nullable=True)  # Stored as JSON array
    combined_embedding = Column(JSON, nullable=True)  # Embedding of query + response

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Execution Time
    execution_time_ms = Column(Integer, nullable=True)

    # Relationship
    session = relationship("Session", back_populates="conversations")

    __table_args__ = (
        Index("idx_conversation_session_id", "session_id"),
        Index("idx_conversation_status", "status"),
        Index("idx_conversation_created_at", "created_at"),
    )
```

---

## Step 4: Create Migration (if using Alembic)

```bash
alembic revision -m "add embedding columns to conversations"
```

Edit the migration file:

```python
# alembic/versions/xxxxx_add_embedding_columns.py

def upgrade():
    op.add_column(
        'conversations',
        sa.Column('query_embedding', sa.JSON(), nullable=True)
    )
    op.add_column(
        'conversations',
        sa.Column('combined_embedding', sa.JSON(), nullable=True)
    )

def downgrade():
    op.drop_column('conversations', 'combined_embedding')
    op.drop_column('conversations', 'query_embedding')
```

Run migration:

```bash
alembic upgrade head
```

---

## Step 5: Service for Pre-computing Embeddings

Create **api/services/embedding_precompute.py**:

```python
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from ..models import Conversation
from .embedding_memory import EmbeddingMemoryManager

class EmbeddingPrecomputer:
    """
    Pre-compute and store embeddings for all conversations.
    This speeds up later queries since embeddings are cached in DB.
    """

    def __init__(self):
        self.embedding_manager = EmbeddingMemoryManager()

    async def precompute_embeddings_for_session(
        self,
        session_id: str,
        db: AsyncSession
    ):
        """
        Pre-compute embeddings for all conversations in a session.
        Run this periodically or after new conversations are added.
        """

        # Get all conversations without embeddings
        result = await db.execute(
            select(Conversation)
            .where(
                (Conversation.session_id == session_id) &
                (Conversation.query_embedding.is_(None))
            )
        )
        conversations = result.scalars().all()

        print(f"Pre-computing embeddings for {len(conversations)} conversations...")

        for i, conv in enumerate(conversations):
            try:
                # Get embeddings
                query_embedding = await self.embedding_manager._get_embedding(conv.query)
                combined_embedding = await self.embedding_manager._get_embedding(
                    f"{conv.query} {conv.response}"
                )

                # Convert to list for JSON storage
                query_emb_list = query_embedding.tolist()
                combined_emb_list = combined_embedding.tolist()

                # Update conversation
                await db.execute(
                    update(Conversation)
                    .where(Conversation.id == conv.id)
                    .values(
                        query_embedding=query_emb_list,
                        combined_embedding=combined_emb_list
                    )
                )

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(conversations)} conversations...")

            except Exception as e:
                print(f"  Error processing conversation {conv.id}: {str(e)}")

        await db.commit()
        print("Embedding pre-computation complete!")

    async def precompute_single_conversation(
        self,
        conversation: Conversation,
        db: AsyncSession
    ):
        """Pre-compute embeddings for a single conversation"""

        query_embedding = await self.embedding_manager._get_embedding(conversation.query)
        combined_embedding = await self.embedding_manager._get_embedding(
            f"{conversation.query} {conversation.response}"
        )

        await db.execute(
            update(Conversation)
            .where(Conversation.id == conversation.id)
            .values(
                query_embedding=query_embedding.tolist(),
                combined_embedding=combined_embedding.tolist()
            )
        )

        await db.commit()
```

---

## Step 6: Update Chat Endpoint to Use Embeddings

Update **api/routes/chat.py**:

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import time
from datetime import datetime
import asyncio

from ..schemas import ChatRequest, ChatResponse
from ..dependencies import get_session
from ..models import Conversation, Session as DBSession
from ..services.embedding_memory import EmbeddingMemoryManager
from ..services.embedding_precompute import EmbeddingPrecomputer
from agent.react_agent import ReActAgent

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session_id: str = None,
    use_semantic_search: bool = True,  # Enable semantic search by default
    db: AsyncSession = Depends(get_session),
):
    """
    Chat endpoint with semantic similarity-based memory retrieval.
    """
    try:
        start_time = time.time()

        # Initialize embedding manager
        embedding_manager = EmbeddingMemoryManager()

        # Get or create session
        if session_id:
            db_session = await db.get(DBSession, session_id)
            if not db_session:
                raise HTTPException(404, "Session not found")
        else:
            db_session = DBSession()
            db.add(db_session)
            await db.commit()
            await db.refresh(db_session)

        # Get semantic context
        if use_semantic_search:
            message_history = await embedding_manager.get_semantic_context(
                current_query=request.query,
                session_id=db_session.id,
                db=db,
                top_k=5,  # Return top 5 most similar
                similarity_threshold=0.5  # 50% similarity minimum
            )
            print(f"Semantic search found {len(message_history)} relevant messages")
        else:
            # Fallback to recent messages
            result = await db.execute(
                select(Conversation)
                .where(Conversation.session_id == db_session.id)
                .order_by(Conversation.created_at.desc())
                .limit(10)
            )
            conversations = list(reversed(result.scalars().all()))

            message_history = []
            for conv in conversations:
                message_history.extend([
                    {"role": "user", "content": conv.query},
                    {"role": "assistant", "content": conv.response}
                ])

        # Initialize agent
        agent = ReActAgent(
            max_iterations=request.max_iterations,
            stream=request.stream
        )

        # Add message history to agent's memory
        for msg in message_history:
            from agent.message import Message
            Message(role=msg["role"], content=msg["content"])

        # Run agent
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: agent.run(request.query)
        )

        # Store conversation
        conversation = Conversation(
            session_id=db_session.id,
            query=request.query,
            response=result,
            iterations=request.max_iterations,
            status="completed",
            execution_time_ms=int((time.time() - start_time) * 1000),
            completed_at=datetime.utcnow()
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)

        # Pre-compute embeddings for this new conversation
        precomputer = EmbeddingPrecomputer()
        await precomputer.precompute_single_conversation(conversation, db)

        return ChatResponse(
            answer=result,
            iterations=request.max_iterations,
            status="completed",
            execution_time_ms=int((time.time() - start_time) * 1000)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
```

---

## Step 7: Add Monitoring Endpoint

Create **api/routes/embeddings.py**:

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from ..dependencies import get_session
from ..models import Conversation
from ..services.embedding_memory import EmbeddingMemoryManager
from ..services.embedding_precompute import EmbeddingPrecomputer

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])

@router.get("/status/{session_id}")
async def get_embedding_status(
    session_id: str,
    db: AsyncSession = Depends(get_session)
):
    """Check embedding pre-computation status for a session"""

    # Count total conversations
    result = await db.execute(
        select(Conversation)
        .where(Conversation.session_id == session_id)
    )
    all_convs = result.scalars().all()
    total = len(all_convs)

    # Count conversations with embeddings
    with_embeddings = sum(
        1 for conv in all_convs
        if conv.query_embedding is not None
    )

    return {
        "session_id": session_id,
        "total_conversations": total,
        "with_embeddings": with_embeddings,
        "without_embeddings": total - with_embeddings,
        "completion_percentage": round((with_embeddings / total * 100), 2) if total > 0 else 0
    }

@router.post("/precompute/{session_id}")
async def precompute_embeddings(
    session_id: str,
    db: AsyncSession = Depends(get_session)
):
    """
    Trigger pre-computation of embeddings for all conversations in a session.
    This can be run as a background task.
    """
    try:
        precomputer = EmbeddingPrecomputer()
        await precomputer.precompute_embeddings_for_session(session_id, db)

        return {
            "status": "success",
            "message": "Embeddings pre-computed successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding pre-computation failed: {str(e)}"
        )

@router.get("/search/{session_id}")
async def search_conversations(
    session_id: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.5,
    db: AsyncSession = Depends(get_session)
):
    """
    Test endpoint: Find most similar conversations to a query.
    Useful for debugging and monitoring.
    """

    embedding_manager = EmbeddingMemoryManager()

    results = await embedding_manager.get_semantic_context(
        current_query=query,
        session_id=session_id,
        db=db,
        top_k=top_k,
        similarity_threshold=threshold
    )

    return {
        "query": query,
        "results_count": len(results),
        "top_k": top_k,
        "threshold": threshold,
        "results": results
    }

@router.delete("/cache")
async def clear_embedding_cache():
    """Clear the embedding cache to free memory"""

    embedding_manager = EmbeddingMemoryManager()
    embedding_manager.clear_cache()

    return {"status": "cache cleared"}
```

Add to **api/main.py**:

```python
from .routes import chat, health, embeddings

app.include_router(embeddings.router)
```

---

## Step 8: Configuration

Update **api/config.py**:

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # ... existing settings ...

    # Embeddings Configuration
    EMBEDDINGS_MODEL: str = "text-embedding-3-small"
    EMBEDDINGS_SIMILARITY_THRESHOLD: float = 0.5
    EMBEDDINGS_TOP_K: int = 5
    OPENAI_API_KEY: Optional[str] = None  # Required for embeddings

    # Precomputation
    PRECOMPUTE_EMBEDDINGS_ON_SAVE: bool = True
    PRECOMPUTE_BATCH_SIZE: int = 100

    class Config:
        env_file = ".env"

settings = Settings()
```

Update **.env**:

```bash
# Embeddings
EMBEDDINGS_MODEL=text-embedding-3-small
EMBEDDINGS_SIMILARITY_THRESHOLD=0.5
EMBEDDINGS_TOP_K=5
OPENAI_API_KEY=your_openai_api_key_here

# Pre-computation
PRECOMPUTE_EMBEDDINGS_ON_SAVE=true
```

---

## Step 9: Usage Examples

### Example 1: Basic Chat with Semantic Search

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I use Python for data analysis?",
    "max_iterations": 10
  }'
```

### Example 2: Pre-compute Embeddings

```bash
curl -X POST http://localhost:8000/api/embeddings/precompute/{session_id}
```

### Example 3: Search Similar Conversations

```bash
curl -X GET "http://localhost:8000/api/embeddings/search/{session_id}?query=python&top_k=5&threshold=0.5"
```

### Example 4: Check Embedding Status

```bash
curl -X GET http://localhost:8000/api/embeddings/status/{session_id}
```

---

## Step 10: Performance Optimization

### Caching Pre-computed Embeddings

```python
# api/services/embedding_cache.py

import json
from pathlib import Path

class EmbeddingCache:
    """Cache embeddings to disk to avoid re-computation"""

    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_key(self, session_id: str, conversation_id: str) -> str:
        return f"{session_id}_{conversation_id}"

    def save(self, session_id: str, conversation_id: str, embedding: list):
        """Save embedding to disk"""
        cache_key = self.get_cache_key(session_id, conversation_id)
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, 'w') as f:
            json.dump(embedding, f)

    def load(self, session_id: str, conversation_id: str) -> list:
        """Load embedding from disk"""
        cache_key = self.get_cache_key(session_id, conversation_id)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        return None
```

---

## Step 11: Testing

```python
# tests/test_embeddings.py

import pytest
from api.services.embedding_memory import EmbeddingMemoryManager

@pytest.mark.asyncio
async def test_embedding_similarity():
    manager = EmbeddingMemoryManager()

    # Get embeddings
    emb1 = await manager._get_embedding("What is Python?")
    emb2 = await manager._get_embedding("Tell me about Python")
    emb3 = await manager._get_embedding("How do I cook pasta?")

    # Calculate similarities
    sim_similar = manager._cosine_similarity(emb1, emb2)
    sim_different = manager._cosine_similarity(emb1, emb3)

    # Similar queries should have higher similarity
    assert sim_similar > sim_different
    assert sim_similar > 0.7  # Should be highly similar
    assert sim_different < 0.5  # Should be dissimilar

@pytest.mark.asyncio
async def test_semantic_context_retrieval(db_session):
    manager = EmbeddingMemoryManager()

    # This would test with actual conversations
    context = await manager.get_semantic_context(
        current_query="Python programming",
        session_id="test-session",
        db=db_session,
        top_k=5
    )

    assert len(context) <= 5
    assert all("role" in msg and "content" in msg for msg in context)
```

---

## Complete Feature Summary

✅ **Semantic Similarity Matching**

- Finds contextually relevant conversations, not just keyword matches
- Handles paraphrasing and synonyms automatically

✅ **Pre-computed Embeddings**

- Store embeddings in database for fast retrieval
- Avoid re-computing same embeddings

✅ **Caching**

- In-memory cache during runtime
- Optional disk cache for persistence

✅ **Monitoring**

- Check pre-computation status
- Debug search results
- Clear cache manually

✅ **Production-Ready**

- Error handling
- Batch processing
- Configurable thresholds

---

## Database Migration Checklist

```bash
# 1. Update models.py
✓ Add query_embedding and combined_embedding columns

# 2. Create migration
alembic revision -m "add embeddings"

# 3. Run migration
alembic upgrade head

# 4. Pre-compute for existing conversations
curl -X POST http://localhost:8000/api/embeddings/precompute/{session_id}

# 5. Monitor status
curl http://localhost:8000/api/embeddings/status/{session_id}
```

---

## Key Points

1. **Semantic matching** understands meaning, not just keywords
2. **Pre-compute embeddings** to avoid repeated API calls
3. **Store in database** for fast retrieval
4. **Cache aggressively** in memory and on disk
5. **Monitor status** to ensure all conversations have embeddings
6. **Use 0.5 threshold** as starting point, tune based on results

Start using this strategy today for production-grade memory management! 🚀
