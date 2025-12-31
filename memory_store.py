# memory_store.py
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from db import engine
from openai import OpenAI
import os
import json
import math
import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"  # or your preferred embedding model


def embed(text: str) -> list[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def _dot(u: list[float], v: list[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def _l2(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1e-8


def _cosine_sim(u: list[float], v: list[float]) -> float:
    return _dot(u, v) / (_l2(u) * _l2(v))


def save_memory(
    user_id: str,
    content: str,
    memory_type: str = "generic",
    importance: int = 3,
    source: str = "chat",
    session_id: Optional[str] = None,
) -> None:
    """
    Save a new memory row (user_id, content, embedding + metadata) into SingleStore.
    Embedding is stored as JSON string.

    memory_type: e.g. 'profile', 'preference', 'goal', 'episode', ...
    importance: 1 (low) .. 5 (very important)
    """
    vector = embed(content)
    embedding_json = json.dumps(vector)

    with engine.connect() as conn:
        conn.execute(text("USE agentic_memory"))
        conn.execute(
            text(
                """
                INSERT INTO memories (
                    user_id, memory_type, importance, source, session_id,
                    content, embedding
                )
                VALUES (
                    :user_id, :memory_type, :importance, :source, :session_id,
                    :content, :embedding
                )
                """
            ),
            {
                "user_id": user_id,
                "memory_type": memory_type,
                "importance": importance,
                "source": source,
                "session_id": session_id,
                "content": content,
                "embedding": embedding_json,
            },
        )
        conn.commit()


def search_memories(
    user_id: str,
    query: str,
    k: int = 5,
    allowed_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic memory search over SingleStore:

    1. Fetch recent memories for this user from SingleStore.
    2. Compute cosine similarity between query embedding and memory embeddings.
    3. Compute recency weight (newer memories get a small boost).
    4. Compute importance weight (importance 1-5).
    5. Combine into a final score and return top-k.

    We avoid MATCH/AGAINST to keep syntax portable and error-free.
    """
    query_vec = embed(query)

    with engine.connect() as conn:
        conn.execute(text("USE agentic_memory"))

        where_clause = "user_id = :user_id"
        params: Dict[str, Any] = {"user_id": user_id}

        if allowed_types:
            # e.g., allowed_types = ['profile', 'preference']
            in_list = ", ".join(f"'{t}'" for t in allowed_types)
            where_clause += f" AND memory_type IN ({in_list})"

        result = conn.execute(
            text(
                f"""
                SELECT
                    content,
                    embedding,
                    importance,
                    memory_type,
                    created_at
                FROM memories
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT 300
                """
            ),
            params,
        )
        rows = result.fetchall()

    now = datetime.datetime.utcnow()
    scored: List[Dict[str, Any]] = []

    for row in rows:
        content = row[0]
        embedding_json = row[1]
        importance = int(row[2])
        memory_type = row[3]
        created_at = row[4]

        try:
            mem_vec = json.loads(embedding_json)
        except Exception:
            continue

        # 1) Cosine similarity
        sim = _cosine_sim(query_vec, mem_vec)

        # 2) Recency weight (decays over ~30 days)
        if isinstance(created_at, str):
            created_dt = datetime.datetime.fromisoformat(created_at)
        else:
            created_dt = created_at

        age_days = (now - created_dt).total_seconds() / 86400.0
        recency_weight = math.exp(-age_days / 30.0)  # ~30-day decay

        # 3) Importance weight: 1..5 -> ~0.5..1.5
        importance_weight = 0.5 + 0.25 * importance

        # Final score (semantic + recency + importance)
        final_score = (
            0.70 * sim +          # main signal: semantic similarity
            0.20 * recency_weight +
            0.10 * (importance_weight / 1.5)
        )

        scored.append(
            {
                "content": content,
                "score": float(final_score),
                "similarity": float(sim),
                "importance": importance,
                "memory_type": memory_type,
                "created_at": created_dt.isoformat(),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]