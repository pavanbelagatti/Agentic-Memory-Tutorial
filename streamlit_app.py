# streamlit_app.py
"""
Streamlit dashboard for exploring agentic memory stored in SingleStore.

Tabs:
  1. Memory Search  - hybrid search (semantic + recency + importance)
  2. User Profile   - grouped view of profile, goals, preferences
  3. Raw Table      - direct view of memories table
  4. Admin / Hygiene - delete/reset memories

Assumes you already have:
  - db.py          -> exposes `engine`
  - memory_store.py -> exposes `search_memories`, `save_memory`
  - .env with SINGLESTORE_URL, OPENAI_API_KEY
"""

import os
from typing import List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

from db import engine
from memory_store import search_memories

load_dotenv()

# ----------------- Global Config -----------------

st.set_page_config(
    page_title="Agentic Memory Dashboard",
    layout="wide",
)

DEFAULT_USER_ID = "user"  # change if you use something else in main.py

# ----------------- Helper functions -----------------


def fetch_raw_memories(
    user_id: Optional[str] = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Fetch raw memory rows from SingleStore into a DataFrame."""
    with engine.connect() as conn:
        conn.execute(text("USE agentic_memory"))
        if user_id:
            result = conn.execute(
                text(
                    """
                    SELECT
                        id,
                        user_id,
                        memory_type,
                        importance,
                        source,
                        session_id,
                        content,
                        created_at
                    FROM memories
                    WHERE user_id = :user_id
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                ),
                {"user_id": user_id, "limit": limit},
            )
        else:
            result = conn.execute(
                text(
                    """
                    SELECT
                        id,
                        user_id,
                        memory_type,
                        importance,
                        source,
                        session_id,
                        content,
                        created_at
                    FROM memories
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )

        rows = result.fetchall()
        cols = result.keys()

    df = pd.DataFrame(rows, columns=cols)
    return df


def delete_memory_by_id(mem_id: int) -> int:
    with engine.connect() as conn:
        conn.execute(text("USE agentic_memory"))
        result = conn.execute(
            text("DELETE FROM memories WHERE id = :id"),
            {"id": mem_id},
        )
        conn.commit()
        return result.rowcount


def delete_memories_for_user(user_id: str) -> int:
    with engine.connect() as conn:
        conn.execute(text("USE agentic_memory"))
        result = conn.execute(
            text("DELETE FROM memories WHERE user_id = :user_id"),
            {"user_id": user_id},
        )
        conn.commit()
        return result.rowcount


def grouped_profile_view(user_id: str) -> dict:
    """
    Use search_memories to pull profile, goals, preferences separately.
    Returns a dict with lists of memory dicts.
    """
    sections = {}

    # Profile
    sections["Profile"] = search_memories(
        user_id=user_id,
        query="basic information about the user",
        k=10,
        allowed_types=["profile"],
    )

    # Goals
    sections["Goals"] = search_memories(
        user_id=user_id,
        query="user's goals and ambitions",
        k=10,
        allowed_types=["goal"],
    )

    # Preferences
    sections["Preferences"] = search_memories(
        user_id=user_id,
        query="user likes and preferences",
        k=10,
        allowed_types=["preference"],
    )

    return sections


# ----------------- Layout -----------------

st.title("🧠 Agentic Memory Dashboard")
st.caption("Inspect, search, and manage long-term memory for your Agentic AI applications (LangGraph + SingleStore).")

with st.sidebar:
    st.header("Global Settings")
    user_id = st.text_input("User ID", value=DEFAULT_USER_ID)
    st.markdown("---")
    st.markdown("Environment:")
    st.code(f"SINGLESTORE_URL = {os.getenv('SINGLESTORE_URL', 'not set')[:60]}...", language="bash")
    st.caption("Make sure this matches what you use in main.py")

tab_search, tab_profile, tab_table, tab_admin = st.tabs(
    [
        "🔍 Memory Search",
        "👤 User Profile View",
        "📄 Raw Memory Table",
        "🧹 Admin / Hygiene",
    ]
)

# ----------------- TAB 1: Memory Search -----------------

with tab_search:
    st.subheader("Hybrid Memory Search")

    search_col1, search_col2 = st.columns([3, 1])

    with search_col1:
        query = st.text_input(
            "Enter a natural language query",
            value="What do you know about me?",
        )
    with search_col2:
        top_k = st.slider("Top K", min_value=1, max_value=20, value=5, step=1)

    type_options: List[str] = ["profile", "preference", "goal", "episode", "generic"]
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        allowed_types = st.multiselect(
            "Filter by memory type (optional)",
            options=type_options,
            default=[],
        )
    with filter_col2:
        min_importance = st.slider(
            "Minimum importance",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
        )

    if st.button("Run Memory Search", type="primary"):
        if not user_id.strip():
            st.error("Please provide a user_id that exists in the DB.")
        elif not query.strip():
            st.warning("Please enter a query.")
        else:
            st.info("Running hybrid search (semantic + recency + importance)…")

            atypes = allowed_types if allowed_types else None
            results = search_memories(
                user_id=user_id.strip(),
                query=query.strip(),
                k=top_k,
                allowed_types=atypes,
            )

            # Filter on importance
            results = [r for r in results if r["importance"] >= min_importance]

            if not results:
                st.warning("No memories found for this query / filters.")
            else:
                st.success(f"Found {len(results)} memories")

                for i, r in enumerate(results, start=1):
                    with st.expander(
                        f"{i}. [{r['memory_type']}, importance={r['importance']}] • score={r['score']:.3f}"
                    ):
                        st.markdown(f"**Content**: {r['content']}")
                        st.markdown(
                            f"- Similarity: `{r['similarity']:.3f}`  \n"
                            f"- Created at: `{r['created_at']}`"
                        )

                # Table view
                st.markdown("### Tabular view")
                df = pd.DataFrame(results)
                st.dataframe(
                    df[
                        [
                            "memory_type",
                            "importance",
                            "score",
                            "similarity",
                            "created_at",
                            "content",
                        ]
                    ],
                    use_container_width=True,
                )
    else:
        st.info("Enter a query and click **Run Memory Search** to explore how the agent retrieves memories.")


# ----------------- TAB 2: User Profile View -----------------

with tab_profile:
    st.subheader("User Profile / Persona from Memory")

    if st.button("Refresh Profile View"):
        if not user_id.strip():
            st.error("Please provide a user_id.")
        else:
            sections = grouped_profile_view(user_id.strip())

            for section_name, memories in sections.items():
                st.markdown(f"#### {section_name}")
                if not memories:
                    st.caption("_No memories of this type yet._")
                else:
                    for r in memories:
                        st.markdown(
                            f"- ({r['importance']}) {r['content']}  "
                            f" _(score={r['score']:.3f}, {r['created_at']})_"
                        )
    else:
        st.info("Click **Refresh Profile View** to generate a persona-style summary from stored memories.")


# ----------------- TAB 3: Raw Memory Table -----------------

with tab_table:
    st.subheader("Raw Memories from SingleStore")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        table_user_filter = st.text_input(
            "Filter by user_id (optional, blank = all users)",
            value=user_id,
            key="table_user_filter",
        )
    with col_b:
        limit = st.slider("Rows to fetch", min_value=10, max_value=500, value=100, step=10)

    if st.button("Load Table"):
        df_raw = fetch_raw_memories(
            user_id=table_user_filter.strip() or None,
            limit=limit,
        )
        if df_raw.empty:
            st.warning("No rows found for this filter yet.")
        else:
            st.dataframe(df_raw, use_container_width=True, height=500)
    else:
        st.info("Click **Load Table** to view recent memory rows directly from SingleStore.")


# ----------------- TAB 4: Admin / Hygiene -----------------

with tab_admin:
    st.subheader("Memory Hygiene & Admin Tools")

    st.markdown("##### Delete a single memory by ID")
    col1, col2 = st.columns([2, 1])
    with col1:
        mem_id_to_delete = st.text_input("Memory ID to delete", value="")
    with col2:
        if st.button("Delete Memory"):
            if not mem_id_to_delete.strip().isdigit():
                st.error("Please enter a numeric memory ID.")
            else:
                deleted = delete_memory_by_id(int(mem_id_to_delete.strip()))
                if deleted:
                    st.success(f"Deleted {deleted} row(s) with id={mem_id_to_delete}.")
                else:
                    st.warning(f"No memory found with id={mem_id_to_delete}.")

    st.markdown("---")
    st.markdown("##### Wipe all memories for this user")

    st.warning(
        "⚠️ This will permanently delete all memories for the selected user_id "
        "from SingleStore."
    )
    wipe_col1, wipe_col2 = st.columns([2, 1])
    with wipe_col1:
        user_to_wipe = st.text_input(
            "User ID to wipe",
            value=user_id,
            key="user_to_wipe",
        )
    with wipe_col2:
        if st.button("Wipe User Memories", type="secondary"):
            if not user_to_wipe.strip():
                st.error("Please enter a user_id.")
            else:
                deleted = delete_memories_for_user(user_to_wipe.strip())
                st.success(f"Deleted {deleted} memories for user_id={user_to_wipe}.")