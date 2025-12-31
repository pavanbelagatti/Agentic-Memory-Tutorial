# graph.py
"""
LangGraph workflow for an agentic "Personal Mentor" agent
that uses SingleStore as long-term memory.

Enhancements:
- Memories have metadata: memory_type, importance, source, session_id.
- Hybrid search in memory_store.py (semantic + FULLTEXT + recency + importance).
- write_memory tool lets the model label memories (type + importance).

Make sure you have memory_store.py with:
    - save_memory(user_id, content, memory_type, importance, source, session_id)
    - search_memories(user_id, query, k=5, allowed_types=None)
"""

from __future__ import annotations

import operator
from typing import List

from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from memory_store import save_memory, search_memories


# ---------------------------------------------------------------------------
# 1. Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    State carried through the LangGraph workflow.

    - messages: short-term conversational state (list of LangChain Messages)
    - user_id: identifier used for per-user memory in SingleStore
    - retrieved_memories: list of retrieved memory strings for this turn
    """
    messages: Annotated[List[AnyMessage], operator.add]
    user_id: str
    retrieved_memories: List[str]


# ---------------------------------------------------------------------------
# 2. Model & tools
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
)

SYSTEM_PROMPT = """You are a helpful personal mentor AI.

You have access to long-term memories about the user (preferences, goals, facts).
When you see stable facts about the user that are likely useful later
(e.g., "I work at X", "my goal is Y", "I love Z"), call the `write_memory` tool
with:
- memory_type: 'profile', 'preference', 'goal', or 'episode'
- importance: 1 (low) to 5 (critical)

Use:
- 'profile'    for stable identity facts (name, role, company, background)
- 'preference' for likes/dislikes (tools, tech, style, topics)
- 'goal'       for medium/long-term objectives
- 'episode'    for specific events or one-off experiences

Use higher importance (4–5) for core identity and goals.

Always:
- Use retrieved memories to personalize your answers.
- Be specific and practical.
- Do not store short-lived or trivial details (like temporary moods or jokes).
"""


@tool
def write_memory(
    user_id: str,
    content: str,
    memory_type: str = "generic",
    importance: int = 3,
) -> str:
    """
    Persist long-term user memory, like preferences, background, or goals.

    memory_type:
      - 'profile'    -> stable identity facts (e.g. name, role, company)
      - 'preference' -> likes/dislikes (tools, tech, style)
      - 'goal'       -> long-term objectives
      - 'episode'    -> specific events or conversations
      - 'generic'    -> anything else

    importance:
      - 1 (low) .. 5 (very important)
    """
    # We keep source="chat" and session_id=None for now; can be extended later.
    save_memory(
        user_id=user_id,
        content=content,
        memory_type=memory_type,
        importance=importance,
        source="chat",
        session_id=None,
    )
    return "Memory saved."


tools = [write_memory]
tools_by_name = {t.name: t for t in tools}

# LLM with tool-calling capabilities bound
model_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------------------------

def _get_last_user_message_content(messages: List[AnyMessage]) -> str | None:
    """Return the content of the most recent HumanMessage, if any."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


# ---------------------------------------------------------------------------
# 4. Nodes
# ---------------------------------------------------------------------------

def retrieve_memories(state: AgentState) -> dict:
    """
    Node 1: Use the latest user message to retrieve semantic memories
    from SingleStore and attach them to state.retrieved_memories.

    This calls `search_memories` which performs:
    - semantic similarity (cosine)
    - FULLTEXT keyword relevance
    - recency & importance weighting (hybrid ranking)
    """
    query = _get_last_user_message_content(state["messages"])
    if not query:
        return {"retrieved_memories": []}

    # You can optionally pass allowed_types, e.g. ['profile', 'preference', 'goal']
    # For now we keep all types.
    results = search_memories(
        state["user_id"],
        query=query,
        k=5,
        allowed_types=None,
    )
    memories = [r["content"] for r in results]

    return {"retrieved_memories": memories}


def call_model(state: AgentState, config: RunnableConfig) -> dict:
    """
    Node 2: Call the LLM with:
    - System prompt (including retrieved memories)
    - All previous messages in this thread

    The model can optionally decide to call tools (e.g., write_memory).
    """
    memories = state.get("retrieved_memories") or []
    memories_text = "\n".join(f"- {m}" for m in memories) or "(none yet)"

    system_content = (
        SYSTEM_PROMPT
        + "\n\nKnown long-term memories about this user:\n"
        + memories_text
    )
    system_msg = SystemMessage(content=system_content)

    # Prepend system message to the conversation
    model_input: List[AnyMessage] = [system_msg] + state["messages"]

    response: AIMessage = model_with_tools.invoke(model_input, config)

    # Thanks to Annotated[..., operator.add]:
    # returning {"messages": [response]} appends this AIMessage to the state.
    return {"messages": [response]}


def call_tools(state: AgentState) -> dict:
    """
    Node 3: Execute any tools requested by the last AI message.

    For each tool_call:
      - Execute the corresponding function (e.g. write_memory)
      - Construct a ToolMessage with the result
      - Return ToolMessages so the model can see them next turn
    """
    last_message = state["messages"][-1]

    tool_calls = getattr(last_message, "tool_calls", None) or []
    if not tool_calls:
        return {}

    tool_messages: List[ToolMessage] = []

    for tc in tool_calls:
        name = tc["name"]
        args = tc["args"]
        tool_fn = tools_by_name.get(name)

        if tool_fn is None:
            # Unknown tool name; skip gracefully
            continue

        # Execute the tool
        result = tool_fn.invoke(args)

        # Add a ToolMessage so the model can see tool output if needed
        tool_messages.append(
            ToolMessage(
                content=result,
                name=name,
                tool_call_id=tc["id"],
            )
        )

    # These ToolMessages are appended to state["messages"]
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """
    Conditional edge after the agent node:

    - If the last AIMessage has tool_calls -> go to 'tools'
    - Otherwise -> end the graph for this turn
    """
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# 5. Build & compile graph
# ---------------------------------------------------------------------------

def build_graph():
    """
    Build and compile the LangGraph workflow.

    Flow per turn:
        entry: retrieve_memories
            -> agent (call_model)
            -> if tool_calls: tools (call_tools) -> agent (loop)
               else: END
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("retrieve_memories", retrieve_memories)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    # Entry point
    workflow.set_entry_point("retrieve_memories")

    # retrieve_memories -> agent
    workflow.add_edge("retrieve_memories", "agent")

    # agent -> tools OR end
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # tools -> agent (loop until no more tool calls)
    workflow.add_edge("tools", "agent")

    # Compile into a runnable app
    app = workflow.compile()
    return app