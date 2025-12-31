# main.py
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph import build_graph
from db import init_db

load_dotenv()  # loads .env variables


def main():
    # Ensure DB + memory table exist
    init_db()

    # Build graph
    graph = build_graph()

    print("🚀 Agentic Memory Mentor (LangGraph + SingleStore)")
    print("Type 'exit' to quit.\n")

    # IMPORTANT: this must match what's stored in SingleStore (your screenshot shows "user")
    user_id = "user"  # In a real app this comes from auth/session

    # Initial empty state
    state = {
        "user_id": user_id,
        "messages": [],
        "retrieved_memories": [],
    }

    while True:
        user_msg = input("You: ")
        if user_msg.lower() in ["exit", "quit"]:
            break

        # Add the new user message
        state["messages"].append(HumanMessage(content=user_msg))

        # Run one full turn (retrieve → agent → tools → agent → end)
        result_state = graph.invoke(state)

        # Update state
        state = result_state

        # Fetch the most recent assistant message
        assistant_messages = [
            msg for msg in state["messages"] if msg.__class__.__name__ == "AIMessage"
        ]
        if assistant_messages:
            reply = assistant_messages[-1].content
            print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()