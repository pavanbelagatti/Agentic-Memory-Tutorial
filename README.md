## Agentic Memory with SingleStore
This repository demonstrates how to build a real Agentic Memory system using LangGraph and SingleStore.
Instead of a stateless chatbot, this project shows how to create an AI agent that can store, retrieve, and evolve long-term memory across conversations.

The agent acts like a Personal Mentor that remembers user profiles, goals, and preferences using a durable database-backed memory layer.

## What This Project Shows
- What an AI Agent is and how it differs from a chatbot
- How agentic applications maintain long-term context
- How to design persistent memory for agents
- How SingleStore acts as the long-term memory backend
- How to retrieve memories using semantic similarity + recency + importance
- How to visualize agent memory with a Streamlit dashboard

## Architecture Overview
High-level flow:
- User talks to the agent (CLI or UI)
- Agent retrieves relevant memories from SingleStore
- LLM reasons using retrieved memory + conversation
- Agent decides whether to store new memories
- Important facts are written back to SingleStore
- Memory continuously evolves over time

## Prerequisites
- Python 3.9+
- A SingleStore account (Free tier works)
- OpenAI API key

### Setup Instructions

#### Clone the repository

```
git clone https://github.com/pavanbelagatti/Agentic-Memory-Tutorial.git
```
```
cd Agentic-Memory-Tutorial
```

#### Create and activate a virtual environment
```
python -m venv venv
```
```
source venv/bin/activate 
```

#### Install dependencies
```
pip install -r requirements.txt
```

#### Configure environment variables
Create a .env file (based on .env.example):

```
cp .env.example .env
```

Edit .env and add:
```
OPENAI_API_KEY=your_openai_api_key
SINGLESTORE_URL=mysql+pymysql://USER:PASSWORD@HOST:3306/agentic_memory
```

#### Running the Agent
```
python main.py
```

Example interaction:
You: My name is ABC and I'm a DEF.
You: I love using SingleStore for AI applications.
You: What do you remember about me?

The agent will store important facts and recall them across turns.

#### Running the Memory Dashboard (Streamlit)
Launch the dashboard:

```
python -m streamlit run streamlit_app.py
```

The dashboard lets you:
- Search memories using natural language
- Filter by memory type and importance
- View raw memory rows from SingleStore
- Delete or reset memories

Memory retrieval uses a hybrid scoring strategy:
- Semantic similarity (embeddings)
- Recency (newer memories matter more)
- Importance (critical memories get boosted)

### Enter Hindsight by Vectorize
[Hindsight](https://vectorize.io/hindsight) is the first agent memory system with a mechanism for agent learning. 
Agents can reflect on their experiences, form opinions, and avoid repeating mistakes.

Hindsight is a memory system for AI agents that’s designed to work more like human memory: 
it doesn’t just “store chunks and retrieve top-k,” it separates what’s true from what happened from what the agent believes, and uses that structure to help an agent stay consistent across sessions.

[Try Hindsight!](https://github.com/vectorize-io/hindsight)

This repository is part of a detailed tutorial and YouTube walkthrough on Agentic Memory with SingleStore.

If you found this useful, ⭐ star the repo and feel free to fork and extend it.



