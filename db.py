# db.py
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

SINGLESTORE_URL = os.getenv("SINGLESTORE_URL")
# Example:
# SINGLESTORE_URL = "mysql+pymysql://admin:password@host:3306/agentic_memory"

# --- THIS MUST BE AT THE TOP LEVEL ---
engine = create_engine(SINGLESTORE_URL, pool_pre_ping=True)


def init_db():
    """Create database + table if not exists."""
    with engine.connect() as conn:
        conn.execute(text("CREATE DATABASE IF NOT EXISTS agentic_memory"))
        conn.execute(text("USE agentic_memory"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    memory_type VARCHAR(64),
                    importance INT,
                    source VARCHAR(64),
                    session_id VARCHAR(64),
                    content TEXT,
                    embedding JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )
        conn.commit()