"""
Application configuration loaded from environment variables.

Centralizes Groq, Neo4j, and document-processing settings for the GraphRAG system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# Load variables from a local .env file when present.
load_dotenv()


@dataclass(frozen=True)
class GroqConfig:
    """Groq LLM configuration."""

    api_key: str
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0


@dataclass(frozen=True)
class Neo4jConfig:
    """Neo4j Aura connection configuration."""

    uri: str
    username: str
    password: str


@dataclass(frozen=True)
class TextSplitterConfig:
    """Document chunking configuration."""

    chunk_size: int = 300
    chunk_overlap: int = 50


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    groq: GroqConfig
    neo4j: Neo4jConfig
    text_splitter: TextSplitterConfig


def _require_env(name: str) -> str:
    """Return an environment variable or raise a descriptive error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}. "
            f"Set it in your .env file or system environment."
        )
    return value


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Build and cache application configuration.

    Returns:
        AppConfig: Validated configuration object.
    """
    groq = GroqConfig(
        api_key=_require_env("GROQ_API_KEY"),
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.0")),
    )

    neo4j = Neo4jConfig(
        uri=_require_env("NEO4J_URI"),
        username=_require_env("NEO4J_USERNAME"),
        password=_require_env("NEO4J_PASSWORD"),
    )

    text_splitter = TextSplitterConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )

    return AppConfig(groq=groq, neo4j=neo4j, text_splitter=text_splitter)


def validate_config() -> tuple[bool, str]:
    """
    Validate configuration without raising.

    Returns:
        Tuple of (is_valid, message).
    """
    try:
        get_config()
        return True, "Configuration is valid."
    except ValueError as exc:
        return False, str(exc)
