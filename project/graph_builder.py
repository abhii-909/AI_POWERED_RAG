"""
Knowledge graph construction pipeline.

Handles PDF ingestion, text splitting, entity/relationship extraction,
and persistence to Neo4j Aura.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import AppConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class GraphStatistics:
    """Summary statistics for the Neo4j knowledge graph."""

    node_count: int
    relationship_count: int
    node_labels: list[str]
    relationship_types: list[str]
    sample_entities: list[dict[str, Any]]


@dataclass
class BuildResult:
    """Result of a graph build operation."""

    chunk_count: int
    graph_document_count: int
    statistics: GraphStatistics


class GraphBuilder:
    """
    Builds and manages a knowledge graph from PDF documents.

    Uses dynamic entity discovery via LLMGraphTransformer without
    hardcoded entity types or relationship schemas.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._llm = ChatGroq(
            api_key=self.config.groq.api_key,
            model=self.config.groq.model,
            temperature=self.config.groq.temperature,
        )
        self._graph: Neo4jGraph | None = None

    @property
    def graph(self) -> Neo4jGraph:
        """Lazy-initialize and return the Neo4j graph connection."""
        if self._graph is None:
            self._graph = Neo4jGraph(
                url=self.config.neo4j.uri,
                username=self.config.neo4j.username,
                password=self.config.neo4j.password,
            )
            self._graph.refresh_schema()
        return self._graph

    def load_pdf(self, file_path: str | Path) -> list[Document]:
        """
        Load a PDF and return LangChain documents with metadata.

        Args:
            file_path: Path to the uploaded PDF.

        Returns:
            List of documents extracted from the PDF.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        loader = PyPDFLoader(str(path))
        documents = loader.load()

        # Enrich metadata for traceability in the graph.
        for index, document in enumerate(documents):
            document.metadata.setdefault("source_file", path.name)
            document.metadata.setdefault("page_number", index + 1)

        logger.info("Loaded %s pages from %s", len(documents), path.name)
        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into smaller chunks while preserving metadata.

        Args:
            documents: Source documents from PDF loading.

        Returns:
            Chunked documents ready for graph extraction.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.text_splitter.chunk_size,
            chunk_overlap=self.config.text_splitter.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info("Created %s text chunks", len(chunks))
        return chunks

    def clear_graph(self) -> None:
        """Remove all nodes and relationships from the Neo4j database."""
        self.graph.query("MATCH (n) DETACH DELETE n")
        self.graph.refresh_schema()
        logger.info("Cleared Neo4j graph database")

    def extract_graph_documents(
        self,
        chunks: list[Document],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[Any]:
        """
        Extract entities and relationships from document chunks.

        Dynamic entity discovery is enabled by omitting allowed node and
        relationship constraints on LLMGraphTransformer.

        Args:
            chunks: Text chunks to process.
            progress_callback: Optional callback receiving (progress, message).

        Returns:
            Graph documents produced by the LLM transformer.
        """
        # No allowed_nodes / allowed_relationships => dynamic discovery.
        transformer = LLMGraphTransformer(
            llm=self._llm,
            ignore_tool_usage=True,
            strict_mode=False,
        )

        graph_documents: list[Any] = []
        total = len(chunks)

        for index, chunk in enumerate(chunks, start=1):
            if progress_callback:
                progress_callback(
                    index / total,
                    f"Extracting entities from chunk {index}/{total}...",
                )

            try:
                batch_docs = transformer.convert_to_graph_documents([chunk])
                graph_documents.extend(batch_docs)
            except Exception as exc:
                logger.warning(
                    "Failed to extract graph from chunk %s: %s", index, exc
                )

        logger.info("Extracted %s graph documents", len(graph_documents))
        return graph_documents

    def store_graph_documents(self, graph_documents: list[Any]) -> None:
        """
        Persist graph documents to Neo4j.

        Args:
            graph_documents: Structured graph documents from the transformer.
        """
        if not graph_documents:
            raise ValueError("No graph documents available to store.")

        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True,
        )
        self.graph.refresh_schema()
        logger.info("Stored graph documents in Neo4j")

    def get_statistics(self) -> GraphStatistics:
        """
        Retrieve graph statistics and sample entities from Neo4j.

        Returns:
            GraphStatistics with counts and representative nodes.
        """
        node_count_result = self.graph.query(
            "MATCH (n) RETURN count(n) AS count"
        )
        rel_count_result = self.graph.query(
            "MATCH ()-[r]->() RETURN count(r) AS count"
        )

        node_labels_result = self.graph.query(
            """
            CALL db.labels() YIELD label
            RETURN label
            ORDER BY label
            """
        )
        rel_types_result = self.graph.query(
            """
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN relationshipType
            ORDER BY relationshipType
            """
        )
        sample_entities_result = self.graph.query(
            """
            MATCH (n)
            WHERE NOT n:Document
            RETURN labels(n) AS labels, properties(n) AS properties
            LIMIT 10
            """
        )

        return GraphStatistics(
            node_count=node_count_result[0]["count"] if node_count_result else 0,
            relationship_count=rel_count_result[0]["count"] if rel_count_result else 0,
            node_labels=[row["label"] for row in node_labels_result],
            relationship_types=[
                row["relationshipType"] for row in rel_types_result
            ],
            sample_entities=[
                {
                    "labels": row["labels"],
                    "properties": row["properties"],
                }
                for row in sample_entities_result
            ],
        )

    def build_from_pdf(
        self,
        file_path: str | Path,
        clear_before_ingest: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> BuildResult:
        """
        End-to-end pipeline: load PDF, split, extract, and store graph.

        Args:
            file_path: Path to the PDF file.
            clear_before_ingest: Whether to wipe the graph before ingestion.
            progress_callback: Optional progress callback for UI updates.

        Returns:
            BuildResult with counts and graph statistics.
        """
        if progress_callback:
            progress_callback(0.05, "Loading PDF...")

        documents = self.load_pdf(file_path)

        if progress_callback:
            progress_callback(0.15, "Splitting documents into chunks...")

        chunks = self.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks were produced from the PDF.")

        if clear_before_ingest:
            if progress_callback:
                progress_callback(0.2, "Clearing existing graph...")
            self.clear_graph()

        if progress_callback:
            progress_callback(0.25, "Extracting entities and relationships...")

        graph_documents = self.extract_graph_documents(
            chunks,
            progress_callback=lambda value, message: progress_callback(
                0.25 + (value * 0.55),
                message,
            )
            if progress_callback
            else None,
        )

        if not graph_documents:
            raise ValueError(
                "Entity extraction produced no graph data. "
                "Try a different document or verify your Groq API key."
            )

        if progress_callback:
            progress_callback(0.85, "Storing graph in Neo4j Aura...")

        self.store_graph_documents(graph_documents)

        if progress_callback:
            progress_callback(0.95, "Collecting graph statistics...")

        statistics = self.get_statistics()

        if progress_callback:
            progress_callback(1.0, "Graph build complete.")

        return BuildResult(
            chunk_count=len(chunks),
            graph_document_count=len(graph_documents),
            statistics=statistics,
        )
