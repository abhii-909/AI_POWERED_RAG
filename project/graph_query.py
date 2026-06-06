"""
GraphRAG query layer.

Converts natural language questions into Cypher queries, retrieves graph
context from Neo4j, and generates grounded answers with Groq.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_groq import ChatGroq
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_core.prompts import PromptTemplate

from config import AppConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured response from a GraphRAG query."""

    question: str
    cypher_query: str
    graph_context: str
    answer: str


class GraphQueryEngine:
    """
    GraphRAG engine for question answering over the Neo4j knowledge graph.

    Returns both the generated Cypher query and the final natural-language
    answer for transparency and debugging.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._llm = ChatGroq(
            api_key=self.config.groq.api_key,
            model=self.config.groq.model,
            temperature=self.config.groq.temperature,
        )
        self._graph: Neo4jGraph | None = None
        self._chain: GraphCypherQAChain | None = None

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

    @property
    def chain(self) -> GraphCypherQAChain:
        if self._chain is None:
            cypher_prompt = PromptTemplate(
                input_variables=["schema", "question"],
                template="""
You are an expert Neo4j Cypher developer.

Use ONLY the schema below.

Schema:
{schema}

Rules:
- Return ONLY Cypher.
- Use labels exactly as shown in schema.
- Use relationships exactly as shown in schema.
- Use node properties exactly as shown in schema.
- Do not invent labels.
- Do not invent relationships.
- Do not explain anything.

Question:
{question}

Cypher:
"""
            )

            self._chain = GraphCypherQAChain.from_llm(
                llm=self._llm,
                graph=self.graph,
                cypher_prompt=cypher_prompt,
                validate_cypher=True,
                return_intermediate_steps=True,
                verbose=True,
                allow_dangerous_requests=True,
            )

        return self._chain

    def refresh_schema(self) -> None:
        self.graph.refresh_schema()
        self._chain = None


    @staticmethod
    def _extract_cypher(intermediate_steps: list[Any]) -> str:
        """Extract the generated Cypher query from chain intermediate steps."""
        for step in intermediate_steps:
            if isinstance(step, dict):
                if "query" in step and step["query"]:
                    return str(step["query"]).strip()
                if "cypher" in step and step["cypher"]:
                    return str(step["cypher"]).strip()

            if isinstance(step, tuple) and len(step) >= 1:
                candidate = step[0]
                if isinstance(candidate, str) and "MATCH" in candidate.upper():
                    return candidate.strip()

            if isinstance(step, str) and "MATCH" in step.upper():
                return step.strip()

        return "Cypher query not captured in intermediate steps."

    @staticmethod
    def _format_context_value(context_value: Any) -> str:
        """Format graph context for display in the UI."""
        if isinstance(context_value, list):
            if not context_value:
                return "No graph context returned."
            return "\n".join(str(row) for row in context_value)
        return str(context_value) if context_value else "No graph context returned."

    @staticmethod
    def _extract_context(intermediate_steps: list[Any]) -> str:
        """Extract retrieved graph context from intermediate steps."""
        for step in intermediate_steps:
            if isinstance(step, dict) and "context" in step:
                return GraphQueryEngine._format_context_value(step["context"])

            if isinstance(step, tuple) and len(step) >= 2:
                return GraphQueryEngine._format_context_value(step[1])

        return "No graph context returned."

    @staticmethod
    def _fallback_cypher_from_logs(text: str) -> str:
        """Best-effort Cypher extraction from verbose chain output."""
        match = re.search(
            r"(MATCH[\s\S]*?)(?:\n\n|\Z)",
            text,
            flags=re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    def ask(self, question: str) -> QueryResult:
        """
        Answer a natural language question using GraphRAG.

        Args:
            question: User question about the knowledge graph.

        Returns:
            QueryResult containing Cypher, context, and final answer.
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        try:
            response = self.chain({"question": cleaned_question})
        except Exception as exc:
            logger.exception("GraphRAG query failed")
            raise RuntimeError(f"Failed to process question: {exc}") from exc

        if isinstance(response, dict):
            answer = str(
                response.get("result")
                or response.get("output")
                or response.get("answer")
                or ""
            ).strip()
            intermediate_steps = response.get("intermediate_steps", [])
        else:
            answer = str(response).strip()
            intermediate_steps = []

        cypher_query = self._extract_cypher(intermediate_steps)
        graph_context = self._extract_context(intermediate_steps)

        if cypher_query.startswith("Cypher query not captured"):
            fallback = self._fallback_cypher_from_logs(str(response))
            if fallback:
                cypher_query = fallback

        return QueryResult(
            question=cleaned_question,
            cypher_query=cypher_query,
            graph_context=graph_context,
            answer=answer or "No answer could be generated from the graph.",
        )

    def run_cypher(self, cypher_query: str) -> list[dict[str, Any]]:
        """
        Execute a raw Cypher query against Neo4j.

        Args:
            cypher_query: Cypher statement to execute.

        Returns:
            Query result rows as dictionaries.
        """
        return self.graph.query(cypher_query)
