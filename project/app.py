"""
Streamlit frontend for the Knowledge Graph GraphRAG application.

Provides PDF upload, graph construction, visualization, and Q&A.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from config import validate_config
from graph_builder import GraphBuilder, GraphStatistics
from graph_query import GraphQueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Knowledge Graph GraphRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }
    .status-success {
        color: #059669;
        font-weight: 600;
    }
    .status-error {
        color: #dc2626;
        font-weight: 600;
    }
    .entity-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Initialize Streamlit session state defaults."""
    defaults = {
        "graph_built": False,
        "build_stats": None,
        "last_answer": None,
        "last_cypher": None,
        "last_context": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_statistics(stats: GraphStatistics) -> None:
    """Render graph statistics and sample entities."""
    st.subheader("Graph Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nodes", stats.node_count)
    with col2:
        st.metric("Relationships", stats.relationship_count)
    with col3:
        st.metric("Entity Labels", len(stats.node_labels))

    label_col, rel_col = st.columns(2)

    with label_col:
        st.markdown("**Node Labels**")
        if stats.node_labels:
            st.write(", ".join(stats.node_labels))
        else:
            st.info("No node labels found.")

    with rel_col:
        st.markdown("**Relationship Types**")
        if stats.relationship_types:
            st.write(", ".join(stats.relationship_types))
        else:
            st.info("No relationship types found.")

    st.markdown("**Sample Graph Entities**")
    if stats.sample_entities:
        for entity in stats.sample_entities:
            labels = ", ".join(entity.get("labels", []))
            properties = entity.get("properties", {})
            display_name = (
                properties.get("id")
                or properties.get("name")
                or properties.get("title")
                or "Unnamed Entity"
            )
            st.markdown(
                f'<div class="entity-card"><strong>{display_name}</strong>'
                f"<br><small>Labels: {labels}</small></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No sample entities available yet.")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Streamlit application."""
    _init_session_state()

    st.markdown(
        '<p class="main-header">Knowledge Graph GraphRAG System</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Extract entities from PDFs, store them in Neo4j Aura, '
        "and ask questions with Graph Retrieval-Augmented Generation.</p>",
        unsafe_allow_html=True,
    )

    # Sidebar configuration status
    with st.sidebar:
        st.header("Configuration")

        is_valid, message = validate_config()
        if is_valid:
            st.success(message)
        else:
            st.error(message)

        clear_graph = st.checkbox(
            "Clear existing graph before ingest",
            value=True,
            help="Remove prior nodes and relationships before storing the uploaded PDF.",
        )

        st.write("---")
        st.markdown("**Runtime status**")
        if st.session_state.graph_built:
            st.write("✅ Graph has been built.")
        else:
            st.write("⚠️ No graph built yet.")

    tab_upload, tab_query, tab_stats = st.tabs([
        "Upload & Build",
        "GraphRAG Q&A",
        "Graph Overview",
    ])

    st.divider()

    st.markdown("**Stack**")
    st.caption("Streamlit · LangChain · Groq · Neo4j Aura")

    # ---------------------------
    # SAMPLE QUESTIONS
    # ---------------------------

    st.divider()

    st.subheader("Sample Questions")

    sample_questions = [
        "Who is the CEO of TechNova?",
        "Who works at TechNova?",
        "What technologies does SmartAnalytics use?",
        "Which company partnered with TechNova?",
        "Who attended the AI Innovation Summit?"
    ]

    for q in sample_questions:
        if st.button(q):
            st.session_state["sample_question"] = q



    # -----------------------------------------------------------------------
    # Tab 1: Upload and build graph
    # -----------------------------------------------------------------------
    with tab_upload:
        st.subheader("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help="The system will extract text, build a knowledge graph, "
            "and store it in Neo4j Aura.",
        )

        if uploaded_file is not None:
            st.success(f"Uploaded: **{uploaded_file.name}**")

            if st.button("Build Knowledge Graph", type="primary", disabled=not is_valid):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(value: float, message: str) -> None:
                    progress_bar.progress(min(max(value, 0.0), 1.0))
                    status_text.info(message)

                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        pdf_path = Path(temp_dir) / uploaded_file.name
                        pdf_path.write_bytes(uploaded_file.getbuffer())

                        builder = GraphBuilder()
                        result = builder.build_from_pdf(
                            pdf_path,
                            clear_before_ingest=clear_graph,
                            progress_callback=update_progress,
                        )

                    st.session_state.graph_built = True
                    st.session_state.build_stats = result.statistics
                    status_text.success("Knowledge graph built successfully.")
                    st.balloons()

                    st.markdown("### Build Summary")
                    summary_col1, summary_col2 = st.columns(2)
                    with summary_col1:
                        st.metric("Text Chunks Processed", result.chunk_count)
                    with summary_col2:
                        st.metric("Graph Documents Created", result.graph_document_count)

                except Exception as exc:
                    logger.exception("Graph build failed")
                    status_text.error(f"Build failed: {exc}")
                    st.session_state.graph_built = False

    # -----------------------------------------------------------------------
    # Tab 2: GraphRAG Q&A
    # -----------------------------------------------------------------------
    with tab_query:
        st.subheader("GraphRAG Question Answering")

        if not st.session_state.graph_built:
            st.info("Build a knowledge graph first in the **Upload & Build** tab.")

        question = st.text_area(
            "Ask a question about your document",
            value=st.session_state.get("sample_question", ""),
            placeholder="Example: What are the main entities and how are they related?",
            height=100,
        )

        if st.button("Get Answer", type="primary", disabled=not is_valid):
            if not question.strip():
                st.warning("Please enter a question.")
            elif not st.session_state.graph_built:
                st.warning("Please build the graph before asking questions.")
            else:
                with st.spinner("Generating Cypher query and answer..."):
                    try:
                        engine = GraphQueryEngine()
                        engine.refresh_schema()
                        result = engine.ask(question)

                        st.session_state.last_answer = result.answer
                        st.session_state.last_cypher = result.cypher_query
                        st.session_state.last_context = result.graph_context

                    except Exception as exc:
                        logger.exception("Query failed")
                        st.error(f"Query failed: {exc}")

        if st.session_state.last_cypher:
            st.markdown("### Generated Cypher Query")
            st.code(st.session_state.last_cypher, language="cypher")

        if st.session_state.last_context:
            with st.expander("Retrieved Graph Context", expanded=False):
                st.text(st.session_state.last_context)

        if st.session_state.last_answer:
            st.markdown("### Final Answer")
            st.markdown(st.session_state.last_answer)

    # -----------------------------------------------------------------------
    # Tab 3: Graph overview
    # -----------------------------------------------------------------------
    with tab_stats:
        st.subheader("Graph Overview")

        refresh_clicked = st.button("Refresh Statistics", disabled=not is_valid)

        if refresh_clicked or st.session_state.build_stats:
            try:
                if refresh_clicked:
                    builder = GraphBuilder()
                    st.session_state.build_stats = builder.get_statistics()

                if st.session_state.build_stats:
                    _render_statistics(st.session_state.build_stats)
                else:
                    st.info("No graph statistics available. Build a graph first.")

            except Exception as exc:
                logger.exception("Failed to load statistics")
                st.error(f"Could not load graph statistics: {exc}")
        else:
            st.info("Upload a PDF and build the knowledge graph to see statistics.")


if __name__ == "__main__":
    main()
