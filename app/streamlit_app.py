"""
RAG-OS Streamlit App — Minimal demo UI.

Features:
  - Question input box
  - Config selector sidebar (cpu.yaml / gpu.yaml)
  - Retrieved evidence display with expandable chunks
  - Generated answer with source citations
  - Index stats display
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# Add project root to path so imports work when running from app/ dir
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.utils import setup_logging
from src.pipeline.query import QueryPipeline
from src.eval.run_eval import run_evaluation


@st.cache_resource
def get_pipeline(config_path: str) -> QueryPipeline:
    """Cache the query pipeline so models aren't reloaded on every query."""
    cfg = load_config(config_path)
    return QueryPipeline(cfg)

# --------------------------------------------------------------------- #
#  Page config                                                           #
# --------------------------------------------------------------------- #

st.set_page_config(
    page_title="RAG-Open Source - Technical Document Assistant",
    page_icon="🔍",
    layout="wide",
)

# --------------------------------------------------------------------- #
#  Sidebar                                                               #
# --------------------------------------------------------------------- #

with st.sidebar:
    st.title("⚙️ Configuration")

    # Config file selector
    config_dir = PROJECT_ROOT / "configs"
    config_files = sorted(config_dir.glob("*.yaml")) if config_dir.exists() else []
    config_names = [f.name for f in config_files]

    try:
        default_idx = config_names.index("groq.yaml")
    except ValueError:
        default_idx = 0

    selected_config = st.selectbox(
        "Config profile",
        config_names,
        index=default_idx,
        help="Select cpu.yaml for local CPU mode or gpu.yaml for GPU mode",
    )

    # Retrieval settings
    st.markdown("---")
    st.subheader("🔎 Retrieval Settings")
    top_k = st.slider("Top-K results", 1, 20, 5)
    skip_generation = st.checkbox(
        "Retrieval only (no LLM)",
        value=False,
        help="Skip the LLM generation step — useful for eval",
    )

    # Metadata filters
    st.markdown("---")
    st.subheader("🏷️ Filters (optional)")
    filter_doc_type = st.selectbox(
        "Document type",
        ["(all)", "manual", "standard", "newsletter", "booklet", "flyer", "slides"],
        index=0,
    )
    filter_source = st.text_input(
        "Source filename",
        "",
        help="Filter by exact source filename (leave empty for all)",
    )

    # Index stats
    st.markdown("---")
    if st.button("📊 Show Index Stats"):
        try:
            config_path = config_dir / selected_config
            pipe = get_pipeline(str(config_path))
            store = pipe._get_store()
            stats = store.get_stats()
            st.json(stats)
        except Exception as e:
            st.error(f"Could not load index stats: {e}")

    # Evaluation Dashboard
    st.markdown("---")
    st.subheader("🧪 Evaluation")
    if st.button("▶️ Run Full Evaluation", type="primary"):
        config_path = config_dir / selected_config
        cfg = load_config(str(config_path))
        with st.spinner("Running evaluation (retrieval + generation)..."):
            try:
                report = run_evaluation(
                    config=cfg,
                    k_values=[1, 3, 5],
                )
                st.session_state["eval_report"] = report
                st.success("Evaluation complete!")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.exception(e)

# --------------------------------------------------------------------- #
#  Main area                                                             #
# --------------------------------------------------------------------- #

st.title("🔍 RAG-OS — Technical Document Assistant")
st.caption("Ask questions about your technical documents. Answers are grounded in retrieved evidence.")

# --------------------------------------------------------------------- #
#  Query input                                                           #
# --------------------------------------------------------------------- #

query = st.text_input(
    "💬 Ask a question:",
    placeholder="e.g. What are the safety precautions for handling refrigerants?",
)

if query and selected_config:
    config_path = config_dir / selected_config

    with st.spinner("Searching and generating answer..."):
        try:
            # Use cached pipeline (embedding model stays loaded)
            pipeline = get_pipeline(str(config_path))

            result = pipeline.run(
                query=query,
                top_k=top_k,
                doc_type=filter_doc_type if filter_doc_type != "(all)" else None,
                source=filter_source if filter_source else None,
                skip_generation=skip_generation,
            )

            # ---- Answer Section ----
            if result.answer:
                st.markdown("---")
                st.subheader("💡 Answer")
                st.markdown(result.answer)

            # ---- Timings ----
            st.markdown("---")
            timing_cols = st.columns(len(result.timings) or 1)
            for i, (key, val) in enumerate(result.timings.items()):
                with timing_cols[i % len(timing_cols)]:
                    label = key.replace("_ms", "").title()
                    st.metric(label=f"⏱️ {label}", value=f"{val:.0f} ms")

            # ---- Retrieved Evidence ----
            st.markdown("---")
            st.subheader(f"📚 Retrieved Evidence ({len(result.hits)} chunks)")

            for i, hit in enumerate(result.hits, 1):
                source = hit.chunk.metadata.get("source", "unknown")
                section = hit.chunk.metadata.get("section_path", "")
                page_start = hit.chunk.metadata.get("page_start", "?")
                page_end = hit.chunk.metadata.get("page_end", "?")
                doc_type = hit.chunk.metadata.get("doc_type", "")
                score = hit.score
                rerank = hit.rerank_score

                # Build header
                header = f"**[{i}]** {source}"
                if section:
                    header += f" — {section}"
                # header += f" (pp {page_start}-{page_end})"
                if doc_type:
                    header += f" `{doc_type}`"

                score_str = f"Score: {score:.4f}"
                if rerank is not None:
                    score_str += f" | Rerank: {rerank:.4f}"

                with st.expander(f"{header} — {score_str}"):
                    st.markdown(hit.chunk.raw_text or hit.chunk.text)
                    st.caption(
                        f"Chunk ID: `{hit.chunk.chunk_id}` | "
                        f"Doc ID: `{hit.chunk.doc_id}`"
                    )

            # ---- Citations Summary ----
            if result.citations:
                st.markdown("---")
                st.subheader("📋 Citations")
                citations_data = []
                for cit in result.citations:
                    citations_data.append({
                        "Rank": cit["rank"],
                        "Source": cit["source"],
                        "Section": cit["section"],
                        # "Pages": cit["pages"],
                        "Score": cit["score"],
                    })
                st.table(citations_data)

        except FileNotFoundError as e:
            st.error(f"Configuration error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

elif not selected_config:
    st.warning("No config files found. Please add a YAML config to `configs/`.")

# --------------------------------------------------------------------- #
#  Evaluation Dashboard (below query results)                            #
# --------------------------------------------------------------------- #

if "eval_report" in st.session_state:
    report = st.session_state["eval_report"]

    st.markdown("---")
    st.header("🧪 Evaluation Dashboard")

    # ---- Retrieval Metrics ----
    st.subheader("📥 Retrieval Quality")
    ret_metrics = report.get("retrieval_metrics", {})
    if ret_metrics:
        # Group by metric type
        recall_cols = st.columns(3)
        mrr_cols = st.columns(3)
        ndcg_cols = st.columns(3)

        for i, k in enumerate([1, 3, 5]):
            with recall_cols[i]:
                val = ret_metrics.get(f"recall@{k}", 0)
                st.metric(f"Recall@{k}", f"{val:.2%}")
            with mrr_cols[i]:
                val = ret_metrics.get(f"mrr@{k}", 0)
                st.metric(f"MRR@{k}", f"{val:.2%}")
            with ndcg_cols[i]:
                val = ret_metrics.get(f"ndcg@{k}", 0)
                st.metric(f"nDCG@{k}", f"{val:.2%}")
    else:
        st.info("No retrieval metrics available.")

    # ---- Generation Metrics (LLM Judge) ----
    gen_metrics = report.get("generation_metrics", {})
    if gen_metrics:
        st.subheader("🤖 Generation Quality (LLM-as-a-Judge)")
        judge_cols = st.columns(4)
        labels = ["faithfulness", "relevance", "completeness", "overall"]
        icons = ["🎯", "📌", "📊", "⭐"]
        for i, (label, icon) in enumerate(zip(labels, icons)):
            with judge_cols[i]:
                val = gen_metrics.get(label, 0)
                st.metric(
                    f"{icon} {label.title()}",
                    f"{val}/5",
                )

    # ---- Latency Summary ----
    lat_summary = report.get("latency_summary", {})
    if lat_summary:
        st.subheader("⏱️ Latency")
        lat_cols = st.columns(len(lat_summary))
        for i, (key, stats) in enumerate(lat_summary.items()):
            with lat_cols[i % len(lat_cols)]:
                label = key.replace("_ms", "").title()
                st.metric(f"{label} (p50)", f"{stats.get('p50', 0):.1f} ms")

    # ---- Per-Query Breakdown ----
    with st.expander("📋 Per-Query Breakdown"):
        for i, pq in enumerate(report.get("per_query_results", []), 1):
            st.markdown(f"**Q{i}:** {pq['question']}")
            st.caption(f"Gold: {pq['gold_sources']} | Retrieved: {pq['retrieved_sources'][:5]}")
            if pq.get("answer") and pq["answer"] != "(skipped)":
                st.markdown(f"> {pq['answer'][:300]}..." if len(pq.get('answer','')) > 300 else f"> {pq.get('answer','')}")
            st.markdown("---")

    # ---- Download Report ----
    import json as _json
    report_json = _json.dumps(report, indent=2, default=str)
    st.download_button(
        "📥 Download Full Report (JSON)",
        data=report_json,
        file_name="eval_report.json",
        mime="application/json",
    )
