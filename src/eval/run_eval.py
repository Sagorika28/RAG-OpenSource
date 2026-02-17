"""
src/eval/run_eval.py — Evaluation runner (v2: reusable pipeline support).

CLI entry point that:
  1. Loads config + eval dataset
  2. Runs each question through the QueryPipeline
  3. Computes retrieval metrics (Recall@k, MRR@k, nDCG@k)
  4. Aggregates latency stats
  5. Outputs a JSON report
"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from src.core.config import load_config
from src.core.utils import setup_logging
from src.eval.datasets import load_eval_dataset
from src.eval.llm_judge import LLMJudge
from src.eval.metrics import aggregate_latencies, compute_retrieval_metrics
from src.pipeline.query import QueryPipeline

logger = logging.getLogger(__name__)


def run_evaluation(
    config: Dict[str, Any],
    eval_path: str | Path | None = None,
    output_path: str | Path | None = None,
    k_values: list[int] | None = None,
    pipeline: "QueryPipeline | None" = None,
    max_workers: int | None = None,
) -> Dict[str, Any]:
    """
    Run the full evaluation.

    Args:
        config:      Loaded config dict.
        eval_path:   Path to questions.jsonl (falls back to config).
        output_path: Where to write the JSON report (optional).
        k_values:    k values for retrieval metrics.
        pipeline:    Optional existing QueryPipeline to reuse (avoids Qdrant lock conflicts).
        max_workers: Number of worker threads. Defaults to config["eval"]["max_workers"]
                     or 1 for safe single-thread execution.

    Returns:
        Evaluation report as a dict.
    """
    eval_path = Path(
        eval_path
        or config.get("data", {}).get("eval_file", "data/eval/questions.jsonl")
    )
    k_values = k_values or [1, 3, 5, 10]

    # Load eval dataset
    examples = load_eval_dataset(eval_path)
    if not examples:
        logger.error("No evaluation examples loaded — aborting")
        return {"error": "empty dataset"}

    # Initialize query pipeline (reuse if provided)
    if pipeline is None:
        pipeline = QueryPipeline(config)

    # Run each example
    all_retrieved: list[list[str]] = []
    all_gold: list[list[str]] = []
    all_timings: list[dict[str, float]] = []
    per_query_results: list[dict] = []
    judge_inputs: list[dict] = []

    per_query_results: list[dict] = [None] * len(examples)  # type: ignore
    judge_inputs: list[dict] = []

    # Determine if generation should be enabled for LLM judge
    gen_cfg = config.get("generation", {})
    gen_enabled = gen_cfg.get("enabled", False)

    def _process_example(idx: int, example: Any):
        logger.info(f"Eval [{idx+1}/{len(examples)}]: {example.question[:80]}...")
        
        # Add a tiny staggered start to avoid hitting rate limits instantly
        time.sleep(idx * 0.1)

        result = pipeline.run(
            query=example.question,
            skip_generation=not gen_enabled,
        )

        # Extract retrieved sources
        retrieved_sources = [
            hit.chunk.metadata.get("source", "")
            for hit in result.hits
        ]
        
        # Return package for aggregation
        return {
            "idx": idx,
            "retrieved_sources": retrieved_sources,
            "gold_sources": example.gold_sources,
            "timings": result.timings,
            "answer": result.answer,
            "question": example.question,
            "hits": result.hits,
        }

    if max_workers is None:
        max_workers = int(config.get("eval", {}).get("max_workers", 1))
    max_workers = max(1, int(max_workers))

    if max_workers == 1:
        logger.info("Running evaluation in single-thread mode")
        for i, ex in enumerate(examples):
            try:
                res = _process_example(i, ex)
                idx = res["idx"]
                per_query_results[idx] = {
                    "question": res["question"],
                    "gold_sources": res["gold_sources"],
                    "retrieved_sources": res["retrieved_sources"][:10],
                    "answer": res["answer"] if gen_enabled else "(skipped)",
                    "timings": res["timings"],
                }
                if gen_enabled and res["answer"]:
                    judge_inputs.append({
                        "question": res["question"],
                        "hits": res["hits"],
                        "answer": res["answer"],
                        "idx": idx,
                    })
            except Exception as e:
                logger.error(f"Eval failed for example {i+1}: {e}")
    else:
        logger.info(f"Running evaluation in parallel mode (workers={max_workers})")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_process_example, i, ex)
                for i, ex in enumerate(examples)
            ]

            for future in futures:
                try:
                    res = future.result()
                    idx = res["idx"]

                    # Collect results in order
                    per_query_results[idx] = {
                        "question": res["question"],
                        "gold_sources": res["gold_sources"],
                        "retrieved_sources": res["retrieved_sources"][:10],
                        "answer": res["answer"] if gen_enabled else "(skipped)",
                        "timings": res["timings"],
                    }

                    if gen_enabled and res["answer"]:
                        judge_inputs.append({
                            "question": res["question"],
                            "hits": res["hits"],
                            "answer": res["answer"],
                            "idx": idx,
                        })
                except Exception as e:
                    logger.error(f"Parallel eval failed for an example: {e}")

    # Re-sort judge_inputs to maintain original order for report consistency
    judge_inputs.sort(key=lambda x: x["idx"])

    # Flatten for metric computation
    all_retrieved = [r["retrieved_sources"] for r in per_query_results if r]
    all_gold = [r["gold_sources"] for r in per_query_results if r]
    all_timings = [r["timings"] for r in per_query_results if r]

    # Compute metrics
    retrieval_metrics = compute_retrieval_metrics(
        all_retrieved, all_gold, k_values
    )
    latency_summary = aggregate_latencies(all_timings)

    # Index stats
    try:
        store = pipeline._get_store()
        index_stats = store.get_stats()
    except Exception:
        index_stats = {}

    # Run LLM-as-a-Judge
    generation_metrics = {}
    if gen_enabled and judge_inputs:
        logger.info(f"Running LLM Judge on {len(judge_inputs)} answers...")
        judge = LLMJudge(config.get("generation", {}))
        generation_metrics = judge.evaluate_batch(judge_inputs)
        logger.info(f"Judge scores: {generation_metrics}")

    # Build report
    report = {
        "config_file": str(eval_path),
        "num_queries": len(examples),
        "k_values": k_values,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "latency_summary": latency_summary,
        "index_stats": index_stats,
        "per_query_results": per_query_results,
    }

    # Write to file
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Eval report written to: {output_path}")

    return report


# --------------------------------------------------------------------- #
#  CLI entry point                                                       #
# --------------------------------------------------------------------- #

def main():
    """Run evaluation from the command line."""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run RAG-OS evaluation harness"
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/cpu.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--eval-file", "-e",
        default=None,
        help="Path to evaluation JSONL (overrides config)",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/eval_report.json",
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--k", "-k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="k values for retrieval metrics",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of eval workers (default: config.eval.max_workers or 1)",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    report = run_evaluation(
        config=config,
        eval_path=args.eval_file,
        output_path=args.output,
        k_values=args.k,
        max_workers=args.workers,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RAG-OS EVALUATION REPORT")
    print("=" * 60)
    print(f"Queries evaluated: {report['num_queries']}")

    print("\n--- Retrieval Metrics ---")
    for key, value in report.get("retrieval_metrics", {}).items():
        print(f"  {key}: {value}")

    if report.get("generation_metrics"):
        print("\n--- Generation Quality (LLM Judge) ---")
        for key, value in report["generation_metrics"].items():
            print(f"  {key}: {value}/5")

    print("\n--- Latency Summary ---")
    for key, stats in report.get("latency_summary", {}).items():
        print(f"  {key}: mean={stats['mean']:.1f}ms, p50={stats['p50']:.1f}ms")

    print("\n--- Index Stats ---")
    for key, value in report.get("index_stats", {}).items():
        print(f"  {key}: {value}")

    if args.output:
        print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
