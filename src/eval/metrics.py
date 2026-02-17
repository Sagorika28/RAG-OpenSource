"""
src/eval/metrics.py — Retrieval evaluation metrics.

Computes:
  - Recall@k: fraction of gold items found in top-k results
  - MRR@k:    Mean Reciprocal Rank at k
  - nDCG@k:   normalized Discounted Cumulative Gain at k

Also aggregates latency and index statistics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def recall_at_k(
    retrieved_sources: List[str],
    gold_sources: List[str],
    k: int,
) -> float:
    """
    Recall@k — fraction of gold sources found in top-k retrieved.

    Args:
        retrieved_sources: Source filenames from retrieved chunks (ordered).
        gold_sources: Ground-truth source filenames.
        k: Cutoff rank.

    Returns:
        Recall score in [0.0, 1.0].
    """
    if not gold_sources:
        return 0.0
    top_k = set(retrieved_sources[:k])
    gold = set(gold_sources)
    return len(top_k & gold) / len(gold)


def mrr_at_k(
    retrieved_sources: List[str],
    gold_sources: List[str],
    k: int,
) -> float:
    """
    Mean Reciprocal Rank@k — 1/(rank of first relevant result).

    Args:
        retrieved_sources: Source filenames from retrieved chunks (ordered).
        gold_sources: Ground-truth source filenames.
        k: Cutoff rank.

    Returns:
        Reciprocal rank in [0.0, 1.0].
    """
    gold = set(gold_sources)
    for i, src in enumerate(retrieved_sources[:k]):
        if src in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_sources: List[str],
    gold_sources: List[str],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain@k.

    Uses binary relevance: 1 if source is in gold set, else 0.
    Each gold source is counted at most once (first occurrence).

    Args:
        retrieved_sources: Source filenames (ordered by score).
        gold_sources: Ground-truth source filenames.
        k: Cutoff rank.

    Returns:
        nDCG score in [0.0, 1.0].
    """
    gold = set(gold_sources)

    # DCG for retrieved order — count each gold source only once
    dcg = 0.0
    seen_gold: set = set()
    for i, src in enumerate(retrieved_sources[:k]):
        if src in gold and src not in seen_gold:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0
            seen_gold.add(src)

    # Ideal DCG (all gold items first)
    ideal_rels = [1.0] * min(len(gold), k) + [0.0] * max(0, k - len(gold))
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    all_retrieved: List[List[str]],
    all_gold: List[List[str]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Compute average retrieval metrics over a set of queries.

    Args:
        all_retrieved: List of retrieved source lists, one per query.
        all_gold: List of gold source lists, one per query.
        k_values: List of k values to evaluate at.

    Returns:
        Dict with keys like "recall@5", "mrr@5", "ndcg@5".
    """
    n = len(all_retrieved)
    if n == 0:
        return {}

    metrics: Dict[str, float] = {}
    for k in k_values:
        recall_scores = [
            recall_at_k(ret, gold, k)
            for ret, gold in zip(all_retrieved, all_gold)
        ]
        mrr_scores = [
            mrr_at_k(ret, gold, k)
            for ret, gold in zip(all_retrieved, all_gold)
        ]
        ndcg_scores = [
            ndcg_at_k(ret, gold, k)
            for ret, gold in zip(all_retrieved, all_gold)
        ]

        metrics[f"recall@{k}"] = round(sum(recall_scores) / n, 4)
        metrics[f"mrr@{k}"] = round(sum(mrr_scores) / n, 4)
        metrics[f"ndcg@{k}"] = round(sum(ndcg_scores) / n, 4)

    return metrics


def aggregate_latencies(
    all_timings: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate timing dicts from multiple queries into summary stats.

    Returns:
        Dict with keys like "embed_ms" → {"mean": ..., "p50": ..., "p95": ...}.
    """
    if not all_timings:
        return {}

    # Collect all keys
    keys = set()
    for t in all_timings:
        keys.update(t.keys())

    summary: Dict[str, Dict[str, float]] = {}
    for key in sorted(keys):
        values = [t[key] for t in all_timings if key in t]
        if not values:
            continue
        values.sort()
        n = len(values)
        summary[key] = {
            "mean": round(sum(values) / n, 2),
            "p50": round(values[n // 2], 2),
            "p95": round(values[int(n * 0.95)], 2) if n >= 2 else round(values[-1], 2),
            "min": round(values[0], 2),
            "max": round(values[-1], 2),
        }

    return summary
