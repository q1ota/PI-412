"""Full-catalog chronological evaluation; no sampled negatives or test fitting."""

from datetime import datetime, timezone
import platform
from time import perf_counter

import numpy as np

from recsys.data import ARCHIVE_SHA256, DATA_URL, Dataset, chronological_split
from recsys.models import METHODS, Recommender


def ranking_metrics(recommended: list[int], relevant: set[int], k: int) -> dict:
    if k < 1 or not relevant:
        raise ValueError("Metrics require k > 0 and a nonempty relevant set.")
    if len(set(recommended)) != len(recommended):
        raise ValueError("Recommendation list contains duplicates.")
    hits = np.array([mid in relevant for mid in recommended[:k]], dtype=float)
    discounts = 1 / np.log2(np.arange(2, len(hits) + 2))
    ideal = (1 / np.log2(np.arange(2, min(k, len(relevant)) + 2))).sum()
    return {"precision": float(hits.sum() / k),
            "recall": float(hits.sum() / len(relevant)),
            "ndcg": float((hits * discounts).sum() / ideal),
            "hit_rate": float(hits.any())}


def run_experiment(dataset: Dataset, k: int = 10) -> dict:
    if not 1 <= k <= 50:
        raise ValueError("k must be between 1 and 50.")
    train, test, cutoff = chronological_split(dataset.ratings)
    start = perf_counter()
    model = Recommender(dataset, train)
    fit_seconds = perf_counter() - start
    truth = {}
    cold_positive = 0
    for uid, mid, rating, _ in test:
        if rating < 4:
            continue
        uid, mid = int(uid), int(mid)
        i = model.item_index[mid]
        if not model.available[i]:
            cold_positive += 1
            continue
        if model.history(uid)[i] == 0:
            truth.setdefault(uid, set()).add(mid)
    warm_users = sorted(uid for uid in truth if uid in model.user_index)
    if not warm_users:
        raise ValueError("No evaluable warm users with positive test ratings.")
    metrics = {}
    per_user = {}
    catalog_size = int(model.available.sum())
    for method in METHODS:
        rows, latencies, lengths, diversities = [], [], [], []
        recommended_catalog = set()
        for uid in warm_users:
            start = perf_counter()
            output = model.recommend(uid, method, k)
            latencies.append((perf_counter() - start) * 1000)
            ids = [movie["movie_id"] for movie in output["recommendations"]]
            lengths.append(len(ids))
            row = {"user_id": uid, "relevant_count": len(truth[uid]),
                   "recommendations": ids, **ranking_metrics(ids, truth[uid], k)}
            rows.append(row)
            recommended_catalog.update(ids)
            features = model.features[[model.item_index[mid] for mid in ids]]
            if len(features) > 1:
                sim = features @ features.T
                diversities.append(float((1 - sim)[np.triu_indices(len(ids), 1)].mean()))
            else:
                diversities.append(0.0)
        per_user[method] = rows
        metrics[method] = {
            **{name + f"@{k}": float(np.mean([row[name] for row in rows]))
               for name in ("precision", "recall", "ndcg", "hit_rate")},
            "catalog_coverage": len(recommended_catalog) / catalog_size,
            "genre_diversity": float(np.mean(diversities)),
            "full_list_rate": float(np.mean(np.array(lengths) == k)),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
        }
    # Paired uncertainty on the same users, versus the heuristic baseline.
    # Fixed resampling seed is independent of model fitting (which is deterministic).
    intervals = {}
    baseline = np.array([row["ndcg"] for row in per_user["popular"]])
    for method in ("content", "item_cf", "genre_popular"):
        difference = np.array([row["ndcg"] for row in per_user[method]]) - baseline
        rng = np.random.default_rng(42)
        samples = rng.choice(difference, size=(2000, len(difference)), replace=True).mean(axis=1)
        intervals[method] = {"mean_ndcg_difference": float(difference.mean()),
                             "bootstrap_95_percent": np.quantile(samples, [0.025, 0.975]).tolist()}
    return {
        "dataset": {"name": "MovieLens 100K", "url": DATA_URL,
                    "archive_sha256": ARCHIVE_SHA256,
                    "ratings": len(dataset.ratings), "users": len(np.unique(dataset.ratings[:, 0])),
                    "movies": len(dataset.movies)},
        "protocol": {"split": "global chronological 80/20, equal timestamps kept together",
                     "cutoff_unix": cutoff,
                     "cutoff_utc": datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
                     "train_ratings": len(train), "test_ratings": len(test),
                     "train_users": len(model.user_index), "candidate_movies": catalog_size,
                     "evaluated_warm_users": len(warm_users),
                     "cold_test_users": len(set(test[:, 0]) - set(train[:, 0])),
                     "cold_item_positive_test_ratings": cold_positive,
                     "cold_users_with_evaluable_positive": len(set(truth) - set(warm_users)),
                     "positive_rating": 4, "k": k,
                     "parameters": model.parameters,
                     "bootstrap_resamples": 2000, "bootstrap_seed": 42},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform(), "fit_seconds": fit_seconds,
                    "timing_scope": "in-process recommend including response construction; no HTTP/load"},
        "metrics": metrics, "paired_ndcg_vs_popular": intervals, "per_user": per_user,
        "examples": {method: model.recommend(1, method, 5) for method in METHODS},
        "cold_start_example": model.recommend(9999, "item_cf", 5),
    }
