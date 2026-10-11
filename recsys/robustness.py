"""Controlled offline shilling sensitivity experiment on the training split."""

import numpy as np

from recsys.data import Dataset, chronological_split
from recsys.models import Recommender


def run_shilling_experiment(dataset: Dataset, fake_users: int = 50) -> dict:
    """Add synthetic ratings locally; no accounts or external services are touched.

    Attackers give five stars to a low-support target and 20 popular anchor films.
    Target and anchors are chosen from train only. Results are a sensitivity check,
    not an estimate of attack frequency or a production security assessment.
    """
    if fake_users < 1:
        raise ValueError("fake_users must be positive.")
    train, test, cutoff = chronological_split(dataset.ratings)
    before = Recommender(dataset, train)
    eligible = np.flatnonzero((before.counts >= 5) & (before.counts <= 10))
    if not len(eligible):
        raise ValueError("No target with 5–10 training ratings.")
    target = int(eligible[0])  # lowest movie ID with the declared support range
    anchors = np.lexsort((before.movie_ids, -before.counts))
    anchors = [int(i) for i in anchors if i != target and before.available[i]][:20]
    rows = []
    first_fake_id = int(dataset.ratings[:, 0].max()) + 1
    for uid in range(first_fake_id, first_fake_id + fake_users):
        rows.extend([uid, int(before.movie_ids[i]), 5, cutoff - 1] for i in [target, *anchors])
    attacked_train = np.vstack((train, np.array(rows, dtype=np.int64)))
    after = Recommender(dataset, attacked_train)
    users = [int(uid) for uid in np.unique(test[:, 0])
             if uid in before.user_index and before.history(int(uid))[target] == 0]
    if not users:
        raise ValueError("No evaluable users for the attack experiment.")
    comparisons = {}
    for method in ("popular", "item_cf"):
        ranks = []
        for model in (before, after):
            positions = []
            for uid in users:
                scores, candidates, _, _ = model.scores(uid, method)
                ids = np.flatnonzero(candidates)
                order = np.lexsort((model.movie_ids[ids], -model.counts[ids],
                                    -model.popularity[ids], -scores[ids]))
                positions.append(int(np.flatnonzero(ids[order] == target)[0]) + 1)
            ranks.append(np.array(positions))
        comparisons[method] = {
            "median_rank_before": float(np.median(ranks[0])),
            "median_rank_after": float(np.median(ranks[1])),
            "target_in_top10_before": int((ranks[0] <= 10).sum()),
            "target_in_top10_after": int((ranks[1] <= 10).sum()),
        }
    return {
        "scenario": f"{fake_users} synthetic five-star profiles, target + 20 most-rated train anchors",
        "fake_users": fake_users, "added_ratings": len(rows),
        "train_rating_increase_fraction": len(rows) / len(train),
        "target_movie_id": int(before.movie_ids[target]),
        "target_title": before.movies[target].title,
        "target_train_ratings_before": int(before.counts[target]),
        "anchor_movie_ids": before.movie_ids[anchors].tolist(),
        "evaluated_existing_users": len(users), "comparisons": comparisons,
    }
