"""Two classical recommenders and two transparent heuristic baselines."""

from typing import Literal

import numpy as np

from recsys.data import Dataset, GENRES, validate_ratings


Method = Literal["content", "item_cf", "popular", "genre_popular"]
METHODS = ("content", "item_cf", "popular", "genre_popular")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)


class Recommender:
    """Fit statistics once; serve deterministic top-K lists without downloads.

    Explicit ratings >= 4 are interpreted as positive feedback. Unknown ratings
    are missing observations, not dislikes. All previously rated films are hidden.
    """

    def __init__(self, dataset: Dataset, train: np.ndarray, *, neighbors: int = 40,
                 shrinkage: float = 10.0, prior_count: float = 50.0):
        validate_ratings(train)
        if neighbors < 1 or shrinkage < 0 or prior_count <= 0:
            raise ValueError("Invalid recommender parameters.")
        self.movies = dataset.movies
        self.movie_ids = np.array([m.movie_id for m in self.movies])
        self.item_index = {int(mid): i for i, mid in enumerate(self.movie_ids)}
        if len(self.item_index) != len(self.movies):
            raise ValueError("Movie IDs must be unique.")
        if not set(train[:, 1]).issubset(self.item_index):
            raise ValueError("Training ratings reference unknown movies.")
        user_ids = np.unique(train[:, 0])
        self.user_index = {int(uid): i for i, uid in enumerate(user_ids)}
        matrix = np.zeros((len(user_ids), len(self.movies)), dtype=np.float64)
        for uid, mid, rating, _ in train:
            matrix[self.user_index[int(uid)], self.item_index[int(mid)]] = rating
        self.ratings = matrix
        self.counts = (matrix > 0).sum(axis=0)
        # Common candidate catalog for every algorithm, known at the train cutoff.
        self.available = self.counts > 0
        self.popularity = (matrix.sum(axis=0) + prior_count * train[:, 2].mean()) / (
            self.counts + prior_count
        )
        self.genre_flags = np.array(
            [[genre in movie.genres for genre in GENRES] for movie in self.movies],
            dtype=np.float64,
        )
        # Smoothed IDF fitted only on the active training catalog.
        idf = np.log((1 + self.available.sum()) /
                     (1 + self.genre_flags[self.available].sum(axis=0))) + 1
        self.features = normalize_rows(self.genre_flags * idf)

        # Cosine between movie columns of a user × liked-movie binary matrix.
        liked = (matrix >= 4).astype(np.float64)
        common = liked.T @ liked
        denominator = np.sqrt(np.outer(liked.sum(axis=0), liked.sum(axis=0)))
        similarities = np.divide(common, denominator, out=np.zeros_like(common),
                                 where=denominator != 0)
        confidence = np.divide(common, common + shrinkage, out=np.zeros_like(common),
                               where=(common + shrinkage) != 0)
        similarities *= confidence
        np.fill_diagonal(similarities, 0)
        self.similarities = np.zeros_like(similarities)
        for i, row in enumerate(similarities):
            # Movie ID breaks similarity ties reproducibly.
            order = np.lexsort((self.movie_ids, -row))[:neighbors]
            self.similarities[i, order] = row[order]
        self.parameters = {"neighbors": neighbors, "shrinkage": shrinkage,
                           "prior_count": prior_count, "positive_rating": 4}

    def history(self, user_id: int) -> np.ndarray:
        index = self.user_index.get(user_id)
        if index is None:
            return np.zeros(len(self.movies), dtype=np.float64)
        return self.ratings[index]

    def scores(self, user_id: int, method: Method, genre: str | None = None):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        if genre is not None and (method != "genre_popular" or genre not in GENRES):
            raise ValueError("genre is a known MovieLens genre for genre_popular only.")
        history = self.history(user_id)
        weights = np.maximum(history - 3, 0)
        candidates = self.available & (history == 0)
        effective = method
        reason = ""
        if method in ("content", "item_cf") and not weights.any():
            effective = "popular"
            reason = "Нет положительной истории: используется общий рейтинг популярности."
        if effective == "content":
            profile = weights @ self.features
            norm = np.linalg.norm(profile)
            if norm == 0:
                effective = "popular"
                reason = "В истории нет жанровых признаков: используется популярность."
                scores = self.popularity.copy()
            else:
                scores = self.features @ (profile / norm)
        elif effective == "item_cf":
            scores = self.similarities @ weights / weights.sum()
        else:
            scores = self.popularity.copy()
            if method == "genre_popular":
                if genre is None and weights.any():
                    genre = GENRES[int(np.argmax(weights @ self.genre_flags))]
                if genre is None:
                    effective = "popular"
                    reason = "Жанр и положительная история отсутствуют: общая популярность."
                else:
                    candidates &= self.genre_flags[:, GENRES.index(genre)].astype(bool)
                    reason = f"Высокий сглаженный рейтинг в жанре {genre}."
        return scores, candidates, effective, reason

    def rank(self, user_id: int, method: Method, k: int = 10, genre: str | None = None):
        if user_id < 1 or not 1 <= k <= 50:
            raise ValueError("user_id must be positive and k must be between 1 and 50.")
        scores, candidates, effective, reason = self.scores(user_id, method, genre)
        indices = np.flatnonzero(candidates)
        # Score, then quality-adjusted popularity, rating count, and movie ID.
        order = np.lexsort((self.movie_ids[indices], -self.counts[indices],
                            -self.popularity[indices], -scores[indices]))
        return indices[order[:k]], scores, effective, reason

    def recommend(self, user_id: int, method: Method = "item_cf", k: int = 10,
                  genre: str | None = None) -> dict:
        indices, scores, effective, reason = self.rank(user_id, method, k, genre)
        weights = np.maximum(self.history(user_id) - 3, 0)
        recommendations = []
        for i in indices:
            movie = self.movies[i]
            explanation = reason
            if effective == "content":
                explanation = "Жанры похожи на жанры фильмов, которым вы поставили 4–5."
            elif effective == "item_cf":
                contributions = self.similarities[i] * weights
                if contributions.any():
                    source = self.movies[int(np.argmax(contributions))]
                    explanation = f"Совместные положительные оценки с «{source.title}»."
                else:
                    explanation = "Нет похожего фильма в истории: место определила популярность."
            elif not explanation:
                explanation = "Высокий средний рейтинг со сглаживанием малых выборок."
            recommendations.append({
                "movie_id": movie.movie_id, "title": movie.title,
                "genres": list(movie.genres), "score": float(scores[i]),
                "reason": explanation,
            })
        return {"user_id": user_id, "method": method, "effective_method": effective,
                "k": k, "recommendations": recommendations}
