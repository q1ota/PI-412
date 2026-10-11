from io import BytesIO
import numpy as np
import pytest
from fastapi.testclient import TestClient

from recsys import api, data
from recsys.data import Dataset, Movie, chronological_split, load_dataset
from recsys.evaluate import ranking_metrics
from recsys.models import METHODS, Recommender


@pytest.fixture
def dataset():
    movies = (
        Movie(10, "Action A", ("Action",)), Movie(20, "Action B", ("Action",)),
        Movie(30, "Comedy C", ("Comedy",)), Movie(40, "Drama D", ("Drama",)),
        Movie(50, "New Action", ("Action",)),
        Movie(60, "Action Comedy", ("Action", "Comedy")),
    )
    ratings = np.array([
        [1, 10, 5, 1], [1, 30, 1, 2], [2, 10, 4, 3], [2, 20, 5, 4],
        [2, 40, 2, 5], [3, 10, 5, 6], [3, 20, 4, 7], [3, 30, 5, 8],
        [4, 40, 5, 9], [4, 60, 4, 10],
    ], dtype=np.int64)
    return Dataset(movies, ratings)


@pytest.fixture
def model(dataset):
    return Recommender(dataset, dataset.ratings)


def test_content_profile_matches_liked_genre(model):
    result = model.recommend(1, "content", 3)
    assert result["recommendations"][0]["movie_id"] == 20
    assert result["recommendations"][0]["score"] == pytest.approx(1)
    assert result["recommendations"][1]["movie_id"] == 60


def test_item_cf_matches_hand_calculated_cosine_and_shrinkage(model):
    # Movies 10 and 20: 3 and 2 positive users, 2 in common.
    expected = (2 / np.sqrt(3 * 2)) * (2 / (2 + 10))
    assert model.similarities[model.item_index[20], model.item_index[10]] == pytest.approx(expected)
    assert np.all(np.diag(model.similarities) == 0)
    result = model.recommend(1, "item_cf", 3)
    assert result["recommendations"][0]["movie_id"] == 20
    assert result["recommendations"][0]["score"] == pytest.approx(expected)


@pytest.mark.parametrize("method", METHODS)
def test_seen_and_unavailable_movies_never_returned_and_order_is_stable(model, method):
    result = model.recommend(1, method, 50)
    ids = [row["movie_id"] for row in result["recommendations"]]
    assert not set(ids) & {10, 30, 50}
    assert len(ids) == len(set(ids))
    assert result == model.recommend(1, method, 50)
    assert all(np.isfinite(row["score"]) for row in result["recommendations"])


def test_popularity_smoothing_is_calculated_only_from_train(model, dataset):
    i = model.item_index[20]
    expected = (9 + 50 * dataset.ratings[:, 2].mean()) / (2 + 50)
    assert model.popularity[i] == pytest.approx(expected)


def test_genre_baseline_filters_and_inferrs_favorite(model):
    auto = model.recommend(1, "genre_popular", 10)
    manual = model.recommend(1, "genre_popular", 10, "Action")
    assert auto == manual
    assert {row["movie_id"] for row in auto["recommendations"]} == {20, 60}
    assert all("Action" in row["genres"] for row in manual["recommendations"])


@pytest.mark.parametrize("method", METHODS)
def test_cold_start_uses_popularity(model, method):
    result = model.recommend(9999, method, 3)
    assert result["effective_method"] == "popular"
    assert [r["movie_id"] for r in result["recommendations"]] == [
        r["movie_id"] for r in model.recommend(9999, "popular", 3)["recommendations"]]


def test_user_with_only_low_ratings_falls_back(dataset):
    train = dataset.ratings.copy()
    train[train[:, 0] == 1, 2] = 2
    result = Recommender(dataset, train).recommend(1, "content", 3)
    assert result["effective_method"] == "popular"
    assert not {10, 30} & {row["movie_id"] for row in result["recommendations"]}


def test_train_only_no_held_out_rating_changes_model(dataset):
    train = dataset.ratings[:8]
    changed = dataset.ratings.copy()
    changed[8:, 2] = 1
    first = Recommender(dataset, train)
    second = Recommender(Dataset(dataset.movies, changed), train)
    for method in METHODS:
        assert first.recommend(1, method) == second.recommend(1, method)


def test_global_chronological_split_keeps_timestamp_ties_together():
    ratings = np.array([[1, i + 1, 4, t] for i, t in enumerate([1, 2, 3, 4, 5, 5])])
    train, test, cutoff = chronological_split(ratings, 0.8)
    assert cutoff == 5
    assert len(train) == 4 and len(test) == 2
    assert train[:, 3].max() < test[:, 3].min()
    assert not set(map(tuple, train[:, :2])) & set(map(tuple, test[:, :2]))


def test_ranking_metrics_match_manual_computation():
    result = ranking_metrics([8, 2, 1], {1, 2, 9}, 3)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["ndcg"] == pytest.approx((1 / np.log2(3) + 0.5) / (1 + 1 / np.log2(3) + 0.5))
    assert result["hit_rate"] == 1
    assert ranking_metrics([], {1}, 3)["precision"] == 0
    # Short lists do not improve precision by reducing the denominator.
    assert ranking_metrics([1], {1}, 3)["precision"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": 51}, {"user_id": 0},
                                   {"method": "bad"}, {"genre": "Action"}])
def test_model_rejects_invalid_request(model, kwargs):
    parameters = {"user_id": 1, "method": "item_cf", **kwargs}
    with pytest.raises(ValueError):
        model.recommend(**parameters)


def test_duplicate_and_invalid_ratings_are_rejected(dataset):
    with pytest.raises(ValueError, match="Duplicate"):
        Recommender(dataset, np.vstack((dataset.ratings, dataset.ratings[:1])))
    bad = dataset.ratings.copy()
    bad[0, 2] = 6
    with pytest.raises(ValueError, match="between 1 and 5"):
        Recommender(dataset, bad)


def test_data_loader_parses_movielens_encoding_and_rejects_unknown_item(tmp_path):
    flags = ["0"] * 19
    flags[1] = "1"
    (tmp_path / "u.item").write_text("|".join(["1", "Café", "", "", "", *flags]) + "\n",
                                   encoding="iso-8859-1")
    (tmp_path / "u.data").write_text("1\t1\t5\t123\n")
    dataset = load_dataset(tmp_path)
    assert dataset.movies == (Movie(1, "Café", ("Action",)),)
    (tmp_path / "u.data").write_text("1\t2\t5\t123\n")
    with pytest.raises(ValueError, match="absent"):
        load_dataset(tmp_path)


def test_download_rejects_corrupt_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: BytesIO(b"corrupted archive"))
    with pytest.raises(ValueError, match="checksum"):
        data.download_dataset(tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_api_serves_genuine_model(model, monkeypatch):
    monkeypatch.setattr(api, "get_recommender", lambda: model)
    with TestClient(api.app) as client:
        response = client.get("/recsys/recommend", params={"user_id": 1, "method": "item_cf", "k": 2})
        assert response.status_code == 200
        assert response.json()["recommendations"][0]["movie_id"] == 20
        assert response.json()["model_version"] == "movielens-100k-temporal-v1"
        assert client.get("/recsys/recommend?user_id=1&method=bad").status_code == 422
        assert client.get("/recsys/recommend?user_id=1&k=51").status_code == 422
        assert client.get("/recsys/recommend?user_id=1&method=genre_popular&genre=bad").status_code == 400


def test_api_reports_missing_data_as_unavailable(monkeypatch):
    def missing():
        raise FileNotFoundError("Run: python -m recsys download")
    monkeypatch.setattr(api, "get_recommender", missing)
    with TestClient(api.app) as client:
        response = client.get("/recsys/recommend?user_id=1")
        assert response.status_code == 503
        assert "download" in response.json()["detail"]


def test_router_is_integrated_in_existing_team_app():
    import main
    paths = set(main.app.openapi()["paths"])
    assert {"/health", "/text/sentiment", "/audio/transcribe", "/image/classify",
            "/video/analyze", "/recsys/recommend"}.issubset(paths)
