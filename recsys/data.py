"""Download the original GroupLens archive and read its documented format."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ARCHIVE_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-100k"
GENRES = (
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    genres: tuple[str, ...]


@dataclass(frozen=True)
class Dataset:
    movies: tuple[Movie, ...]
    ratings: np.ndarray  # user_id, movie_id, rating, Unix timestamp


def download_dataset(destination: Path = DEFAULT_DATA_DIR) -> Path:
    """Verify the archive, extract only needed files, retain the usage license.

    No demographic data is read or extracted. Downloaded data stays out of Git.
    """
    destination = Path(destination)
    with urlopen(DATA_URL, timeout=60) as response:
        archive_bytes = response.read()
    if hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("MovieLens archive checksum differs from the verified version.")
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(archive_bytes)) as archive:
        for filename in ("u.data", "u.item", "u.genre", "u.info", "README"):
            # Fixed member names: no arbitrary paths from an archive are extracted.
            (destination / filename).write_bytes(archive.read(f"ml-100k/{filename}"))
    return destination


def validate_ratings(ratings: np.ndarray) -> None:
    if ratings.ndim != 2 or ratings.shape[1] != 4 or not len(ratings):
        raise ValueError("Ratings must be a nonempty array with four columns.")
    if not np.issubdtype(ratings.dtype, np.integer):
        raise ValueError("MovieLens ratings must be integers.")
    if (ratings[:, :2] <= 0).any() or (ratings[:, 3] < 0).any():
        raise ValueError("IDs must be positive and timestamps nonnegative.")
    if ((ratings[:, 2] < 1) | (ratings[:, 2] > 5)).any():
        raise ValueError("Ratings must be between 1 and 5.")
    if len(np.unique(ratings[:, :2], axis=0)) != len(ratings):
        raise ValueError("Duplicate user/movie rating pairs are not supported.")


def load_dataset(directory: Path = DEFAULT_DATA_DIR) -> Dataset:
    directory = Path(directory)
    if not (directory / "u.data").is_file() or not (directory / "u.item").is_file():
        raise FileNotFoundError(
            f"MovieLens not found in {directory}. Run: python -m recsys download"
        )
    movies = []
    for line in (directory / "u.item").read_text(encoding="iso-8859-1").splitlines():
        fields = line.split("|")
        if len(fields) != 24 or any(flag not in ("0", "1") for flag in fields[5:]):
            raise ValueError("Invalid u.item record.")
        genres = tuple(g for g, flag in zip(GENRES, fields[5:]) if flag == "1")
        movies.append(Movie(int(fields[0]), fields[1], genres))
    ids = [movie.movie_id for movie in movies]
    if not movies or len(set(ids)) != len(ids) or min(ids) <= 0:
        raise ValueError("Movie IDs must be positive and unique.")
    ratings = np.loadtxt(directory / "u.data", dtype=np.int64, ndmin=2)
    validate_ratings(ratings)
    if not set(ratings[:, 1]).issubset(ids):
        raise ValueError("Ratings reference movies absent from the catalog.")
    return Dataset(tuple(sorted(movies, key=lambda m: m.movie_id)), ratings)


def chronological_split(ratings: np.ndarray, train_fraction: float = 0.8):
    """Global time cutoff; equal timestamps always stay on the same side."""
    validate_ratings(ratings)
    if not 0 < train_fraction < 1 or len(ratings) < 2:
        raise ValueError("Need at least two ratings and a fraction between 0 and 1.")
    cutoff = int(np.sort(ratings[:, 3])[int(len(ratings) * train_fraction)])
    train = ratings[ratings[:, 3] < cutoff].copy()
    test = ratings[ratings[:, 3] >= cutoff].copy()
    if not len(train) or not len(test):
        raise ValueError("Timestamp cutoff produced an empty train or test set.")
    return train, test, cutoff
