"""Small standalone app and a router for the team's existing FastAPI app."""

from functools import lru_cache
import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Query

from recsys.data import DEFAULT_DATA_DIR, chronological_split, load_dataset
from recsys.models import Method, Recommender


router = APIRouter(prefix="/recsys", tags=["RecSys"])


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    dataset = load_dataset(Path(os.environ.get("RECSYS_DATA_DIR", str(DEFAULT_DATA_DIR))))
    train, _, _ = chronological_split(dataset.ratings)
    return Recommender(dataset, train)


@router.get("/recommend")
def recommend(user_id: int = Query(ge=1), method: Method = "item_cf",
              k: int = Query(default=10, ge=1, le=50), genre: str | None = None):
    try:
        result = get_recommender().recommend(user_id, method, k, genre)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {**result, "model_version": "movielens-100k-temporal-v1"}


app = FastAPI(title="MovieLens RecSys demo", version="1.0.0")
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
