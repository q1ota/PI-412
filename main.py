"""FastAPI endpoints for the four multimodal models in this project."""

from collections import Counter
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import math
import tempfile

import cv2
import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from torchvision.models import ResNet50_Weights, resnet50

from recsys.api import router as recsys_router


MAX_UPLOAD_SIZE = 50 * 1024 * 1024
MAX_VIDEO_FRAMES = 30

app = FastAPI(
    title="Multimodal ML API",
    description=(
        "API for sentiment analysis, speech recognition, image classification, "
        "and video frame classification. Models are downloaded on first use."
    ),
    version="1.0.0",
)
app.include_router(recsys_router)


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@lru_cache(maxsize=1)
def get_asr_pipeline():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


@lru_cache(maxsize=1)
def get_video_pipeline():
    return pipeline("image-classification", model="google/vit-base-patch16-224")


@lru_cache(maxsize=1)
def get_image_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    return model, weights.transforms(), weights.meta["categories"]


def read_upload(file: UploadFile) -> bytes:
    contents = file.file.read(MAX_UPLOAD_SIZE + 1)
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File must be 50 MB or smaller.")
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return contents


@app.get("/", tags=["Status"])
def root():
    return {"message": "Multimodal ML API", "docs": "/docs"}


@app.get("/health", tags=["Status"])
def health():
    return {"status": "ok"}


@app.post("/text/sentiment", tags=["Text"])
def analyze_sentiment(request: SentimentRequest):
    result = get_sentiment_pipeline()(request.text)[0]
    return {
        "label": result["label"],
        "score": float(result["score"]),
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
    }


@app.post("/audio/transcribe", tags=["Audio"])
def transcribe_audio(file: UploadFile = File(...)):
    contents = read_upload(file)
    try:
        audio, sample_rate = librosa.load(BytesIO(contents), sr=16_000, mono=True)
    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail="Could not decode audio. Try WAV, FLAC, or OGG format.",
        ) from error

    if audio.size == 0:
        raise HTTPException(status_code=422, detail="Audio contains no samples.")

    result = get_asr_pipeline()({"raw": audio, "sampling_rate": sample_rate})
    return {"text": result["text"], "model": "openai/whisper-tiny"}


@app.post("/image/classify", tags=["Image"])
def classify_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=5),
):
    contents = read_upload(file)
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=422, detail="Uploaded file is not a valid image.") from error

    model, preprocess, categories = get_image_model()
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.nn.functional.softmax(model(input_tensor)[0], dim=0)
        scores, class_ids = torch.topk(probabilities, k=top_k)

    predictions = [
        {"label": categories[int(class_id)], "score": float(score)}
        for score, class_id in zip(scores, class_ids)
    ]
    return {"predictions": predictions, "model": "ResNet-50 (ImageNet)"}


@app.post("/video/analyze", tags=["Video"])
def analyze_video(
    file: UploadFile = File(...),
    interval_seconds: float = Query(default=1.0, gt=0, le=60),
):
    contents = read_upload(file)
    suffix = Path(file.filename or "video.mp4").suffix[:10] or ".mp4"
    temporary_path = None
    capture = None
    frames = []

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(contents)
            temporary_path = temporary_file.name

        capture = cv2.VideoCapture(temporary_path)
        if not capture.isOpened():
            raise HTTPException(status_code=422, detail="Could not open the uploaded video.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise HTTPException(status_code=422, detail="Video has no readable frames.")

        requested_step = max(1, round(fps * interval_seconds))
        max_step = max(1, math.ceil(frame_count / MAX_VIDEO_FRAMES))
        frame_step = max(requested_step, max_step)

        for frame_index in range(0, frame_count, frame_step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(
                {
                    "timestamp_seconds": round(frame_index / fps, 2),
                    "image": Image.fromarray(rgb_frame),
                }
            )
            if len(frames) >= MAX_VIDEO_FRAMES:
                break
    finally:
        if capture is not None:
            capture.release()
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)

    if not frames:
        raise HTTPException(status_code=422, detail="Could not extract frames from the video.")

    classifier = get_video_pipeline()
    frame_results = []
    label_counts = Counter()
    for frame in frames:
        predictions = classifier(frame["image"], top_k=3)
        label_counts[predictions[0]["label"]] += 1
        frame_results.append(
            {
                "timestamp_seconds": frame["timestamp_seconds"],
                "predictions": [
                    {"label": item["label"], "score": float(item["score"])}
                    for item in predictions
                ],
            }
        )

    return {
        "frames_analyzed": len(frame_results),
        "dominant_labels": [
            {"label": label, "frames": count}
            for label, count in label_counts.most_common(5)
        ],
        "frames": frame_results,
        "model": "google/vit-base-patch16-224",
    }
