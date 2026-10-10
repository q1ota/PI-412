import numpy as np
import torch
from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sentiment_analysis(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_sentiment_pipeline",
        lambda: lambda text: [{"label": "POSITIVE", "score": 0.98}],
    )

    response = client.post("/text/sentiment", json={"text": "I liked the result"})

    assert response.status_code == 200
    assert response.json()["label"] == "POSITIVE"
    assert response.json()["model"] == "distilbert-base-uncased-finetuned-sst-2-english"


def test_sentiment_rejects_empty_text():
    response = client.post("/text/sentiment", json={"text": ""})

    assert response.status_code == 422


def test_audio_transcription(monkeypatch):
    monkeypatch.setattr(main.librosa, "load", lambda *args, **kwargs: (np.ones(8), 16_000))
    monkeypatch.setattr(
        main,
        "get_asr_pipeline",
        lambda: lambda audio: {"text": "recognized words"},
    )

    response = client.post(
        "/audio/transcribe",
        files={"file": ("sample.wav", b"audio bytes", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "recognized words"


def test_image_classification(monkeypatch):
    class FakeModel:
        def __call__(self, _input_tensor):
            return torch.tensor([[0.1, 0.8, 0.05, 0.03, 0.02]])

    monkeypatch.setattr(
        main,
        "get_image_model",
        lambda: (FakeModel(), lambda _image: torch.zeros((3, 2, 2)), ["a", "b", "c", "d", "e"]),
    )

    image = main.Image.new("RGB", (2, 2), color="white")
    buffer = main.BytesIO()
    image.save(buffer, format="PNG")
    response = client.post(
        "/image/classify?top_k=3",
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3
    assert response.json()["predictions"][0]["label"] == "b"


def test_image_classification_rejects_invalid_file():
    response = client.post(
        "/image/classify",
        files={"file": ("sample.png", b"not an image", "image/png")},
    )

    assert response.status_code == 422


def test_video_analysis(monkeypatch):
    class FakeCapture:
        def __init__(self, _path):
            self.frame_index = 0

        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == main.cv2.CAP_PROP_FPS:
                return 1.0
            if property_id == main.cv2.CAP_PROP_FRAME_COUNT:
                return 3
            return 0

        def set(self, _property_id, value):
            self.frame_index = int(value)

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(main.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(
        main,
        "get_video_pipeline",
        lambda: lambda _image, top_k: [
            {"label": "tabby cat", "score": 0.9},
            {"label": "tiger cat", "score": 0.05},
            {"label": "Egyptian cat", "score": 0.02},
        ][:top_k],
    )

    response = client.post(
        "/video/analyze?interval_seconds=1",
        files={"file": ("sample.mp4", b"video bytes", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["frames_analyzed"] == 3
    assert response.json()["dominant_labels"][0] == {"label": "tabby cat", "frames": 3}
