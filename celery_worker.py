import base64
import io
import os

from celery import Celery
from PIL import Image, ImageOps

from app import load_artifacts, full_system_prediction


REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Load once per worker process
artifacts = load_artifacts()


@celery.task(name="predict_task")
def predict_task(img_b64: str, country: str):
    img_bytes = base64.b64decode(img_b64)

    img = Image.open(io.BytesIO(img_bytes))
    img = ImageOps.exif_transpose(img)

    result = full_system_prediction(
        img,
        country,
        confidence_threshold=0.80,
        image_model=artifacts["image_model"],
        lstm_model=artifacts["lstm_model"],
        country_windows=artifacts["country_windows"],
        low_threshold=artifacts["low_threshold"],
        high_threshold=artifacts["high_threshold"],
    )

    # Return everything needed to render the existing result page.
    return {
        "result": result,
        "country": country,
        "img_b64": img_b64,
    }
