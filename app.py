import io
import base64
from pathlib import Path
from functools import lru_cache
import os

import joblib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, render_template, request, redirect, url_for, flash

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.secret_key = "mpox-detection-skey"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COUNTRIES = [
    "Africa",
    "Andorra",
    "Angola",
    "Argentina",
    "Aruba",
    "Asia",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahamas",
    "Bahrain",
    "Barbados",
    "Belgium",
    "Benin",
    "Bermuda",
    "Bolivia",
    "Bosnia and Herzegovina",
    "Brazil",
    "Bulgaria",
    "Burundi",
    "Cambodia",
    "Cameroon",
    "Canada",
    "Central African Republic",
    "Chile",
    "China",
    "Colombia",
    "Congo",
    "Costa Rica",
    "Cote d'Ivoire",
    "Croatia",
    "Cuba",
    "Curacao",
    "Cyprus",
    "Czechia",
    "Democratic Republic of Congo",
    "Denmark",
    "Dominican Republic",
    "Ecuador",
    "Egypt",
    "El Salvador",
    "Estonia",
    "Europe",
    "Finland",
    "France",
    "Gabon",
    "Georgia",
    "Germany",
    "Ghana",
    "Gibraltar",
    "Greece",
    "Greenland",
    "Guadeloupe",
    "Guam",
    "Guatemala",
    "Guinea",
    "Guyana",
    "Honduras",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iran",
    "Ireland",
    "Israel",
    "Italy",
    "Jamaica",
    "Japan",
    "Jordan",
    "Kenya",
    "Kosovo",
    "Laos",
    "Latvia",
    "Lebanon",
    "Liberia",
    "Lithuania",
    "Luxembourg",
    "Malaysia",
    "Malta",
    "Martinique",
    "Mauritius",
    "Mexico",
    "Moldova",
    "Monaco",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Nepal",
    "Netherlands",
    "New Caledonia",
    "New Zealand",
    "Nigeria",
    "North America",
    "Norway",
    "Oceania",
    "Oman",
    "Pakistan",
    "Panama",
    "Paraguay",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "Russia",
    "Rwanda",
    "Saint Martin (French part)",
    "San Marino",
    "Saudi Arabia",
    "Serbia",
    "Sierra Leone",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "South Africa",
    "South America",
    "South Sudan",
    "Spain",
    "Sri Lanka",
    "Sudan",
    "Sweden",
    "Switzerland",
    "Thailand",
    "Trinidad and Tobago",
    "Turkey",
    "Uganda",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Uruguay",
    "Venezuela",
    "Vietnam",
    "Zambia",
    "Zimbabwe",
]


CLASS_NAMES = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]
DEFAULT_CONFIDENCE_THRESHOLD = 0.80
COUNTRY_LOOKUP = {country.casefold(): country for country in COUNTRIES}


def normalize_country_name(country_name: str) -> str:
    normalized = (country_name or "").strip()
    if not normalized:
        return ""
    return COUNTRY_LOOKUP.get(normalized.casefold(), normalized)


# ---------------------------------------------------------------------------
# Model loading  (cached so models are loaded only once per process)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_artifacts():
    models_dir = BASE_DIR / "models"

    onnx_dir = models_dir / "onnx"
    image_onnx_path = onnx_dir / "hybrid_model.onnx"
    lstm_onnx_path = onnx_dir / "mpox_lstm_model.onnx"

    if not (image_onnx_path.exists() and lstm_onnx_path.exists()):
        raise RuntimeError(
            "ONNX models not found. Expected: "
            f"{image_onnx_path} and {lstm_onnx_path}. "
            "Run tools/convert_models_to_onnx.py to generate them."
        )

    import onnxruntime as ort

    image_sess = ort.InferenceSession(
        str(image_onnx_path), providers=["CPUExecutionProvider"]
    )
    lstm_sess = ort.InferenceSession(
        str(lstm_onnx_path), providers=["CPUExecutionProvider"]
    )

    predictive_dir = models_dir / "predictive_models"
    thresholds = joblib.load(str(predictive_dir / "mpox_thresholds.pkl"))
    country_windows = joblib.load(str(predictive_dir / "country_windows.pkl"))

    return {
        "image_model": image_sess,
        "lstm_model": lstm_sess,
        "country_windows": country_windows,
        "low_threshold": thresholds["low_threshold"],
        "high_threshold": thresholds["high_threshold"],
    }


# ---------------------------------------------------------------------------
# Helper / prediction functions
# ---------------------------------------------------------------------------
def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    img_array = np.asarray(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    # In TF 2.18, tf.keras.applications.efficientnet.preprocess_input is a no-op
    # (models include preprocessing internally). Keep pixels in [0,255].
    return img_array.astype(np.float32, copy=False)


def open_and_validate_uploaded_image(file_storage) -> Image.Image:
    """Open an uploaded/captured file as an image.

    Accepts any image format Pillow can decode. Raises ValueError for non-images.
    """
    img_bytes = file_storage.read()
    if not img_bytes:
        raise ValueError("Empty upload")

    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)
    except UnidentifiedImageError as exc:
        raise ValueError("Not an image") from exc

    return img


def predict_disease(img: Image.Image, *, threshold: float, image_model) -> dict:
    img_array = preprocess_pil_image(img)
    input_name = image_model.get_inputs()[0].name
    predictions = image_model.run(None, {input_name: img_array})[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    if confidence < threshold:
        return {"valid": False, "message": "Invalid or non-skin lesion image detected."}

    predicted_class = CLASS_NAMES[predicted_index]
    return {"valid": True, "disease": predicted_class, "confidence": confidence}


def classify_risk(
    predicted_value: float, *, low_threshold: float, high_threshold: float
) -> str:
    if predicted_value < low_threshold:
        return "Low"
    if predicted_value < high_threshold:
        return "Medium"
    return "High"


def predict_outbreak(
    country_name: str,
    *,
    lstm_model,
    country_windows,
    low_threshold: float,
    high_threshold: float,
):
    if country_name not in country_windows:
        return {"error": "Country data not available."}

    recent_data = country_windows[country_name]
    recent_data = np.expand_dims(recent_data, axis=0).astype(np.float32, copy=False)
    input_name = lstm_model.get_inputs()[0].name
    predicted_value = float(lstm_model.run(None, {input_name: recent_data})[0][0][0])
    risk_level = classify_risk(
        predicted_value, low_threshold=low_threshold, high_threshold=high_threshold
    )
    return {"Predicted Rolling Avg (Scaled)": predicted_value, "Risk Level": risk_level}


def full_system_prediction(
    img: Image.Image,
    country_name: str,
    *,
    confidence_threshold: float,
    image_model,
    lstm_model,
    country_windows,
    low_threshold: float,
    high_threshold: float,
):
    disease_result = predict_disease(
        img, threshold=confidence_threshold, image_model=image_model
    )

    if not disease_result["valid"]:
        return {"error": "Invalid or non-skin lesion image detected please try again."}

    predicted_disease = disease_result["disease"]
    confidence = f"{(float(disease_result['confidence']) * 100):.2f}%"

    country_available = country_name in country_windows

    if not country_available:
        return {
            "Disease Prediction": predicted_disease,
            "Confidence": confidence,
            "Country": country_name,
            "Message": "Country data not available for outbreak prediction.",
        }

    if predicted_disease != "Monkeypox":
        return {
            "Disease Prediction": predicted_disease,
            "Confidence": confidence,
            "Country": country_name,
        }

    outbreak_result = predict_outbreak(
        country_name,
        lstm_model=lstm_model,
        country_windows=country_windows,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )

    return {
        "Disease Prediction": predicted_disease,
        "Confidence": confidence,
        "Country": country_name,
        "Regional Risk Level": outbreak_result["Risk Level"],
    }


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    selected_country = normalize_country_name(request.args.get("country", ""))

    return render_template(
        "index.html",
        countries=COUNTRIES,
        selected_country=selected_country,
    )


@app.route("/predict", methods=["POST"])
def predict():
    # ---- image ----
    file = request.files.get("capture_image") or request.files.get("image")
    if file is None or (file.filename == "" and not getattr(file, "content_length", None)):
        flash("Upload or capture an image first.", "warning")
        return redirect(url_for("index"))

    # ---- form values ----
    raw_country = request.form.get("country", "")
    country = normalize_country_name(raw_country) or COUNTRIES[0]
    confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

    # ---- read image ----
    try:
        img = open_and_validate_uploaded_image(file)
    except Exception:
        flash("Could not read the uploaded file as an image. Please try a valid image.", "danger")
        return redirect(url_for("index"))

    # ---- run inference ----
    artifacts = load_artifacts()
    result = full_system_prediction(
        img,
        country,
        confidence_threshold=confidence_threshold,
        image_model=artifacts["image_model"],
        lstm_model=artifacts["lstm_model"],
        country_windows=artifacts["country_windows"],
        low_threshold=artifacts["low_threshold"],
        high_threshold=artifacts["high_threshold"],
    )

    # ---- encode image as base64 for inline display ----
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render_template(
        "result.html",
        result=result,
        country=country,
        img_b64=img_b64,
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host = "0.0.0.0", port=5000)
