
import os
import requests


# ==========================================
# API CONFIGURATION
# ==========================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

HEALTH_URL = f"{API_URL}/health"
PREDICT_URL = f"{API_URL}/predict"

REQUEST_TIMEOUT = 180


# ==========================================
# CHECK API HEALTH
# ==========================================

def check_api_health():

    try:

        response = requests.get(
            HEALTH_URL,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as exc:

        return {
            "status": "disconnected",
            "model_loaded": False,
            "error": str(exc)
        }


# ==========================================
# SEND IMAGE TO API
# ==========================================

def predict_image(image_bytes, filename):

    if not image_bytes:

        raise ValueError(
            "Uploaded image is empty."
        )

    files = {
        "file": (
            filename,
            image_bytes,
            "application/octet-stream"
        )
    }

    try:

        response = requests.post(
            PREDICT_URL,
            files=files,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        result = response.json()

        if "detections" not in result:
            raise ValueError(
                "Invalid prediction response."
            )

        if "total_detections" not in result:
            raise ValueError(
                "Prediction count missing from response."
            )

        return result

    except requests.HTTPError as exc:

        try:
            message = response.json().get(
                "detail",
                str(exc)
            )
        except ValueError:
            message = str(exc)

        raise RuntimeError(
            f"API error: {message}"
        ) from exc

    except requests.RequestException as exc:

        raise RuntimeError(
            f"Could not connect to FastAPI: {exc}"
        ) from exc
