
import os
from pathlib import Path

import pytest
import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE = Path(
    os.getenv(
        "TEST_IMAGE_PATH",
        str(
            PROJECT_ROOT
            / "data"
            / "NEU-DET"
            / "train"
            / "images"
            / "crazing"
            / "crazing_1.jpg"
        )
    )
)

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def upload_test_image():
    """
    Upload a test image to the prediction endpoint.
    """

    if not TEST_IMAGE.is_file():
        pytest.skip(
            f"Test image not found: {TEST_IMAGE}"
        )

    with TEST_IMAGE.open("rb") as image:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={
                "file": (
                    TEST_IMAGE.name,
                    image,
                    "image/jpeg"
                )
            },
            timeout=180
        )

    return response


# ============================================================
# TEST 1: ROOT ENDPOINT
# ============================================================

def test_api_root():

    response = requests.get(
        f"{BASE_URL}/",
        timeout=30
    )

    assert response.status_code == 200

    print("\nAPI root endpoint working.")


# ============================================================
# TEST 2: HEALTH ENDPOINT
# ============================================================

def test_api_health():

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=30
    )

    assert response.status_code == 200

    print("\nAPI health endpoint working.")


# ============================================================
# TEST 3: INVALID FILE UPLOAD
# ============================================================

def test_invalid_file_upload():

    files = {
        "file": (
            "invalid.txt",
            b"This is not an image",
            "text/plain"
        )
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        files=files,
        timeout=60
    )

    assert response.status_code in (400, 415)

    print("\nInvalid file upload rejected.")


# ============================================================
# TEST 4: VALID IMAGE PREDICTION
# ============================================================

def test_valid_image_prediction():

    response = upload_test_image()

    assert response.status_code == 200

    result = response.json()

    assert "total_detections" in result
    assert "detections" in result

    assert isinstance(
        result["detections"],
        list
    )

    assert isinstance(
        result["total_detections"],
        int
    )

    assert result["total_detections"] >= 0

    assert result["total_detections"] == len(
        result["detections"]
    )

    print("\nImage prediction successful.")


# ============================================================
# TEST 5: PREDICTION RESPONSE STRUCTURE
# ============================================================

def test_prediction_response_structure():

    response = upload_test_image()

    assert response.status_code == 200

    result = response.json()

    assert "total_detections" in result
    assert "detections" in result

    for detection in result["detections"]:

        # Required fields
        assert "class_id" in detection
        assert "class_name" in detection
        assert "confidence" in detection
        assert "box" in detection

        # Class validation
        assert isinstance(
            detection["class_id"],
            int
        )

        assert detection["class_name"] in [
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled-in_scale",
            "scratches"
        ]

        # Confidence validation
        confidence = detection["confidence"]

        assert isinstance(
            confidence,
            (int, float)
        )

        assert 0 <= confidence <= 1

        # Bounding box validation
        box = detection["box"]

        assert isinstance(box, list)
        assert len(box) == 4

        x1, y1, x2, y2 = box

        assert all(
            isinstance(value, (int, float))
            for value in box
        )

        assert 0 <= x1 <= 1
        assert 0 <= y1 <= 1
        assert 0 <= x2 <= 1
        assert 0 <= y2 <= 1

        assert x2 > x1
        assert y2 > y1

    print("\nPrediction response structure validated.")


# ============================================================
# TEST 6: IMAGE FILE EXTENSION VALIDATION
# ============================================================

def test_valid_image_extension():

    assert TEST_IMAGE.suffix.lower() in VALID_EXTENSIONS

    print("\nTest image extension is valid.")
