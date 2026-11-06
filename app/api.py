
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference.predictor import DefectPredictor


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

UPLOAD_DIR = PROJECT_ROOT / "outputs" / "api_uploads"

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

MAX_FILE_SIZE = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = None


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Industrial Surface Defect Detection API",
    description="CNN-based industrial surface defect detection using TensorFlow and MobileNetV2",
    version="1.0.0"
)


# ==========================================
# LOAD MODEL
# ==========================================

@app.on_event("startup")
def load_model():

    global predictor

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info("Loading TensorFlow model...")

    predictor = DefectPredictor(
        confidence_threshold=0.25,
        nms_iou_threshold=0.50
    )

    logger.info("Model loaded successfully.")


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def root():

    return {
        "message": "Industrial Surface Defect Detection API",
        "status": "running",
        "version": "1.0.0"
    }


# ==========================================
# HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy" if predictor is not None else "not_ready",
        "model_loaded": predictor is not None
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
async def predict_defect(
    file: UploadFile = File(...)
):

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not ready."
        )

    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    temp_path = UPLOAD_DIR / (
        f"{uuid.uuid4().hex}{extension}"
    )

    try:

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        if len(image_bytes) > MAX_FILE_SIZE:

            raise HTTPException(
                status_code=413,
                detail="Maximum image size is 10 MB."
            )

        with temp_path.open("wb") as buffer:
            buffer.write(image_bytes)

        result = predictor.predict(
            str(temp_path)
        )

        return {
            "filename": file.filename,
            "total_detections": result["total_detections"],
            "detections": result["detections"]
        }

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check backend logs."
        ) from exc

    finally:

        if temp_path.exists():
            temp_path.unlink()

        await file.close()
