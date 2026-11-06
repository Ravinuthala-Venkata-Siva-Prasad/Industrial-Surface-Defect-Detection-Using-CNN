
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.inference.preprocessing import preprocess_image
from src.inference.postprocessing import decode_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Use the Docker-compatible TensorFlow SavedModel
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "improved_experiment"
    / "best_model_savedmodel"
)

ANCHORS_PATH = PROJECT_ROOT / "models" / "anchors.npy"


class DefectPredictor:
    """Production inference interface for industrial defect detection."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        anchors_path=ANCHORS_PATH,
        confidence_threshold=0.25,
        nms_iou_threshold=0.50,
    ):

        self.model_path = Path(model_path)
        self.anchors_path = Path(anchors_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        if not self.anchors_path.is_file():
            raise FileNotFoundError(
                f"Anchor file not found: {self.anchors_path}"
            )

        self.anchors = np.load(self.anchors_path).astype(np.float32)

        if self.anchors.shape != (5, 2):
            raise ValueError(
                f"Expected anchors with shape (5, 2), "
                f"received {self.anchors.shape}."
            )

        # Load exported TensorFlow SavedModel
        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False,
        )

        expected_output = (7, 7, 5, 11)

        if tuple(self.model.output_shape[1:]) != expected_output:
            raise ValueError(
                f"Unexpected model output shape: {self.model.output_shape}. "
                f"Expected (None, 7, 7, 5, 11)."
            )

        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

        print("Model loaded successfully.")
        print(f"Model: {self.model_path}")
        print(f"Anchors: {self.anchors.shape}")

    def predict(self, image_path):

        image_path = Path(image_path)

        image_batch = preprocess_image(image_path)

        predictions = self.model.predict(
            image_batch,
            verbose=0,
        )

        detections = decode_predictions(
            predictions=predictions,
            anchors=self.anchors,
            confidence_threshold=self.confidence_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
        )

        return {
            "image": str(image_path),
            "total_detections": len(detections),
            "detections": detections,
        }
