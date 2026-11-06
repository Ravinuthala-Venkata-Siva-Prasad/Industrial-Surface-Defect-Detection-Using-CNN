from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = (224, 224)


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """
    Load and preprocess an image for the trained MobileNetV2 detector.

    Returns:
        np.ndarray: Image batch with shape (1, 224, 224, 3).
    """
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = image.resize(
                IMAGE_SIZE,
                resample=Image.Resampling.BILINEAR
            )
            image = np.asarray(image, dtype=np.float32)

    except Exception as exc:
        raise ValueError(
            f"Could not read image {image_path}: {exc}"
        ) from exc

    # Match MobileNetV2 preprocess_input.
    image = (image / 127.5) - 1.0

    # Add batch dimension.
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)