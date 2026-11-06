from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def visualize_predictions(
    image_path,
    detections,
    output_path=None
):
    """
    Visualize industrial surface defect predictions.

    Args:
        image_path: Original image path.
        detections: List of predicted detections.
        output_path: Optional path to save visualization.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    image_width, image_height = image.size

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.imshow(image)

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        # Convert normalized coordinates to image pixels.
        x1 *= image_width
        x2 *= image_width
        y1 *= image_height
        y2 *= image_height

        box_width = x2 - x1
        box_height = y2 - y1

        rectangle = patches.Rectangle(
            (x1, y1),
            box_width,
            box_height,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rectangle)

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        label = f"{class_name}: {confidence:.2%}"

        ax.text(
            x1,
            max(0, y1 - 5),
            label,
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox={
                "facecolor": "red",
                "alpha": 0.8,
                "pad": 3
            }
        )

    ax.set_title(
        f"Industrial Surface Defect Detection | "
        f"Detections: {len(detections)}",
        fontsize=14
    )

    ax.axis("off")

    plt.tight_layout()

    if output_path is not None:

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        print(f"Visualization saved: {output_path}")

    plt.show()
    plt.close(fig)