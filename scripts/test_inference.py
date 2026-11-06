import sys
from pathlib import Path

# =====================================================
# PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import DefectPredictor
from src.inference.visualization import visualize_predictions


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# =====================================================
# FIND DATASET IMAGES
# =====================================================

def find_training_images():

    image_folder = (
        PROJECT_ROOT
        / "data"
        / "NEU-DET"
        / "train"
        / "images"
    )

    if not image_folder.exists():

        raise FileNotFoundError(
            f"Training image directory not found: {image_folder}"
        )

    image_files = [
        file
        for file in image_folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_files.sort()

    if not image_files:

        raise FileNotFoundError(
            f"No images found inside: {image_folder}"
        )

    return image_files


# =====================================================
# PRINT PREDICTION RESULTS
# =====================================================

def display_results(result):

    print("\n" + "=" * 55)
    print("INDUSTRIAL SURFACE DEFECT DETECTION")
    print("=" * 55)

    print("\nImage:", result["image"])

    print(
        "\nTotal detections:",
        result["total_detections"]
    )

    if not result["detections"]:

        print("\nNo defects detected.")

        return

    for index, detection in enumerate(
        result["detections"],
        start=1
    ):

        print("\nDetection:", index)

        print(
            "Defect Class:",
            detection["class_name"]
        )

        print(
            "Confidence:",
            f'{detection["confidence"]:.4f}'
        )

        print(
            "Bounding Box:",
            detection["box"]
        )


# =====================================================
# MAIN INFERENCE FUNCTION
# =====================================================

def main():

    print("\nStarting inference pipeline...")

    # Step 1: Find images.
    image_files = find_training_images()

    print(
        f"\nTotal training images found: {len(image_files)}"
    )

    # Step 2: Select an image.
    image_path = image_files[0]

    print("\nSelected image:")
    print(image_path)

    # Step 3: Load trained model.
    predictor = DefectPredictor(
        confidence_threshold=0.25,
        nms_iou_threshold=0.50
    )

    # Step 4: Generate predictions.
    print("\nGenerating predictions...")

    result = predictor.predict(image_path)

    # Step 5: Display prediction results.
    display_results(result)

    # Step 6: Save visualization.
    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "inference_result.jpg"
    )

    visualize_predictions(
        image_path=image_path,
        detections=result["detections"],
        output_path=output_path
    )

    print("\n" + "=" * 55)
    print("INFERENCE PIPELINE COMPLETED")
    print("=" * 55)


# =====================================================
# EXECUTION
# =====================================================

if __name__ == "__main__":

    main()