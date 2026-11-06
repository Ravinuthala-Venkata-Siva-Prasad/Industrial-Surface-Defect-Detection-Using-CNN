
import cv2
import matplotlib.pyplot as plt

from src.data.dataset import get_dataset_splits

DATASET_PATH = "data/NEU-DET"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

train, val, test = get_dataset_splits(DATASET_PATH)

sample = train[0]

# Read image
image_path = sample["image"]
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Cannot read image: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get bounding boxes and labels
boxes = sample["boxes"]
labels = sample["labels"]

height, width = image.shape[:2]

# Draw bounding boxes
for box, label in zip(boxes, labels):

    x_min, y_min, x_max, y_max = box

    # Handle normalized coordinates
    if max(abs(x_min), abs(y_min), abs(x_max), abs(y_max)) <= 1.0:
        x_min *= width
        x_max *= width
        y_min *= height
        y_max *= height

    x_min, y_min, x_max, y_max = map(
        int, [x_min, y_min, x_max, y_max]
    )

    class_name = CLASS_NAMES[int(label)]

    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_max, y_max),
        (255, 0, 0),
        2
    )

    cv2.putText(
        image,
        class_name,
        (x_min, max(y_min - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )

# Display
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title("NEU-DET Defect Detection")
plt.axis("off")
plt.show()

print("Image:", image_path)
print("Boxes:", boxes)
print("Labels:", labels)
print("Defect classes:", [CLASS_NAMES[int(x)] for x in labels])
