import os
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf

from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

CLASS_MAP = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


def load_annotation(xml_path):

    root = ET.parse(xml_path).getroot()

    boxes = []
    labels = []

    size = root.find("size")

    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    for obj in root.findall("object"):

        class_name = obj.findtext("name")

        if class_name not in CLASS_MAP:
            continue

        bbox = obj.find("bndbox")

        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        xmin = np.clip(xmin / width, 0, 1)
        ymin = np.clip(ymin / height, 0, 1)
        xmax = np.clip(xmax / width, 0, 1)
        ymax = np.clip(ymax / height, 0, 1)

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])

        labels.append(CLASS_MAP[class_name])

    return (
        np.array(boxes, dtype=np.float32),
        np.array(labels, dtype=np.int32)
    )


def load_records(dataset_path, split):

    image_dir = Path(dataset_path) / split / "images"

    annotation_dir = Path(dataset_path) / split / "annotations"

    records = []

    for image_path in image_dir.rglob("*"):

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png"
        ]:
            continue

        xml_path = annotation_dir / (
            image_path.stem + ".xml"
        )

        if not xml_path.exists():
            continue

        boxes, labels = load_annotation(xml_path)

        if len(boxes) == 0:
            continue

        records.append({
            "image": str(image_path),
            "boxes": boxes,
            "labels": labels,
            "stratify": int(labels[0])
        })

    if not records:
        raise ValueError(
            f"No annotated images found in {split}"
        )

    return records


def create_grid_target(
    boxes,
    labels,
    grid_size=7
):

    target = np.zeros(
        (grid_size, grid_size, 11),
        dtype=np.float32
    )

    areas = np.full(
        (grid_size, grid_size),
        -1.0
    )

    collisions = 0

    for box, label in zip(boxes, labels):

        xmin, ymin, xmax, ymax = box

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        gx = min(int(cx * grid_size), grid_size - 1)
        gy = min(int(cy * grid_size), grid_size - 1)

        area = (xmax - xmin) * (ymax - ymin)

        if areas[gy, gx] >= 0:

            collisions += 1

            if area <= areas[gy, gx]:
                continue

        areas[gy, gx] = area

        target[gy, gx, 0] = 1

        target[gy, gx, 1:5] = [
            cx,
            cy,
            xmax - xmin,
            ymax - ymin
        ]

        target[gy, gx, 5:] = 0

        target[gy, gx, 5 + label] = 1

    return target, collisions


def create_dataset(
    records,
    image_size=224,
    grid_size=7,
    batch_size=16,
    training=False
):

    def generator():

        for record in records:

            image = Image.open(
                record["image"]
            ).convert("RGB")

            image = image.resize(
                (image_size, image_size)
            )

            image = np.array(
                image,
                dtype=np.float32
            )

            target, _ = create_grid_target(
                record["boxes"],
                record["labels"],
                grid_size
            )

            if training and np.random.random() < 0.5:

                image = np.fliplr(image).copy()

                target = target[:, ::-1, :].copy()

                obj = target[..., 0] > 0

                target[..., 1][obj] = (
                    1 - target[..., 1][obj]
                )

            yield image, target

    signature = (

        tf.TensorSpec(
            (image_size, image_size, 3),
            tf.float32
        ),

        tf.TensorSpec(
            (grid_size, grid_size, 11),
            tf.float32
        )

    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=signature
    )

    if training:

        dataset = dataset.shuffle(
            min(len(records), 512)
        )

    return dataset.batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )


def get_dataset_splits(dataset_path):

    train_records = load_records(
        dataset_path,
        "train"
    )

    test_records = load_records(
        dataset_path,
        "validation"
    )

    labels = [
        record["stratify"]
        for record in train_records
    ]

    train_records, val_records = train_test_split(

        train_records,

        test_size=0.20,

        random_state=42,

        stratify=labels

    )

    return train_records, val_records, test_records