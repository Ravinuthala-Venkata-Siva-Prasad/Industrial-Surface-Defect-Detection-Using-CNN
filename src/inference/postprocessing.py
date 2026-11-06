import numpy as np
import tensorflow as tf


GRID_SIZE = 7
NUM_ANCHORS = 5
NUM_CLASSES = 6

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def softmax(values):
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def decode_predictions(
    predictions,
    anchors,
    confidence_threshold=0.25,
    nms_iou_threshold=0.50,
    max_detections=100,
):
    """
    Decode predictions from the trained detector.

    Expected prediction shape:
        (1, 7, 7, 5, 11) or (7, 7, 5, 11)

    Prediction channels:
        0:4  -> tx, ty, tw, th
        4    -> objectness logit
        5:11 -> six class logits

    Returns:
        List of detection dictionaries.
    """

    predictions = np.asarray(predictions, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)

    if predictions.ndim == 5:
        if predictions.shape[0] != 1:
            raise ValueError(
                "decode_predictions currently supports one image at a time."
            )
        predictions = predictions[0]

    expected_shape = (
        GRID_SIZE,
        GRID_SIZE,
        NUM_ANCHORS,
        5 + NUM_CLASSES,
    )

    if predictions.shape != expected_shape:
        raise ValueError(
            f"Expected prediction shape {expected_shape}, "
            f"received {predictions.shape}."
        )

    if anchors.shape != (NUM_ANCHORS, 2):
        raise ValueError(
            f"Expected anchors shape {(NUM_ANCHORS, 2)}, "
            f"received {anchors.shape}."
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError("Model predictions contain NaN or infinity.")

    if not np.all(np.isfinite(anchors)) or np.any(anchors <= 0):
        raise ValueError("Anchors must be finite positive values.")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    if not 0.0 <= nms_iou_threshold <= 1.0:
        raise ValueError("nms_iou_threshold must be between 0 and 1.")

    boxes = []
    scores = []
    labels = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            for anchor_idx in range(NUM_ANCHORS):

                pred = predictions[row, col, anchor_idx]

                tx, ty, tw, th = pred[:4]

                objectness = float(sigmoid(pred[4]))

                class_probabilities = softmax(pred[5:])
                class_id = int(np.argmax(class_probabilities))

                confidence = float(
                    objectness * class_probabilities[class_id]
                )

                if confidence < confidence_threshold:
                    continue

                # tx and ty are direct cell offsets in this model.
                tx = float(np.clip(tx, 0.0, 1.0))
                ty = float(np.clip(ty, 0.0, 1.0))

                center_x = (col + tx) / GRID_SIZE
                center_y = (row + ty) / GRID_SIZE

                width = float(
                    anchors[anchor_idx, 0]
                    * np.exp(np.clip(tw, -10.0, 10.0))
                )

                height = float(
                    anchors[anchor_idx, 1]
                    * np.exp(np.clip(th, -10.0, 10.0))
                )

                x1 = float(np.clip(center_x - width / 2.0, 0.0, 1.0))
                y1 = float(np.clip(center_y - height / 2.0, 0.0, 1.0))
                x2 = float(np.clip(center_x + width / 2.0, 0.0, 1.0))
                y2 = float(np.clip(center_y + height / 2.0, 0.0, 1.0))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                scores.append(confidence)
                labels.append(class_id)

    if not boxes:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    detections = []

    # Perform NMS independently for each class.
    for class_id in range(NUM_CLASSES):

        indices = np.where(labels == class_id)[0]

        if len(indices) == 0:
            continue

        class_boxes = boxes[indices]
        class_scores = scores[indices]

        # TensorFlow NMS expects [y1, x1, y2, x2].
        nms_boxes = class_boxes[:, [1, 0, 3, 2]]

        selected_indices = tf.image.non_max_suppression(
            boxes=nms_boxes,
            scores=class_scores,
            max_output_size=max_detections,
            iou_threshold=nms_iou_threshold,
            score_threshold=confidence_threshold,
        ).numpy()

        for selected_index in selected_indices:
            box = class_boxes[selected_index]

            detections.append({
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[class_id],
                "confidence": float(class_scores[selected_index]),
                "box": [float(value) for value in box],
            })

    detections.sort(
        key=lambda detection: detection["confidence"],
        reverse=True,
    )

    return detections[:max_detections]