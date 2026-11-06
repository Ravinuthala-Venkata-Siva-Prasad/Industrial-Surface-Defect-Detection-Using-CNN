
from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from streamlit_controller import (
    get_backend_status,
    process_uploaded_image
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Industrial Surface Defect Detection",
    page_icon="🔍",
    layout="wide"
)


# ==========================================
# CONSTANTS
# ==========================================

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


# ==========================================
# DRAW BOUNDING BOXES
# ==========================================

def draw_predictions(image, detections):

    image = image.convert("RGB")

    draw = ImageDraw.Draw(image)

    width, height = image.size

    try:

        font = ImageFont.truetype(
            "arial.ttf",
            16
        )

    except OSError:

        font = ImageFont.load_default()

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        x1 = int(x1 * width)
        y1 = int(y1 * height)

        x2 = int(x2 * width)
        y2 = int(y2 * height)

        class_name = detection["class_name"]

        confidence = detection["confidence"]

        label = (
            f"{class_name}: "
            f"{confidence:.2%}"
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        text_bbox = draw.textbbox(
            (x1, y1),
            label,
            font=font
        )

        draw.rectangle(
            text_bbox,
            fill="red"
        )

        draw.text(
            (x1, y1),
            label,
            fill="white",
            font=font
        )

    return image


# ==========================================
# DISPLAY PREDICTION RESULTS
# ==========================================

def display_prediction_results(
    result,
    original_image
):

    detections = result["detections"]

    predicted_image = draw_predictions(
        original_image.copy(),
        detections
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            original_image,
            use_container_width=True
        )

    with col2:

        st.subheader("Detection Results")

        st.image(
            predicted_image,
            use_container_width=True
        )

    st.divider()

    st.subheader("Prediction Summary")

    metric1, metric2 = st.columns(2)

    metric1.metric(
        "Total Detections",
        result["total_detections"]
    )

    metric2.metric(
        "Unique Defect Classes",
        len({
            item["class_name"]
            for item in detections
        })
    )

    if detections:

        st.subheader("Detected Defects")

        detection_data = []

        for item in detections:

            detection_data.append({
                "Class": item["class_name"],
                "Confidence (%)": round(
                    item["confidence"] * 100,
                    2
                ),
                "Bounding Box": item["box"]
            })

        st.dataframe(
            detection_data,
            use_container_width=True
        )

    else:

        st.info(
            "No defects detected in this image."
        )

    # Download prediction image

    output = BytesIO()

    predicted_image.save(
        output,
        format="PNG"
    )

    st.download_button(
        label="Download Detection Image",
        data=output.getvalue(),
        file_name="defect_detection_result.png",
        mime="image/png"
    )


# ==========================================
# HEADER
# ==========================================

st.title(
    "Industrial Surface Defect Detection"
)

st.markdown(
    """
    ### CNN-Based Surface Defect Detection and Classification

    Upload an industrial surface image to detect and
    classify six types of industrial surface defects.
    """
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Application Information")

st.sidebar.subheader(
    "Supported Defect Classes"
)

for class_name in CLASS_NAMES:

    st.sidebar.write(
        f"• {class_name}"
    )


# ==========================================
# BACKEND STATUS
# ==========================================

with st.sidebar:

    st.divider()

    st.subheader("Model Service Status")

    backend_status = get_backend_status()

    if (
        backend_status.get("status") == "healthy"
        and backend_status.get("model_loaded") is True
    ):

        st.success("Prediction service ready")

    elif backend_status.get("status") == "healthy":

        st.warning("Model is not ready")

    else:

        st.error("Prediction service unavailable")


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Industrial Surface Image",
    type=["jpg", "jpeg", "png", "bmp"]
)


# ==========================================
# IMAGE PREVIEW
# ==========================================

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    try:

        original_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as exc:

        st.error(
            f"Invalid image: {exc}"
        )

        st.stop()

    st.subheader("Uploaded Image")

    st.image(
        original_image,
        use_container_width=True
    )

    st.success(
        f"Image uploaded: {uploaded_file.name}"
    )

    st.divider()

    # ======================================
    # PREDICTION BUTTON
    # ======================================

    if st.button(
        "Detect Surface Defects",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Analyzing image with the trained model..."
            ):

                result = process_uploaded_image(
                    image_bytes,
                    uploaded_file.name
                )

            st.success(
                "Prediction completed successfully."
            )

            display_prediction_results(
                result,
                original_image
            )

        except Exception as exc:

            st.error(
                f"Prediction failed: {exc}"
            )

else:

    st.info(
        "Upload an industrial surface image to start detection."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Industrial Surface Defect Detection | "
    "TensorFlow | MobileNetV2 | FastAPI | Streamlit"
)
