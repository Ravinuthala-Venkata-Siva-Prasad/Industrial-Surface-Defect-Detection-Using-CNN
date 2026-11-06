
from streamlit_api_connect import (
    check_api_health,
    predict_image
)


# ==========================================
# BACKEND STATUS
# ==========================================

def get_backend_status():

    return check_api_health()


# ==========================================
# PROCESS IMAGE
# ==========================================

def process_uploaded_image(
    image_bytes,
    filename
):

    result = predict_image(
        image_bytes,
        filename
    )

    return result
