# Industrial Surface Defect Detection Using CNN

An end-to-end industrial surface defect detection and classification system built using **TensorFlow, CNN, and MobileNetV2 Transfer Learning**. The project identifies and localizes defects in steel surface images and provides predictions through a FastAPI backend and Streamlit frontend.

## Project Overview

Automated steel surface inspection helps identify manufacturing defects and supports quality control. This project uses deep learning and computer vision to detect six categories of surface defects.

## Defect Classes

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | TensorFlow, Keras, CNN |
| Transfer Learning | MobileNetV2 |
| Computer Vision | OpenCV, Pillow |
| Backend | FastAPI, REST API |
| Frontend | Streamlit |
| Deployment | Docker, Docker Compose |
| Testing | Pytest |
| Version Control | Git, GitHub |

## Project Architecture

```text
Input Steel Surface Image
          |
          v
    Streamlit Frontend
          |
          v
      FastAPI Backend
          |
          v
   Image Preprocessing
          |
          v
   Trained Detection Model
          |
          v
  Defect Detection Results
          |
          v
   Display Predictions
```

## Project Structure

```text
Industrial-Surface-Defect-Detection-Using-CNN/
│
├── app/
│   ├── api.py
│   ├── streamlit_app.py
│   ├── streamlit_api_connect.py
│   └── streamlit_controller.py
│
├── data/
├── models/
│   ├── anchors.npy
│   └── improved_experiment/
│       └── best_model_savedmodel/
│
├── notebooks/
├── outputs/
├── results/
├── scripts/
├── src/
│   └── inference/
│       └── predictor.py
│
├── tests/
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── requirements.frontend.txt
├── .dockerignore
└── .gitignore
```

## Model Details

- CNN-based object detection pipeline.
- MobileNetV2 pretrained backbone.
- Input image size: 224 × 224.
- Six industrial surface defect categories.
- Detection output includes class name, confidence score, and bounding box coordinates.

## Run the Project Using Docker

### 1. Clone the repository

```bash
git clone https://github.com/Ravinuthala-Venkata-Siva-Prasad/Industrial-Surface-Defect-Detection-Using-CNN.git
```

### 2. Navigate to the project

```bash
cd Industrial-Surface-Defect-Detection-Using-CNN
```

### 3. Build and start containers

```bash
docker compose up -d --build
```

### 4. Access the applications

- Streamlit Frontend: http://localhost:8501
- FastAPI Backend: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 5. Stop containers

```bash
docker compose down
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend and model status |
| POST | `/predict` | Predict defects from an uploaded image |

## Testing

Run the project tests using:

```bash
pytest -v
```

## Key Features

- Industrial steel surface defect detection.
- Deep learning with TensorFlow and MobileNetV2.
- Image preprocessing and inference pipeline.
- FastAPI REST API integration.
- Streamlit interactive frontend.
- Docker and Docker Compose deployment.
- Automated API testing.

## Future Improvements

- Improve detection precision and localization.
- Expand training data and model evaluation.
- Optimize inference performance.
- Explore cloud deployment and CI/CD automation.

## Author

**Ravinuthala Venkata Siva Prasad**

M.Tech – Artificial Intelligence

GitHub: [Ravinuthala-Venkata-Siva-Prasad](https://github.com/Ravinuthala-Venkata-Siva-Prasad)

LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/ravinuthala-venkata-siva-prasad/)

---

If you find this project useful, consider giving the repository a ⭐.
