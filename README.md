# Pneumonia X-ray Classification System

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green.svg)](https://fastapi.tiangolo.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.3.3-blue.svg)](https://tailwindcss.com/)

An end-to-end **deep learning web application** for detecting **pneumonia from chest X-ray images** using a **ResNet-18** model and a **FastAPI-based web interface**.

Users can upload a chest X-ray image through a simple UI and receive an instant prediction indicating whether the image is **NORMAL** or **PNEUMONIA**.

---

## Features

- Deep Learning model using **ResNet-18**
- Transfer learning–ready architecture
- Web application built with **FastAPI**
- Image upload with real-time preview
- Model performance metrics (Accuracy, Precision, Recall, F1-score, Balanced accuracy)
- CPU/GPU automatic support

---

## Project Structure

```text
root/
├── app_FastAPI.py                 # FastAPI backend
├── model
│   └── pneumonia_classifier.pth   # Trained model weights
├── train.py                       # Model training script
├── templates/
│   └── index.html                 # Frontend UI
├── static/
│   ├── styles.css                 # Styling
│   ├── script.js                  # Frontend logic
│   └── uploads/                   # Uploaded images
├── requirements.txt
└── README.md
```
## Model Details

- Architecture: ResNet-18
- Framework: PyTorch
- Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Input Size: 224 × 224 RGB
- Classes: NORMAL, PNEUMONIA

## Training Configuration

- Epochs: 10
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Image normalization: ImageNet mean & standard deviation

## Model Performance

| Dataset    | Accuracy | Precision | Recall  | F1-score | Balanced Accuracy |
|------------|----------|-----------|---------|----------|-------------------|
| Validation | 93.75%   | 88.89%    | 100.00% | 94.12%   | 93.75%            |
| Test       | 74.52%   | 71.04%    | 100.00% | 83.07%   | 66.03%            |

Model trained for 10 epochs.

## Web Application

### Backend
- FastAPI
- PyTorch & TorchVision
- PIL for image preprocessing

### Frontend
- HTML + Tailwind CSS
- JavaScript for image preview and UI interactions

## How to Run the Project

### Install Dependencies
```text
- pip install -r requirements.txt
- python train.py # to create model weights (pneumonia_classifier.pth)
```
  
### Start the FastAPI Server
```text
- uvicorn app_FastAPI:app --reload
```

### Open in Browser
- http://127.0.0.1:8000
