from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time

# -------------------
# Setup
# -------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Model
# -------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model/pneumonia_classifier.pth", map_location=device))
model.to(device)
model.eval()

classes = ["NORMAL", "PNEUMONIA"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -------------------
# Routes
# -------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    processing_time = int((time.time() - start_time) * 1000)

    return JSONResponse({
        "label": classes[pred.item()],
        "confidence": round(conf.item() * 100, 2),
        "processing_time_ms": processing_time,
        "description": (
            "Signs of pneumonia detected in the chest X-ray."
            if classes[pred.item()] == "PNEUMONIA"
            else "No signs of pneumonia detected. The X-ray appears normal."
        )
    })
