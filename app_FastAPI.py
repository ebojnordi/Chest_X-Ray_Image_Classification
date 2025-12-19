from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import shutil

# Setup
app = FastAPI()
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # NORMAL, PNEUMONIA
model.load_state_dict(torch.load("pneumonia_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Classes
classes = ["NORMAL", "PNEUMONIA"]

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    # Save uploaded file
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and preprocess
    image = Image.open(filepath).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        prediction = classes[preds.item()]

    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
