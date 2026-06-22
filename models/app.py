
from fastapi import FastAPI
import torch
import timm
import json
from fastapi import UploadFile, File
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

app = FastAPI()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
with open(
    "models/class_names.json",
    "r"
) as f:
    class_names = json.load(f)

# Load disease information
with open(
    "models/disease_info.json",
    "r"
) as f:
    disease_info = json.load(f)

# Load model
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=38
)

model.load_state_dict(
    torch.load(
    "models/crop_disease_model.pth",
    map_location=device
    )
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/")
def home():
    return {
        "message": "Crop Disease API is running!",
        "classes": len(class_names),
        "diseases": len(disease_info)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    class_name = class_names[predicted.item()]

    info = disease_info[class_name]

    return {
    "crop": info["crop"],
    "disease": info["disease"],
    "status": info["status"],
    "confidence": round(confidence.item() * 100, 2),
    "remedy": info.get("remedy", [])
    }
