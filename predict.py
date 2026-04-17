import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import sys
import os

# ── Device ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Model Loader ─────────────────────────────────────────────────────
def load_model(weights_path):
    model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2),
        nn.LogSoftmax(dim=1)
    )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✓ Model loaded from: {weights_path}\n")
    return model

# ── Transform ────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class labels (adjust if needed)
class_names = ['fake', 'real']

# ── Predict Single Image ─────────────────────────────────────────────
def predict_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"✗ File not found: {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        probs = F.softmax(output, dim=1)
        conf, predicted = torch.max(probs, 1)

    label = class_names[predicted.item()]
    confidence = conf.item() * 100
    fake_prob = probs[0][0].item() * 100
    real_prob = probs[0][1].item() * 100

    print(f"Image      : {os.path.basename(image_path)}")
    print(f"Prediction : {label.upper()}")
    print(f"Confidence : {confidence:.2f}%")
    print(f"  → Real   : {real_prob:.2f}%")
    print(f"  → Fake   : {fake_prob:.2f}%")
    print("-" * 40)

# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [image_path2 ...]")
        print("Example: python predict.py face.jpg")
        sys.exit(1)

    model = load_model("best_model.pth")

    for image_path in sys.argv[1:]:
        predict_image(model, image_path)