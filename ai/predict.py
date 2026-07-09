import os
import cv2
from networkx import density
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from ai.models.complete_model import CrowdCounter


# ---------------------------------------
# Device
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------
# Load Model
# ---------------------------------------

model = CrowdCounter().to(device)

model_path = os.path.join(
    os.path.dirname(__file__),
    "best_model.pth"
)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)

model.eval()


# ---------------------------------------
# Image Transform
# ---------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------
# Prediction Function
# ---------------------------------------

def predict_image(image_path):

    # Load Image
    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        density = model(tensor)

    # Remove negative values
    density = torch.relu(density)

    print("Maximum :", density.max().item())
    print("Minimum :", density.min().item())
    print("Sum :", density.sum().item())

    # Crowd Count
    count = density.sum().item()

    # Convert to numpy
    density_map = density.squeeze().cpu().numpy()

    # Normalize
    density_map = cv2.normalize(
        density_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    density_map = density_map.astype(np.uint8)

    # Heatmap
    heatmap = cv2.applyColorMap(
        density_map,
        cv2.COLORMAP_TURBO
    )

    # Convert original image to OpenCV format
    original = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    # Overlay heatmap
    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    # Save heatmap
    filename = os.path.basename(image_path)

    name, ext = os.path.splitext(filename)

    heatmap_name = f"{name}_heatmap{ext}"

    heatmap_path = os.path.join(
        os.path.dirname(image_path),
        heatmap_name
    )

    cv2.imwrite(
        heatmap_path,
        overlay
    )

    return int(round(count)), heatmap_path