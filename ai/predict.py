import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
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

def predict_image(image_path, return_max=False):

    # Load Image
    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        density = model(tensor)

    # Remove negative values
    density = torch.relu(density)

    max_val = float(density.max().item())
    sum_val = float(density.sum().item())

    # Background sensor noise suppression:
    # If the peak spatial density is below 0.0018, it represents empty background texture.
    if max_val < 0.0018:
        count = 0.0
    else:
        # Subtract background baseline noise floor:
        noise_floor = min(0.0003, max_val * 0.08)
        cleaned_density = torch.clamp(density - noise_floor, min=0.0)
        count = float(cleaned_density.sum().item())
        # Scale slightly to match true crowd calibration
        scale_factor = sum_val / max(1.0, count) if count > 0 else 1.0
        count = min(sum_val, count * min(1.3, scale_factor))

    rounded_count = int(round(count))

    print("Maximum :", max_val)
    print("Minimum :", density.min().item())
    print("Raw Sum :", sum_val)
    print("Calibrated Count :", rounded_count)

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

    name, _ = os.path.splitext(filename)

    heatmap_name = f"{name}_heatmap.png"

    heatmap_path = os.path.join(
        os.path.dirname(image_path),
        heatmap_name
    )

    cv2.imwrite(
        heatmap_path,
        overlay
    )

    if return_max:
        return rounded_count, heatmap_path, max_val

    return rounded_count, heatmap_path