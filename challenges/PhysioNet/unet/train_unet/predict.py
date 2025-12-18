import torch
from PIL import Image
import torchvision.transforms as transforms
from unet import UNet
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Konfiguration ---
device = "cuda" if torch.cuda.is_available() else "cpu"

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

dat_path = cwd.parent.parent / 'data_sample'
model_path = "unet_multiclass_model_mini.pth"
# model_path = "unet_resnet34_pretrained.pth"

save_masks = True
# save_dir = Path.cwd() / "pred_masks"
# save_dir.mkdir(exist_ok=True)

# --- Transform für Modell ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# --- Modell laden ---
model = UNet(in_channels=3, out_channels=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# --- Farbmap ---
color_map = np.array([
    [0, 0, 0],       # Background schwarz
    [255, 0, 0],     # Text rot
    [0, 255, 0],     # Curve grün
    [0, 0, 255]      # Grid blau
], dtype=np.uint8)

# --- Schleife über Bilder ---
for folder_name in os.listdir(dat_path):
    image_path = dat_path / folder_name / f"{folder_name}-0001.png"
    if not image_path.exists():
        continue

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Vorhersage
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        for i in range(4):
            plt.imshow(output[0,i].cpu(), cmap='gray')
            plt.title(f"Channel {i}")
            plt.show()

    # --- Resize Maske auf Originalgröße ---
    mask_resized = Image.fromarray(pred_mask.astype(np.uint8))
    mask_resized = mask_resized.resize(image.size, resample=Image.NEAREST)
    pred_mask = np.array(mask_resized, dtype=np.uint8)  # sicherstellen uint8 0-3

    # Farbige Maske
    mask_rgb = color_map[pred_mask]

    # --- Anzeige ---
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Originalbild")

    plt.subplot(1,2,2)
    plt.imshow(mask_rgb)
    plt.axis("off")
    plt.title("Segmentierungsmaske")
    plt.show()

    # --- Maske speichern ---
    # if save_masks:
    #     Image.fromarray(mask_rgb).save(save_dir / f"{folder_name}_mask.png")
