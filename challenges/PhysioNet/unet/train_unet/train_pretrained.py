# ===================== training mit VORTRAINIERTEM U-Net =====================
import torch
import torch.nn as nn
from pathlib import Path
from dataset import get_dataloader
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import os

# ------------------ Pfade ------------------
cwd = Path(os.path.dirname(os.path.abspath(__file__)))
dat_path = cwd.parent / "train_data" / "train"
image_dir = dat_path / "img"

mask_dirs = {
    "text":  dat_path / "text_mask",
    "curve": dat_path / "curve_mask",
    "grid":  dat_path / "grid_mask",
}

NUM_CLASSES = len(mask_dirs) + 1  # Hintergrund + Klassen

# ------------------ Transforms ------------------
class RandomGrayScale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = F.rgb_to_grayscale(img, num_output_channels=3)
        return img

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=15),
    RandomGrayScale(p=0.2),
    transforms.ToTensor(),
])

# ------------------ DataLoader ------------------
dataloader = get_dataloader(
    image_dir=image_dir,
    mask_dirs=mask_dirs,
    batch_size=1,
    shuffle=True,
    transform=transform,
    resize=(256, 256),
)

# ------------------ Device ------------------
device = "cpu"  # kein CUDA

# ------------------ VORTRAINIERTES U-NET ------------------
model = smp.Unet(
    encoder_name="resnet34",        # vortrainierter Encoder
    encoder_weights="imagenet",     # ImageNet-Gewichte
    in_channels=3,
    classes=NUM_CLASSES,            # 4 Klassen
    activation=None,                # wichtig für CrossEntropyLoss
).to(device)

# ------------------ Loss & Optimizer ------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------ Training ------------------
num_epochs = 5  # Mini-Test

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)   # (B,3,H,W)
        masks = masks.to(device)     # (B,H,W), long

        optimizer.zero_grad()
        outputs = model(images)      # (B,4,H,W)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Batch {batch_idx+1}/{len(dataloader)} | "
            f"Loss: {loss.item():.4f}"
        )

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}  Loss: {epoch_loss:.4f}")

# ------------------ Modell speichern ------------------
save_path = cwd / "unet_resnet34_pretrained.pth"
save_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "encoder": "resnet34",
        "num_classes": NUM_CLASSES,
    },
    save_path,
)

print(f"Training abgeschlossen. Modell gespeichert unter: {save_path}")
