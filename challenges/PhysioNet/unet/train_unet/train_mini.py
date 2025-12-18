import torch
import torch.nn as nn
from pathlib import Path
from dataset import get_dataloader
from unet import UNet
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import os

# Pfade zu deinen Daten
cwd = Path(os.path.dirname(os.path.abspath(__file__)))
dat_path = cwd.parent / "train_data" / "train"
image_dir = dat_path / "img"

mask_dirs = {
    'text': dat_path  / "text_mask",
    'curve': dat_path  / "curve_mask",
    'grid': dat_path  / "grid_mask"
}

# Transform: Resize + ToTensor
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])
class RandomGrayScale(object):
    """Wandelt ein Bild zufällig in Graustufen um"""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = F.rgb_to_grayscale(img, num_output_channels=3)
        return img

transform = transforms.Compose([
    transforms.Resize((256, 256)),             # Resize für CPU
    transforms.RandomHorizontalFlip(p=0.5),    # Horizontal flip
    transforms.RandomVerticalFlip(p=0.1),      # Vertical flip
    transforms.RandomRotation(degrees=15),     # kleine Rotation
    RandomGrayScale(p=0.2),                    # zufällig Graustufen
    transforms.ToTensor()                       # in Tensor umwandeln
])


# DataLoader
dataloader = get_dataloader(
    image_dir,
    mask_dirs,
    batch_size=1,
    shuffle=True,
    transform=transform,
    resize=(256,256)  # auch im Dataset resize
)

# Gerät
device = "cpu"  # CPU-Training, da kein CUDA verfügbar

# Modell, Loss, Optimizer
model = UNet(in_channels=3, out_channels=4).to(device)
criterion = nn.CrossEntropyLoss()  # 4 Klassen: 0=BG, 1=text, 2=curve, 3=grid
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training (Mini-Test)
num_epochs = 5  # für schnellen Test
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)  # LongTensor: 0=Background, 1=text, 2=curve, 3=grid

        # Forward
        outputs = model(images)  # [B,4,H,W]

        # Loss
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Durchschnitts-Loss: {epoch_loss:.4f}")

# Modell speichern
torch.save(model.state_dict(), "unet_multiclass_model_mini.pth")
print("Mini-Training abgeschlossen und Modell gespeichert.")
