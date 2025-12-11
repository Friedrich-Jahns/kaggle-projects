import torch
import torch.nn as nn
from pathlib import Path
from dataset import get_dataloader
from unet import UNet

# Pfade zu deinen Daten
dat_path = Path.cwd().parent / "train_data" / "train" 
image_dir = dat_path / "img"

mask_dirs = {
    'text': dat_path / "text_mask",
    'curve': dat_path / "curve_mask",
    'grid': dat_path / "grid_mask"
}

# DataLoader
dataloader = get_dataloader(image_dir, mask_dirs, batch_size=1, shuffle=True)

# Gerät
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modell, Loss, Optimizer
model = UNet(in_channels=3, out_channels=3).to(device)
criterion = nn.CrossEntropyLoss()  # für 3 Klassen + Background
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)  # LongTensor mit 0=Background, 1=text, 2=curve, 3=grid

        # Forward
        outputs = model(images)  # [B,3,H,W]

        # Loss
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Modell speichern
torch.save(model.state_dict(), "unet_multiclass_model.pth")
