from pathlib import Path
from dataset import get_dataloader
import torch

def test_dataloader():
    # Pfade zu den Daten
    dat_path = Path.cwd().parent / "train_data" / "train"
    image_dir = dat_path / "img"

    mask_dirs = {
        'text': dat_path  / "text_mask",
        'curve': dat_path  / "curve_mask",
        'grid': dat_path  / "grid_mask"
    }

    # DataLoader mit Batch=1
    dataloader = get_dataloader(image_dir, mask_dirs, batch_size=1, shuffle=False)

    # Ein Batch laden
    for i, (images, masks) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print("  Image shape:", images.shape)  # [B, 3, H, W]
        print("  Mask shape: ", masks.shape)   # [B, H, W]
        print("  Mask unique classes:", torch.unique(masks))
        
        # Optional: nur 1 Batch testen
        break

if __name__ == "__main__":
    test_dataloader()
