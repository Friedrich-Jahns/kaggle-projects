import numpy as np # linear algebra
import pandas as pd
from PIL import Image, ImageDraw
import random
from pathlib import Path
import os

# # Basisverzeichnis für die generierten Trainingsdaten anlegen
# base = Path("/kaggle/working/ekg_data")
# (base / "images").mkdir(parents=True, exist_ok=True)
# (base / "masks").mkdir(parents=True, exist_ok=True)


cwd = Path(os.path.dirname(os.path.abspath(__file__)))
dat_path = cwd.parent / "train_data" / "train"
image_dir = dat_path / "img"



image_path = dat_path / "img"
mask_path = dat_path / "curve_mask"


# image_folder = Path("/kaggle/working/ekg_data/images")
# mask_folder  = Path("/kaggle/working/ekg_data/masks")

def check_and_fix_mask(mask_file, img_file):
    mask = Image.open(mask_file).convert("L")  # 1-Kanal
    arr = np.array(mask)
    
    # Alle Pixel > 0 auf 1 setzen
    arr = (arr > 0).astype(np.uint8)
    
    # Prüfen: gleiche Größe wie das Bild
    img = Image.open(img_file)
    if arr.shape != (img.height, img.width):
        arr = np.array(Image.fromarray(arr).resize((img.width, img.height)))
        arr = (arr > 0).astype(np.uint8)
    
    # Maske zurückspeichern
    Image.fromarray(arr).save(mask_file)
    
    # Validierung
    unique = np.unique(arr)
    if not set(unique).issubset({0,1}):
        raise ValueError(f"Maskenwerte außer 0/1 in {mask_file}")
    if arr.shape != (img.height, img.width):
        raise ValueError(f"Maskengröße stimmt nicht mit Bild überein: {mask_file}")

# Alle Masken prüfen
for img_file in image_path.glob("*.png"):
    mask_file = mask_path / img_file.name
    if not mask_file.exists():
        raise FileNotFoundError(f"Maskendatei fehlt für {img_file.name}")
    check_and_fix_mask(mask_file, img_file)

print("✅ Alle Masken sind jetzt 1-Kanal, Werte 0/1 und passen zu den Bildern.")



data_path = "/kaggle/working/ekg_data"

# print(os.listdir(data_path))

# image_path = f"{data_path}/images"
# mask_path = f"{data_path}/masks"

print("Bilder:", len(os.listdir(image_path)))
print("Masken:", len(os.listdir(mask_path)))




from fastai.vision.all import *

# Ordnerpfade
# data_path = Path("/kaggle/working/ekg_data")

# path = Path("/kaggle/working/ekg_data/images")
# mask_path = Path("/kaggle/working/ekg_data/masks")

def get_mask(fn):
    return mask_path/fn.name

dblock = DataBlock(
    blocks=(ImageBlock, MaskBlock(codes=[0,1])),
    get_items=get_image_files,
    get_y=get_mask,
    splitter=RandomSplitter(),
    item_tfms=Resize(512),
    batch_tfms=aug_transforms()
)

dls = dblock.dataloaders(image_dir, bs=4)

dls.show_batch(max_n=4)





learn = unet_learner(dls, resnet34, metrics=Dice())
learn.fine_tune(5)


learn.show_results(max_n=6, figsize=(7,8))





interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(k=2)

