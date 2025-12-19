import numpy as np
from PIL import Image
from pathlib import Path
import os

from fastai.vision.all import *

# ============================================================
# 1️⃣ Pfade (LOKAL, konsistent)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent / "train_data" / "train"

IMAGE_DIR = BASE_DIR / "img"
MASK_DIR  = BASE_DIR / "curve_mask"

assert IMAGE_DIR.exists(), "Image-Verzeichnis nicht gefunden"
assert MASK_DIR.exists(),  "Masken-Verzeichnis nicht gefunden"

print(f"Bilder: {len(list(IMAGE_DIR.glob('*.png')))}")
print(f"Masken: {len(list(MASK_DIR.glob('*.png')))}")

# ============================================================
# 2️⃣ Masken prüfen & korrigieren (0/1, richtige Größe)
# ============================================================

def check_and_fix_mask(mask_file: Path, img_file: Path):
    img  = Image.open(img_file)
    mask = Image.open(mask_file).convert("L")

    arr = np.array(mask)
    arr = (arr > 0).astype(np.uint8)  # nur im Speicher binär 0/1

    # auf Bildgröße prüfen und ggf. anpassen
    if arr.shape != (img.height, img.width):
        arr = np.array(
            Image.fromarray(arr).resize(
                (img.width, img.height), resample=Image.NEAREST
            )
        )
        arr = (arr > 0).astype(np.uint8)

    # ✅ NICHT speichern, nur prüfen / zurückgeben
    if not set(np.unique(arr)).issubset({0, 1}):
        raise ValueError(f"Ungültige Maskenwerte in {mask_file}")

    return arr  # nur im Speicher

# Beispiel: alle Masken prüfen (nur im Speicher)
for img_file in IMAGE_DIR.glob("*.png"):
    mask_file = MASK_DIR / img_file.name
    if not mask_file.exists():
        raise FileNotFoundError(f"Maske fehlt für {img_file.name}")
    arr = check_and_fix_mask(mask_file, img_file)

# ============================================================
# 3️⃣ FastAI DataBlock
# ============================================================
def get_mask(fn):
    mask_file = MASK_DIR / fn.name
    mask = PILMask.create(mask_file)
    arr = np.array(mask)
    arr = (arr > 0).astype(np.uint8)  # 0/1 im Speicher
    return TensorMask(arr) 

dblock = DataBlock(
    blocks=(
        ImageBlock,
        MaskBlock(codes=["background", "curve"])
    ),
    get_items=get_image_files,
    get_y=get_mask,
    splitter=IndexSplitter([0]),#RandomSplitter(seed=42),
    item_tfms=Resize(512, method="pad", pad_mode="zeros"),
    batch_tfms=aug_transforms(
        do_flip=False,
        max_rotate=2,
        max_warp=0.0,
        max_zoom=1.05
    )
)

dls = dblock.dataloaders(IMAGE_DIR, bs=2)

dls.show_batch(max_n=4)

# ============================================================
# 4️⃣ Modell (U-Net)
# ============================================================

def dice_thresh(inp, targ, thresh=0.5):
    inp = (inp.sigmoid() > thresh).float()
    targ = targ.float()
    inter = (inp * targ).sum()
    return (2 * inter) / (inp.sum() + targ.sum() + 1e-8)

learn = unet_learner(
    dls,
    resnet34,
    metrics=dice_thresh
)


learn.fine_tune(5)

# ============================================================
# 5️⃣ Ergebnisse & Fehleranalyse
# ============================================================

learn.show_results(max_n=6, figsize=(7, 8))

interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(k=1)

# ============================================================
# 6️⃣ Modell exportieren
# ============================================================

EXPORT_PATH = Path(__file__).resolve().parent / "curve_segmentation_model.pkl"
learn.export(EXPORT_PATH)

print(f"✅ Modell exportiert nach: {EXPORT_PATH}")
