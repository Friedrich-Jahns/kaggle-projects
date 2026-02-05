import numpy as np
from pathlib import Path
from PIL import Image
import torch
from fastai.vision.all import *

# -------------------------------------------------
# 1️⃣ Pfade
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent / "train_data" / "train"
IMAGE_DIR = BASE_DIR / "img"

MASK_DIRS = [
    BASE_DIR / "grid_mask",   # Klasse 1
    BASE_DIR / "text_mask",   # Klasse 2
    BASE_DIR / "curve_mask",  # Klasse 3
]

MODEL_SAVE_PATH = Path("./curve_multiclass_model.pkl")

# -------------------------------------------------
# 2️⃣ Masken korrekt zusammenführen
#    - KEIN Überschreiben
#    - Hintergrund bleibt stabil
# -------------------------------------------------
def combine_masks(img_name):
    mask_files = [mask_dir / img_name for mask_dir in MASK_DIRS]

    base_mask = np.array(Image.open(mask_files[0]).convert("L"))
    combined = np.zeros_like(base_mask, dtype=np.uint8)

    for cls_idx, f in enumerate(mask_files, start=1):
        mask = np.array(Image.open(f).convert("L"))

        if mask.shape != combined.shape:
            mask = np.array(
                Image.fromarray(mask).resize(
                    combined.shape[::-1],
                    resample=Image.NEAREST
                )
            )

        mask = mask > 0

        combined[(mask) & (combined == 0)] = cls_idx

    return TensorMask(combined)

# -------------------------------------------------
# 3️⃣ DataBlock
# -------------------------------------------------
def get_mask(fn):
    return combine_masks(fn.name)

codes = [0, 1, 2, 3]  # Hintergrund + 3 Klassen

dblock = DataBlock(
    blocks=(ImageBlock, MaskBlock(codes=codes)),
    get_items=get_image_files,
    get_y=get_mask,
    splitter=RandomSplitter(seed=42),
    item_tfms=Resize(
        256,
        method=ResizeMethod.Pad,   # ❗ schützt dünne Klassen (curve)
        pad_mode='zeros'
    ),
    batch_tfms=aug_transforms(
        flip_vert=True,
        max_rotate=10,
        max_zoom=1.2
    )
)

dls = dblock.dataloaders(IMAGE_DIR, bs=2)

# -------------------------------------------------
# 4️⃣ Sanity-Check (EXTREM wichtig)
# -------------------------------------------------
x, y = dls.one_batch()
print("x:", x.shape)     # (B,3,H,W)
print("y:", y.shape)     # (B,H,W)
print("Label-Werte pro Sample:")
print([torch.unique(m) for m in y])

# -------------------------------------------------
# 5️⃣ Modell + RICHTIGER Loss
# -------------------------------------------------
from fastai.losses import FocalLossFlat

loss_func = FocalLossFlat(
    gamma=2.0,
    flatten=False
    # alpha=tensor([0.05, 1.0, 1.0, 1.5])
)

learn = unet_learner(
    dls,
    resnet34,              # ❗ besser für Klassentrennung
    loss_func=loss_func,
    metrics=[DiceMulti()]
)

# -------------------------------------------------
# 6️⃣ Training
# -------------------------------------------------
learn.fine_tune(15)

# -------------------------------------------------
# 7️⃣ Modell speichern
# -------------------------------------------------
learn.export(MODEL_SAVE_PATH)
print(f"✅ Modell gespeichert: {MODEL_SAVE_PATH}")
