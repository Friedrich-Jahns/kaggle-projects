import numpy as np
from pathlib import Path
import torch
from fastai.vision.all import *
from PIL import Image
import napari

# -------------------------------------------------
# Pfade
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent / "train_data" / "train"
IMAGE_DIR = BASE_DIR / "img"
MODEL_PATH = Path(__file__).resolve().parent / "curve_multiclass_model.pkl"

assert IMAGE_DIR.exists(), "Image-Verzeichnis nicht gefunden"
assert MODEL_PATH.exists(), "Modell (.pkl) nicht gefunden"

# -------------------------------------------------
# Modell laden
# -------------------------------------------------
learn = load_learner(MODEL_PATH)
learn.model.eval()

# -------------------------------------------------
# Testbild laden
# -------------------------------------------------
img_file = sorted(IMAGE_DIR.glob("*.png"))[0]
print(f"🖼 Testbild: {img_file.name}")

img = PILImage.create(img_file)
img_np = np.array(img)

# -------------------------------------------------
# Vorhersage
# -------------------------------------------------
pred_mask, pred_class, pred_logits = learn.predict(img)
# pred_logits: [4, H, W] → 0=Hintergrund, 1-3=Klassen

# -------------------------------------------------
# Softmax auf Klassen
# -------------------------------------------------
pred_probs = torch.softmax(pred_logits, dim=0)  # Jetzt echte Wahrscheinlichkeiten 0-1

# -------------------------------------------------
# Einzelmasken erstellen
# -------------------------------------------------
OBJ_THRESHOLD = 0.25  # 0.2–0.35 ausprobieren
class_masks = []

for i in range(1, 4):  # Klassen 1-3
    mask_i = (pred_probs[i] > OBJ_THRESHOLD).cpu().numpy().astype(np.uint8)
    class_masks.append(mask_i)

# -------------------------------------------------
# Kombinierte Farb-Maske
# -------------------------------------------------
combined_mask = np.zeros_like(class_masks[0], dtype=np.uint8)

for idx, m in enumerate(class_masks, start=1):
    combined_mask[m > 0] = idx

# -------------------------------------------------
# Napari Visualisierung
# -------------------------------------------------
viewer = napari.Viewer()

# Originalbild
viewer.add_image(
    img_np,
    name="Input",
    colormap="gray"
)

# Einzelne Klassenmasken
for i, mask in enumerate(class_masks, start=1):
    viewer.add_image(
        mask,
        name=f"Klasse {i} Prob> {OBJ_THRESHOLD}",
        colormap="inferno",
        opacity=0.5
    )

# Kombinierte Farb-Maske über Original
viewer.add_labels(
    combined_mask,
    name="Farb-Maske"
)

napari.run()
