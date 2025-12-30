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
pred_mask, pred_class, pred_probs = learn.predict(img)
# pred_probs: [4, H, W]  → 0 = Hintergrund, 1–3 = Klassen

# -------------------------------------------------
# BESSERE Multiclass-Maskenlogik
# -------------------------------------------------
# Objektklassen konkurrieren lassen
obj_probs = pred_probs[1:]                 # [3, H, W]
max_obj_prob, obj_class = obj_probs.max(dim=0)

mask = torch.zeros_like(pred_probs[0], dtype=torch.long)

OBJ_THRESHOLD = 0.25    # 0.2–0.35 ausprobieren
mask[max_obj_prob > OBJ_THRESHOLD] = obj_class[max_obj_prob > OBJ_THRESHOLD] + 1

mask_np = mask.cpu().numpy().astype(np.uint8)

# -------------------------------------------------
# Napari Visualisierung (RICHTIG!)
# -------------------------------------------------
viewer = napari.Viewer()

# Originalbild
viewer.add_image(
    img_np,
    name="Input",
    colormap="gray"
)

# Vorhersage als Labels (NICHT add_image!)
viewer.add_image(
    mask_np,
    name="Prediction",
    opacity=0.6
)

# -------------------------------------------------
# Debug: Wahrscheinlichkeitskarten anzeigen
# -------------------------------------------------
for i in range(1, 4):
    viewer.add_image(
        pred_probs[i].cpu().numpy(),
        name=f"Prob Klasse {i}",
        colormap="inferno",
        opacity=0.7
    )

napari.run()
