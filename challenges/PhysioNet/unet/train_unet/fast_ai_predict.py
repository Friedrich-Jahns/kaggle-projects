import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from fastai.vision.all import *

# ============================================================
# 1️⃣ Pfade
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent / "train_data" / "train"

IMAGE_DIR = BASE_DIR / "img"
MODEL_PATH = Path(__file__).resolve().parent / "curve_segmentation_model.pkl"

assert IMAGE_DIR.exists(), "Image-Verzeichnis nicht gefunden"
assert MODEL_PATH.exists(), "Modell (.pkl) nicht gefunden"

# ============================================================
# 2️⃣ Modell laden
# ============================================================

learn = load_learner(MODEL_PATH)
learn.model.eval()


# ============================================================
# 3️⃣ Beispielbild auswählen
# ============================================================

img_file = sorted(IMAGE_DIR.glob("*.png"))[0]
print(img_file)
print(f"🖼 Testbild: {img_file.name}")

img = PILImage.create(img_file)

# ============================================================
# 4️⃣ Vorhersage
# ============================================================

pred_mask, pred_class, pred_probs = learn.predict(img)

# Klassenmaske (0/1)
mask_np = pred_mask.numpy()

# Wahrscheinlichkeitskarte für Klasse "curve"
prob_curve = pred_probs[..., 1].numpy()

# ============================================================
# 5️⃣ Plot: Input + Prediction
# ============================================================

# Vorhersage
pred_mask, pred_class, pred_probs = learn.predict(img)

# Wahrscheinlichkeiten für Klasse "curve"
prob_curve = pred_probs[1].numpy()  # Shape: (H, W)

# Originalbild in NumPy
img_np = np.array(img)  # Shape: (H, W) oder (H, W, 3)

fig, axes = plt.subplots(1, 2, figsize=(10, 10))

# Input
axes[0].imshow(img_np, cmap="gray")
axes[0].axis("off")
axes[0].set_title("Input")

# Prediction Overlay
# axes[1].imshow(img_np, cmap="gray")
# med_prob_curve = np.median(prob_curve)
prob_curve = np.where(prob_curve>.15,1,0)
axes[1].imshow(prob_curve, alpha=0.6, cmap="Reds", extent=(0, img_np.shape[1], img_np.shape[0], 0))
axes[1].axis("off")
axes[1].set_title("Prediction (Wahrscheinlichkeit)")

plt.show()