import numpy as np
from pathlib import Path
from PIL import Image

from fastai.vision.all import *

# ============================================================
# 1️⃣ Pfade anpassen
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent / "train_data" / "train"

IMAGE_DIR = BASE_DIR / "img"
MODEL_PATH = BASE_DIR / "curve_segmentation_model.pkl"

assert IMAGE_DIR.exists(), "Image-Verzeichnis nicht gefunden"
assert MODEL_PATH.exists(), "Modell (.pkl) nicht gefunden"

# ============================================================
# 2️⃣ Modell laden
# ============================================================

learn = load_learner(MODEL_PATH)
learn.model.eval()

print("✅ Modell geladen")

# ============================================================
# 3️⃣ Beispielbild auswählen
# ============================================================

img_file = sorted(IMAGE_DIR.glob("*.png"))[0]
print(f"🖼 Testbild: {img_file.name}")

img = PILImage.create(img_file)

# ============================================================
# 4️⃣ Vorhersage
# ============================================================

pred_mask, pred_class, pred_probs = learn.predict(img)

# pred_mask ist ein Tensor mit Klassen-IDs (0/1)
mask_np = pred_mask.numpy().astype(np.uint8)

# ============================================================
# 5️⃣ Maske speichern
# ============================================================

out_dir = BASE_DIR / "predictions"
out_dir.mkdir(exist_ok=True)

out_mask_path = out_dir / f"{img_file.stem}_pred_mask.png"
Image.fromarray(mask_np * 255).save(out_mask_path)

print(f"✅ Vorhersage gespeichert: {out_mask_path}")

# ============================================================
# 6️⃣ Optional: Overlay anzeigen
# ============================================================

learn.show_results(
    max_n=1,
    figsize=(5, 5)
)
