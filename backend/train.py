from ultralytics import YOLO
import os
from pathlib import Path

# -----------------------------------------------------------------------
# FINE-TUNE (resume learning) from the previously trained model
# -----------------------------------------------------------------------
# The old model already knows the 4 classes from annotation_old.json.
# We load its weights and continue training on the NEW annotation.json
# dataset so it gains extra knowledge without forgetting the old one.
# -----------------------------------------------------------------------

OLD_MODEL = Path("../runs/detect/train/weights/best.pt")   # adjust if path differs
FALLBACK  = "yolo11s.pt"                                    # download fresh only if old not found

def train_yolo():
    if OLD_MODEL.exists():
        print(f"[Fine-tune] Loading existing model: {OLD_MODEL}")
        model = YOLO(str(OLD_MODEL))
    else:
        print(f"[WARNING] Old model not found at {OLD_MODEL}.")
        print(f"[Fallback] Starting from pretrained {FALLBACK} instead.")
        model = YOLO(FALLBACK)

    results = model.train(
        data="data.yaml",

        # ---- Training length ----
        # Fewer epochs for fine-tuning (model already has a good base)
        epochs=80,
        patience=20,

        # ---- Image / batch ----
        imgsz=1280,          # keep high – drawings have small details
        batch=8,             # adjust to your GPU VRAM (lower if OOM)

        # ---- Hardware ----
        device="cuda",
        workers=0,           # avoid WinError 1455 on Windows

        # ---- Output ----
        project="runs/detect",
        name="finetune",     # saved separately from the old run
        exist_ok=True,
        plots=True,
        verbose=True,

        # ---- Learning rate (lower for fine-tuning to avoid forgetting) ----
        lr0=0.001,           # initial LR (half of default 0.01)
        lrf=0.01,            # final LR factor

        # ---- Augmentation (same conservative set as before) ----
        single_cls=False,
        rect=True,

        degrees=8.0,
        translate=0.08,
        scale=0.15,
        shear=0.0,
        perspective=0.0,

        flipud=0.0,          # documents should not be flipped
        fliplr=0.0,

        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.05,

        close_mosaic=10,
    )

    print("\n=== Fine-tuning complete ===")
    print("Best model saved at: runs/detect/finetune/weights/best.pt")


if __name__ == "__main__":
    if not os.path.exists("data.yaml"):
        print("ERROR: data.yaml not found! Run prepare_dataset.py first.")
    else:
        train_yolo()