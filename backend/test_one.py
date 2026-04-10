import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
import re

def normalize_canvas(img, target_size=1280):
    """
    Resizes image to target_size using letterboxing (maintaining aspect ratio with padding).
    Returns: padded_image, scale_used, padding_x, padding_y
    """
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))
    
    canvas = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    dx = (target_size - nw) // 2
    dy = (target_size - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = img_resized
    
    return canvas, scale, dx, dy

def get_latest_model():
    import re
    finetune_path = Path("runs/detect/runs/detect/finetune/weights/best.pt")
    if finetune_path.exists():
        return finetune_path
    return None

if __name__ == "__main__":
    model_path = get_latest_model()
    if not model_path:
        print("Model not found")
        exit()
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Create a dummy image
    img = np.zeros((2000, 1500, 3), dtype=np.uint8)
    
    print("Normalizing...")
    norm_img, scale, dx, dy = normalize_canvas(img)
    
    print("Predicting...")
    start = time.time()
    results = model.predict(norm_img, imgsz=1280, conf=0.05, verbose=True)
    end = time.time()
    print(f"Prediction took {end - start:.2f} seconds")
