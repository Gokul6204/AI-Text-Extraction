import cv2
import fitz
import json
import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def setup_predictor(model_path):
    return YOLO(model_path)

# -------------------------------
# NEW: Normalize landscape → portrait
# -------------------------------
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


def process_pdf(pdf_path, model, original_filename=None):
    doc = fitz.open(pdf_path)
    best_detections = {}
    
    class_names = ["REVISION_TABLE", "DRAWING_NO", "DRAWING_DESCRIPTION", "PROJECT_NO"]
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization: Most details are on the first 1-2 pages. 
    # Limit search to 3 pages max to speed up multi-page documents
    max_pages = min(len(doc), 3)

    for i in range(max_pages):
        page = doc[i]
        pix = page.get_pixmap(dpi=150) 
        
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        norm_img, scale_val, dx, dy = normalize_canvas(img, target_size=1280)

        # No disk write: pass image directly
        prediction_results = model.predict(norm_img, imgsz=1280, conf=0.05, verbose=False, device=device)
        
        for r in prediction_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0].item())
                score = float(box.conf[0].item())
                
                label = class_names[cls] if cls < len(class_names) else f"class_{cls}"

                # Coordinates translation back to original image (accounting for padding and scale)
                x1_orig = (x1 - dx) / scale_val
                y1_orig = (y1 - dy) / scale_val
                x2_orig = (x2 - dx) / scale_val
                y2_orig = (y2 - dy) / scale_val

                # PDF coordinates (72 dpi) from image coordinates (150 dpi)
                scale_to_pdf = 72 / 150
                pdf_box = [x1_orig * scale_to_pdf, y1_orig * scale_to_pdf, 
                           x2_orig * scale_to_pdf, y2_orig * scale_to_pdf]

                text = page.get_textbox(pdf_box)
                rows = []
                if label == "REVISION_TABLE":
                    blocks = page.get_text("blocks", clip=pdf_box)
                    blocks.sort(key=lambda b: (b[1], b[0]))
                    current_row = []
                    last_y = -1
                    threshold = 5
                    
                    # Robust header filtering keywords
                    header_keywords = {"REV", "DESCRIPTION", "DATE", "DWN", "CHKD", "APPROV", "REVISION", "REVISIONS"}
                    
                    for b in blocks:
                        txt = b[4].strip()
                        if not txt: continue
                        
                        # Count how many header keywords are present in the text
                        txt_upper = txt.upper()
                        # Remove non-alpha chars for cleaner keyword matching
                        clean_txt = "".join(c if c.isalnum() or c.isspace() else " " for c in txt_upper)
                        words = set(clean_txt.split())
                        
                        match_count = len(words.intersection(header_keywords))
                        
                        # If more than 1 keyword matches, OR it's a short string with a specific header like "REV #"
                        is_header = match_count >= 2 or (match_count >= 1 and len(txt) < 15)
                        
                        if is_header and len(txt) < 60:
                            continue

                        if last_y == -1 or abs(b[1] - last_y) < threshold:
                            current_row.append(txt)
                        else:
                            rows.append(current_row)
                            current_row = [txt]
                        last_y = b[1]
                    if current_row:
                        rows.append(current_row)
                
                elif label in ["PROJECT_NO", "DRAWING_NO", "DRAWING_DESCRIPTION"]:
                    rows = [[line.strip()] for line in text.split('\n') if line.strip()]

                detection_data = {
                    "page": i,
                    "label": label,
                    "confidence": score,
                    "box": [x1, y1, x2, y2],
                    "text": text.strip(),
                    "rows": rows if rows else None
                }

                if label not in best_detections or score > best_detections[label]["confidence"]:
                    best_detections[label] = detection_data

        # Early Exit Algorithm: If we found all 4 key components on the current page 
        # with high confidence, we don't need to scan remaining pages.
        if len(best_detections) == 4:
            is_satisfied = True
            for det in best_detections.values():
                conf = det.get("confidence", 0.0)
                if float(conf) < 0.6:
                    is_satisfied = False
                    break
            if is_satisfied:
                break

    doc.close()
    
    # ---------------------------------------------------------
    # FALLBACK: If DRAWING_NO is missing or confidence is very low (< 0.1)
    # USE FILENAME (split by '_')
    # ---------------------------------------------------------
    # USE ORIGINAL FILENAME OR FALLBACK TO PATH
    name_to_use = original_filename if original_filename else pdf_path
    filename_stem = Path(name_to_use).stem
    fallback_val = filename_stem.split('_')[0]
    
    if "DRAWING_NO" not in best_detections or \
       best_detections["DRAWING_NO"]["confidence"] < 0.15 or \
       not best_detections["DRAWING_NO"]["text"].strip():
        
        if "DRAWING_NO" not in best_detections:
            best_detections["DRAWING_NO"] = {
                "page": 0,
                "label": "DRAWING_NO",
                "confidence": 0.0,
                "box": [0, 0, 0, 0],
                "text": fallback_val,
                "rows": [[fallback_val]],
                "note": "Extracted from filename (fallback)"
            }
        else:
            # Update the existing low-conf or empty one
            orig_conf = best_detections["DRAWING_NO"]["confidence"]
            best_detections["DRAWING_NO"]["text"] = fallback_val
            best_detections["DRAWING_NO"]["rows"] = [[fallback_val]]
            best_detections["DRAWING_NO"]["note"] = f"Original (conf {orig_conf:.2f}) was empty or low-conf; used filename"

    return list(best_detections.values())


def get_latest_model():
    """
    Model priority (highest to lowest):
      1. Fine-tuned model  -> runs/detect/runs/detect/finetune/weights/best.pt
      2. Latest train run  -> runs/detect/train*/weights/best.pt  (newest first)
    """
    import re

    # --- Priority 1: explicitly look for the fine-tuned model ---
    finetune_path = Path("runs/detect/runs/detect/finetune/weights/best.pt")
    if finetune_path.exists():
        print(f"[Model] Using fine-tuned model: {finetune_path}")
        return finetune_path

    # --- Priority 2: newest 'train*' run under runs/detect ---
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        return None

    def get_run_num(p):
        match = re.search(r'train(\d*)', p.name)
        return int(match.group(1)) if match and match.group(1) else 0

    latest_runs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train")],
        key=get_run_num,
        reverse=True
    )

    for run in latest_runs:
        for weight_name in ["best.pt", "last.pt"]:
            potential_weight = run / "weights" / weight_name
            if potential_weight.exists():
                print(f"[Model] Using training model: {potential_weight}")
                return potential_weight
    return None


if __name__ == "__main__":

    model_path = get_latest_model()

    if not model_path or not model_path.exists():
        print("No model found. Cannot run inference.")
    else:
        
        print(f"Using model: {model_path}")
        model = setup_predictor(str(model_path))    

        # List of directories to process
        directories = ["drawing"]

        all_results = {}

        for dir_name in directories:
            drawings_dir = Path(dir_name)
            pdf_files = list(drawings_dir.glob("*.pdf"))

            if not pdf_files:
                print(f"No PDF files found in {dir_name}.")
                continue

            print(f"\nProcessing directory: {dir_name}")

            for pdf_file in pdf_files:
                print(f"Processing {pdf_file.name}...")
                results = process_pdf(str(pdf_file), model, original_filename=pdf_file.name)
                
                # Store results with directory + filename (to avoid overwrite)
                key = f"{dir_name}/{pdf_file.name}"
                all_results[key] = results

        # Save all results
        with open("detection_results.json", "w") as f:
            json.dump(all_results, f, indent=4)

        print("Done!")