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
    h, w = img.shape[:2]

    # If portrait → keep as is
    if h >= w:
        return cv2.resize(img, (target_size, target_size))

    # If landscape → resize to portrait shape
    return cv2.resize(img, (target_size, target_size))


def process_pdf(pdf_path, model):
    doc = fitz.open(pdf_path)
    results = []
    
    class_names = ["REVISION_TABLE", "MATERIAL_ROW", "GRADE", "DRAWING_NO", "CONTRACT", "DATE"]

    for i, page in enumerate(doc):

        # -------------------------------
        # Convert PDF → image (NO CHANGE)
        # -------------------------------
        pix = page.get_pixmap(dpi=300)

        img = cv2.imdecode(
            np.frombuffer(pix.samples, dtype=np.uint8),
            cv2.IMREAD_COLOR
        ) if False else None  # fallback if needed

        # Better method (stable):
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # -------------------------------
        # NEW: Normalize canvas
        # -------------------------------
        norm_img = normalize_canvas(img, target_size=1280)

        img_path = f"temp_page_{i}.png"
        cv2.imwrite(img_path, norm_img)

        # -------------------------------
        # YOLO inference (UNCHANGED)
        # -------------------------------
        prediction_results = model.predict(img_path, imgsz=1280, conf=0.1, verbose=False)
        
        page_results = []

        for r in prediction_results:
            boxes = r.boxes

            if len(boxes) > 0:
                print(f"  Found {len(boxes)} detections on page {i}")
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0].item())
                score = float(box.conf[0].item())
                
                label = class_names[cls] if cls < len(class_names) else f"class_{cls}"

                # -------------------------------
                # SCALE BACK TO ORIGINAL PDF
                # -------------------------------
                scale_x = img.shape[1] / 1280
                scale_y = img.shape[0] / 1280

                x1 *= scale_x
                x2 *= scale_x
                y1 *= scale_y
                y2 *= scale_y

                scale = 72 / 300
                pdf_box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]

                # -------------------------------
                # TEXT EXTRACTION (UNCHANGED)
                # -------------------------------
                text = page.get_textbox(pdf_box)

                rows = []
                if label == "REVISION_TABLE":
                    blocks = page.get_text("blocks", clip=pdf_box)
                    blocks.sort(key=lambda b: (b[1], b[0]))

                    current_row = []
                    last_y = -1
                    threshold = 5

                    for b in blocks:
                        txt = b[4].strip()
                        if not txt:
                            continue

                        if last_y == -1 or abs(b[1] - last_y) < threshold:
                            current_row.append(txt)
                        else:
                            rows.append(current_row)
                            current_row = [txt]

                        last_y = b[1]

                    if current_row:
                        rows.append(current_row)

                page_results.append({
                    "label": label,
                    "confidence": score,
                    "box": [x1, y1, x2, y2],
                    "text": text.strip(),
                    "rows": rows if rows else None
                })

        results.append({
            "page": i,
            "detections": page_results
        })

        if os.path.exists(img_path):
            os.remove(img_path)

    return results


def get_latest_model():
    import re
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
        for weight_name in ["last.pt", "best.pt"]:
            potential_weight = run / "weights" / weight_name
            if potential_weight.exists():
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
        directories = ["drawings", "drawing"]

        all_results = {}

        for dir_name in directories:
            drawings_dir = Path(dir_name)
            pdf_files = list(drawings_dir.glob("*.pdf"))

            if not pdf_files:
                print(f"No PDF files found in {dir_name}.")
                continue

            print(f"\n📁 Processing directory: {dir_name}")

            for pdf_file in pdf_files[:10]:
                print(f"Processing {pdf_file.name}...")
                results = process_pdf(str(pdf_file), model)
                
                # Store results with directory + filename (to avoid overwrite)
                key = f"{dir_name}/{pdf_file.name}"
                all_results[key] = results

        # Save all results
        with open("detection_results.json", "w") as f:
            json.dump(all_results, f, indent=4)

        print("✅ Done!")