import cv2
import fitz
import json
import os
from pathlib import Path
from ultralytics import YOLO

def setup_predictor(model_path):
    return YOLO(model_path)

def process_pdf(pdf_path, model):
    doc = fitz.open(pdf_path)
    results = []
    
    # Class names mapping
    class_names = ["REVISION_TABLE", "MATERIAL_ROW", "GRADE", "DRAWING_NO", "CONTRACT", "DATE"]

    for i, page in enumerate(doc):
        # Convert page to image for detection
        # Use high DPI for better text extraction later
        pix = page.get_pixmap(dpi=300)
        img_path = f"temp_page_{i}.png"
        pix.save(img_path)
        
        # Run inference
        # imgsz=1280 should match training
        # Lowering confidence to 0.1 to see if we get ANY results
        prediction_results = model.predict(img_path, imgsz=1280, conf=0.1, verbose=False)
        
        page_results = []
        for r in prediction_results:
            boxes = r.boxes
            if len(boxes) > 0:
                print(f"  Found {len(boxes)} detections on page {i}")
            
            for box in boxes:
                # Bbox in xyxy format
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0].item())
                score = float(box.conf[0].item())
                
                label = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                
                # Extract text from the bounding box using PyMuPDF
                # Scale coordinates back to PDF space (72 DPI)
                scale = 72 / 300
                pdf_box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
                
                # Get raw text
                text = page.get_textbox(pdf_box)
                
                # Improved: Parse rows for REVISION_TABLE
                rows = []
                if label == "REVISION_TABLE":
                    # Get text blocks within the box
                    blocks = page.get_text("blocks", clip=pdf_box)
                    # Sort by Y then X
                    blocks.sort(key=lambda b: (b[1], b[0]))
                    
                    # Group blocks into rows based on Y proximity
                    current_row = []
                    last_y = -1
                    threshold = 5 # 5 points vertical distance
                    
                    for b in blocks:
                        if last_y == -1 or abs(b[1] - last_y) < threshold:
                            current_row.append(b[4].strip())
                        else:
                            rows.append(current_row)
                            current_row = [b[4].strip()]
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
        
        # Cleanup temp image
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
        
    latest_runs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train")], key=get_run_num, reverse=True)
    
    for run in latest_runs:
        for weight_name in ["last.pt", "best.pt"]:
            potential_weight = run / "weights" / weight_name
            if potential_weight.exists():
                return potential_weight
    return None

if __name__ == "__main__":
    # Path to the trained model
    model_path = get_latest_model()

    if not model_path or not model_path.exists():
        print("No model found. Cannot run inference.")
    else:
        print(f"Using model: {model_path}")
        model = setup_predictor(str(model_path))
        
        # Process all PDFs in drawings directory
        drawings_dir = Path("drawings")
        pdf_files = list(drawings_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found in drawings directory.")
        else:
            all_results = {}
            for pdf_file in pdf_files[:10]: # Process first 10 for now
                print(f"Processing {pdf_file.name}...")
                results = process_pdf(str(pdf_file), model)
                all_results[pdf_file.name] = results
            
            output_file = "detection_results.json"
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=4)
            print(f"Done! Results for {len(all_results)} files saved to {output_file}")
