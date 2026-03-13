import json
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def convert_ls_to_yolo():
    with open('annotation.json', 'r') as f:
        data = json.load(f)

    # Categories to IDs
    classes = ["REVISION_TABLE", "MATERIAL_ROW", "GRADE", "DRAWING_NO", "CONTRACT", "DATE"]
    cls_to_id = {cls: i for i, cls in enumerate(classes)}

    # Create directories
    base_dir = Path("datasets/drawings")
    for split in ['train', 'val']:
        (base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Prepare data for splitting
    valid_tasks = []
    for task in data:
        if not task.get('annotations'):
            continue
        valid_tasks.append(task)

    train_tasks, val_tasks = train_test_split(valid_tasks, test_size=0.2, random_state=42)

    def process_split(tasks, split):
        for task in tasks:
            # Handle filename (Label Studio adds a prefix like 6bf9d97e-)
            full_filename = task['file_upload']
            # Find the original filename by matching against our images folder
            # Usually it's after the first dash
            # Example: 6bf9d97e-a12_0_page_0.png -> a12_0_page_0.png
            parts = full_filename.split('-', 1)
            original_filename = parts[1] if len(parts) > 1 else full_filename
            
            src_img_path = Path("images") / original_filename
            if not src_img_path.exists():
                # Try finding it if naming is different
                print(f"Warning: Could not find image {src_img_path}")
                continue

            # Copy image
            shutil.copy(src_img_path, base_dir / "images" / split / original_filename)

            # Create label file
            label_filename = src_img_path.stem + ".txt"
            label_path = base_dir / "labels" / split / label_filename
            
            with open(label_path, 'w') as lf:
                for ann in task['annotations'][0]['result']:
                    if ann['type'] != 'labels':
                        continue
                        
                    label = ann['value']['labels'][0]
                    if label not in cls_to_id:
                        continue
                        
                    cls_id = cls_to_id[label]
                    
                    # LS provides coordinates in percentages (0-100)
                    # YOLO wants relative (0-1) [center_x, center_y, width, height]
                    x_perc = ann['value']['x']
                    y_perc = ann['value']['y']
                    w_perc = ann['value']['width']
                    h_perc = ann['value']['height']
                    
                    # Convert to relative center format
                    x_center = (x_perc + w_perc / 2) / 100
                    y_center = (y_perc + h_perc / 2) / 100
                    width = w_perc / 100
                    height = h_perc / 100
                    
                    lf.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    process_split(train_tasks, 'train')
    process_split(val_tasks, 'val')

    # Create data.yaml
    yaml_content = f"""
path: {base_dir.absolute()}
train: images/train
val: images/val

names:
  0: REVISION_TABLE
  1: MATERIAL_ROW
  2: GRADE
  3: DRAWING_NO
  4: CONTRACT
  5: DATE
"""
    with open('data.yaml', 'w') as f:
        f.write(yaml_content.strip())
    
    print("Dataset prepared successfully in datasets/drawings/")

if __name__ == "__main__":
    convert_ls_to_yolo()
