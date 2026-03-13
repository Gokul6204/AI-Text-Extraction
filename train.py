from ultralytics import YOLO
import os

def train_yolo():
    # Load a pre-trained YOLOv11n model (nano version is fast and effective for regions)
    # You can also use 'yolo11s.pt' or 'yolo11m.pt' for more accuracy
    model = YOLO("yolo11n.pt")

    # Train the model
    # data="data.yaml" points to our prepared dataset
    # epochs=100 is usually good, but we can start with 50
    # imgsz=1280 because drawings have small text/details
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=1280,
        device="cpu",  # Force CPU as CUDA is not available
        plots=True
    )
    
    print("Training complete. Model saved in runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    if not os.path.exists("data.yaml"):
        print("Error: data.yaml not found! Run prepare_dataset.py first.")
    else:
        train_yolo()
