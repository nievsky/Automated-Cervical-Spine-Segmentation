from ultralytics import YOLO

def start_training():
    # 1. Load the YOLOv11 Small Segmentation model
    # It's pre-trained on generic objects, we will fine-tune it for spines
    model = YOLO("yolo11s-seg.pt")

    # 2. Train the model
    # We use 'device=0' for your GPU and 'imgsz=640' for standard resolution
    results = model.train(
        data="data.yaml",
        epochs=100,          # 100 is standard; it will stop early if it stops improving
        imgsz=640,
        batch=16,            # If you get "Out of Memory", change this to 8
        device=0,            # Your RTX 3070
        patience=20,         # Early stopping: stops if no improvement for 20 epochs
        project="SpineSeminar",
        name="v1_segmentation",
        save=True
    )

if __name__ == "__main__":
    start_training()