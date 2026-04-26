import os
from ultralytics import YOLO
import torch

def evaluate_model():
    # 1. Verification
    if not torch.cuda.is_available():
        print("❌ Still on CPU! Check your installation.")
        return
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

    # 2. Updated Path (Simplified)
    # If the file is in your project root under v1_segmentation/weights/
    model_path = "runs\\segment\\SpineSeminar\\v1_segmentation\\weights\\best.pt"

    if not os.path.exists(model_path):
        # Let's help find it if it's missing
        print(f"❌ Error: Could not find {model_path}")
        print(f"Current working directory: {os.getcwd()}")
        return

    model = YOLO(model_path)

    # 3. Execution
    # Adding 'verbose=True' helps debug data loading issues
    metrics = model.val(data="data.yaml", split='test', device=0, verbose=True)
    print(f"\n🚀 Final Test Mask mAP50: {metrics.seg.map50:.4f}")

if __name__ == "__main__":
    evaluate_model()