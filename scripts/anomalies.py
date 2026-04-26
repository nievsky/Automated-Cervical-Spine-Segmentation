import os
import shutil
from ultralytics import YOLO

# 1. Load your trained model
model = YOLO("runs\\segment\\SpineSeminar\\v1_segmentation\\weights\\best.pt")

# 2. Paths
test_images_path = "dataset/images/test"
error_folder = "error_analysis_results"
os.makedirs(error_folder, exist_ok=True)

print("🔍 Mining for mistakes...")

# 3. Run prediction and filter errors
results = model.predict(source=test_images_path, conf=0.25)

for r in results:
    img_name = os.path.basename(r.path)
    num_detected = len(r.boxes)

    # CRITERIA FOR AN ERROR:
    # 1. Missing or extra bones (Expected 5)
    # 2. Low confidence detections (The AI is 'unsure')
    is_error = False

    if num_detected != 5:
        is_error = True
        reason = f"Detected {num_detected} bones"

    # Check if any detection is very low confidence
    if any(conf < 0.5 for conf in r.boxes.conf):
        is_error = True
        reason = "Low confidence detection"

    if is_error:
        print(f"❌ Error found in {img_name}: {reason}")
        # Save the visualized error to our special folder
        r.save(filename=os.path.join(error_folder, f"error_{img_name}"))

print(f"\n✅ Error mining complete. Check the '{error_folder}' directory.")