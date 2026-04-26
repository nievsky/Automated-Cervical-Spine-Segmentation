# from ultralytics import YOLO
# import os
# from pathlib import Path

# def run_prediction():
#     # model = YOLO("SpineSeminar/v1_segmentation/weights/best.pt")
#     model = YOLO("yolo11s-seg.pt")
#     test_images_path = Path("dataset/images/test")
#     output_dir = Path("prediction_results")
#     output_dir.mkdir(exist_ok=True)

#     # Predict on all images in the test set
#     results = model.predict(source=str(test_images_path), conf=0.3, save=True)

#     # Error Analysis: Find images where the model missed bones
#     print("\n--- ERROR ANALYSIS ---")
#     for result in results:
#         # We expect 5 vertebrae (C3-C7)
#         detected_count = len(result.boxes)
#         if detected_count < 5:
#             print(f"⚠️ Image {result.path.name}: Only detected {detected_count}/5 bones.")
#         elif detected_count > 5:
#             print(f"⚠️ Image {result.path.name}: Over-detected ({detected_count}/5). Possible noise.")

# if __name__ == "__main__":
#     run_prediction()

from ultralytics import YOLO
import os

# 1. Load your best brain
model = YOLO("runs\\segment\\SpineSeminar\\v1_segmentation\\weights\\best.pt")

# 2. Run prediction on the TEST set
# save=True: Saves the images with the colored outlines
# line_width=2: Makes the boxes/labels look cleaner
# show_labels=True: Shows "C3", "C4", etc.
results = model.predict(
    source="dataset/images/test",
    save=True,
    imgsz=640,
    conf=0.5,
    project="VisualResults",
    name="test_predictions"
)

print(f"✅ Visualizations saved to: {os.path.abspath('VisualResults/test_predictions')}")