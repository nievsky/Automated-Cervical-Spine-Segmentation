# Automated Cervical Spine Segmentation in 2D Radiographs
**Subject:** Essentials of Machine Learning
**Author:** [Your Name]
**Date:** April 2026

---

## 1. Project Overview
This project addresses the challenge of automated anatomical identification in medical imaging. Specifically, we developed an **Instance Segmentation** pipeline for cervical vertebrae (C3-C7) using the Cervical Spine X-ray Atlas (CSXA). The goal was to move beyond simple object detection (bounding boxes) to provide precise anatomical outlines (polygons).

## 2. Methodology Selection: YOLOv11-seg vs. TotalSegmentator
Initial research suggested *TotalSegmentator*; however, it was rejected for this project due to:
* **Dimensionality Mismatch:** TotalSegmentator is optimized for 3D volumetric data (CT/MRI). Applying it to 2D X-rays involves significant data "hallucination."
* **Computational Constraints:** YOLOv11-seg offers a superior balance of accuracy and latency on consumer-grade hardware (NVIDIA RTX 3070).

---

## 3. Data Preprocessing: The Engineering Core
Preprocessing was the most critical step in ensuring model convergence.

### 3.1 Vertex Ordering (Clockwise Polygonization)
Standard segmentation models require non-self-intersecting polygons. To solve the problem of geometric "bow-tie" artifacts, we implemented a strict clockwise re-ordering:
1. **Top-Left (TL)** -> 2. **Top-Right (TR)** -> 3. **Bottom-Right (BR)** -> 4. **Bottom-Left (BL)**

### 3.2 Coordinate Normalization
To ensure the model is scale-invariant, raw pixel coordinates $(x, y)$ were converted to normalized values $[0, 1]$ relative to the image dimensions $(W, H)$:
$$x_{norm} = \frac{x}{W}, \quad y_{norm} = \frac{y}{H}$$

### 3.3 Data Partitioning (70/15/15 Split)
We implemented a 3-way partition to ensure scientific validity:
* **Training (70%):** Core learning set.
* **Validation (15%):** Used for hyperparameter tuning and 'best.pt' weight selection.
* **Testing (15%):** An "Unseen" dataset used only once to provide an unbiased evaluation.

---

## 4. The Training Process
### 4.1 Transfer Learning & Optimization
We utilized **Transfer Learning** from the `yolo11s-seg` pre-trained weights. This allowed the model to leverage existing knowledge of universal visual features (edges/textures) while specializing in spinal anatomy.

* **Early Stopping:** We set a `patience` of 20 epochs. This ensures that the training stops as soon as the model begins to **Overfit** (memorizing training data instead of learning general patterns).

### 4.2 Training Metrics Visualization
> **[INSERT results.png HERE]**
> ![Training Results Placeholder](path/to/results.png)

---

## 5. Crucial Evaluation Metrics
We evaluated the model using the following mathematical foundations:

1. **Intersection over Union (IoU):**
   $$IoU = \frac{|A \cap B|}{|A \cup B|}$$
   Measures the spatial overlap between predicted ($A$) and ground truth ($B$).

2. **mAP50-95:**
   The "Gold Standard" metric, averaging precision across multiple IoU thresholds (0.5 to 0.95). High scores here indicate superior "boundary fidelity" (how well the mask matches the bone's edge).

---

## 6. Qualitative Error Analysis (Failure Mode Discovery)
Despite achieving a **mAP50 of 0.994**, qualitative analysis of the test set revealed specific failure modes. Analyzing these "mistakes" provides insight into the model's logic:

### 6.1 Failure Mode: Ordinal Assignment Errors
In images such as `error_0661149.jpg` and `error_1752053.jpg`, the model correctly identified 5 vertebral bodies but mislabeled their sequence (e.g., predicting two C3s).
* **Cause:** Morphological similarity. C3-C6 look nearly identical to a pixel-based model.
* **Interpretation:** The model is an excellent "Feature Detector" but lacks "Anatomical Sequence Awareness."

### 6.2 Failure Mode: Obscuration & Noise
In cases where the mandible (jawbone) or high shoulders overlapped with the spine, the model's confidence dropped (e.g., $Conf < 0.50$). This indicates that image noise from overlapping anatomy remains the primary challenge for 2D radiographs.

---

## 7. Conclusion & Future Work
The project successfully demonstrated that a correctly engineered 2D segmentation pipeline is highly effective for spinal analysis.
**Future Work:** To improve the model, we propose adding a **"Sequence Correction" Layer** in post-processing to ensure ordinal consistency (C3 must always be above C4), and utilizing **Hard Example Mining** to retrain the model on the specific error cases identified in this report.

---














## UPD. Minimal Git Add Checklist
Use this quick checklist before each commit to avoid staging datasets and large artifacts.

1. Check what changed:
   ```bash
   git status
   ```
2. Preview exactly what will be staged:
   ```bash
   git add -n README.md data.yaml scripts/
   ```
3. Stage only essential project files:
   ```bash
   git add README.md data.yaml scripts/ .gitignore
   ```
4. Verify staged content:
   ```bash
   git diff --staged
   ```