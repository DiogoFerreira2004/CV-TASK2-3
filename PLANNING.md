# CV Tasks 2 & 3 — Project Planning

## Context & Constraints

| Parameter | Value |
|---|---|
| Experience | Intermediate PyTorch / deep learning |
| Hardware | Cloud GPU (Google Colab / Kaggle) |
| Group size | 3+ people |
| Deadline | June 22, 2026 (23:59 AoE) |
| Time available | Moderate (a few weekends / evenings) |
| Goal | Balanced: decent results + readable code |
| Extras | Both Task 2 and Task 3 extras (architecture comparisons) |

---

## Preliminary: Dataset Setup *(shared by both tasks)*

The `train/` folder currently contains **247 images** in COCO JSON format with the following ball categories: `Black`, `Cue`, `Dot`, `Solid`, `Striped`.

### Options

| Option | Approach | Pros | Cons |
|---|---|---|---|
| A ✅ | Download full dataset from Roboflow (official train/val/test splits) | Reproducible, no data leakage, standard benchmark | Requires download |
| B | Manually split the 247 images (e.g. 70/15/15) | Works offline immediately | Arbitrary splits, no standard benchmark |

**Recommendation:** Option A — download from [Roboflow](https://universe.roboflow.com/bachelorthesis/8-ball-pool-l530o) in **both COCO and YOLO formats** in one step (needed for Task 3). Also consider the supplementary datasets listed in the spec for extra training data.

---

## Task 2 — CNN-Based Ball Counting

> **Input:** Pool table image  
> **Output:** Single integer — total number of balls on the table  
> **Deliverables:** `task2.py` (one script) + `model.pth` + section in report

---

### Step 1: Extract Ground Truth Counts

Parse the COCO JSON to sum all ball annotations per image (Black + Cue + Dot + Solid + Striped).

| Option | Approach | Pros | Cons |
|---|---|---|---|
| A ✅ | Count all ball category annotations per image | Simple, directly uses existing labels | Needs careful filtering of the parent `balls` category |
| B | Count only non-cue balls | Matches game logic | Slightly more complex filtering |

---

### Step 2: Problem Formulation

| Option | Approach | Loss | Pros | Cons |
|---|---|---|---|---|
| A | **Classification** (0–15 classes) | CrossEntropy | Well-understood, confidence scores, maps cleanly to pool ball counts | No ordinal relation between classes |
| B | **Regression** (predict float, round to int) | MSE / Smooth L1 | Natural for counting, single output neuron | May predict non-integers; no structured output |
| C | **Ordinal regression** (hybrid) | Custom ordinal loss | Best of both worlds | More complex to implement |

**Recommendation:** Start with **Classification** as the primary approach. Also train a **Regression head** for the "extra" comparison — easy to implement, creates a meaningful narrative.

---

### Step 3: Architecture Selection *(drives the "extra" comparison)*

| Option | Architecture | Pretrained | Pros | Cons |
|---|---|---|---|---|
| A | **Custom CNN (from scratch)** | No | Full control, educational, no ImageNet bias | Needs more data, lower ceiling with 247 images |
| B ✅ | **ResNet-50** | Yes (ImageNet) | Strong baseline, widely understood, fast convergence | May overfit ImageNet features |
| C ✅ | **EfficientNet-B0** | Yes (ImageNet) | Best accuracy/compute tradeoff, compact | Slightly more complex architecture |
| D | **MobileNetV3** | Yes (ImageNet) | Very lightweight | Lower accuracy ceiling |
| E | **ViT-B/16** | Yes (ImageNet21k) | State-of-the-art, global attention (Lab 11) | Needs more data, heavier compute on Colab |

**Recommendation for the "extra":** Compare **3 architectures**:
1. Custom CNN (from-scratch baseline)
2. ResNet-50 (fine-tuned)
3. EfficientNet-B0 (fine-tuned)

This creates a clear from-scratch vs. transfer-learning narrative and two fine-tuned models with different efficiency profiles — all trainable on Colab free tier.

---

### Step 4: Training Pipeline

| Decision | Recommended Choice | Alternative |
|---|---|---|
| Optimizer | Adam (lr ≈ 1e-4) for fine-tuning | SGD + momentum for scratch |
| LR schedule | CosineAnnealingLR | ReduceLROnPlateau |
| Augmentation | Horizontal flip, color jitter, random crop | + Mixup / CutOut for regularization |
| Normalization | ImageNet mean/std (pretrained) | Dataset mean/std (scratch) |
| Batch size | 16–32 | Adjust to Colab VRAM |
| Epochs | 30–50 (fine-tuning) | 80–100 (from scratch) |

> **Important:** Color augmentation is critical — felt color and lighting vary widely across images.

---

### Step 5: Evaluation Metrics *(for the "extra" comparison table)*

| Metric | Description | Notes |
|---|---|---|
| **MAE** | Mean Absolute Error | Primary counting metric — "off by X balls on average" |
| **MSE / RMSE** | Mean / Root Mean Squared Error | Penalizes large errors more |
| **Exact Accuracy** | % images with perfectly correct count | Strict |
| **Off-by-one Accuracy** | % images where \|pred − gt\| ≤ 1 | More meaningful for counting |
| **Confusion Matrix** | Which counts are confused with each other | Classification only |

Build a results table comparing all 3 architectures on these metrics for the report.

---

### Step 6: Deliverables

- [ ] `task2.py` — single script: loads model, runs inference, outputs required JSON format
- [ ] `model.pth` — saved weights of best model
- [ ] Report section: dataset, architecture choices, training details, results table, error analysis

---

## Task 3 — Ball Detection + Table Retrieval

> **Deliverable:** Notebook with results (not a script)

---

## Part A: Ball Detection

> **Goal:** Detect all balls with bounding boxes and category labels

---

### Step 1: Data Preparation

| Option | Format | Notes |
|---|---|---|
| A ✅ | **YOLO format** (for YOLOv8 / RT-DETR) | Export directly from Roboflow |
| B ✅ | **COCO JSON format** (for DETR variants) | Already available in `train/` |

**Recommendation:** Export both formats from Roboflow at once.

---

### Step 2: Architecture Selection *(drives the "extra" comparison)*

| Option | Model | Type | Pros | Cons |
|---|---|---|---|---|
| A ✅ | **YOLOv8-s** (Ultralytics) | Single-stage CNN | Fastest training/inference, 5-line API, excellent small-object detection | Less interpretable |
| B ✅ | **RT-DETR-l** (Ultralytics) | Transformer, real-time | No NMS, global attention, same API as YOLO | Longer to converge |
| C | **DETR** (Facebook original) | Transformer | Covered in lectures, end-to-end | ~300 epochs to converge, impractical on Colab |
| D | **Faster R-CNN** | Two-stage CNN | Strong baseline, well-studied | Slower, more complex pipeline |

**Recommendation for the "extra":** Compare **YOLOv8-s** vs **RT-DETR-l** — both use the exact same Ultralytics API, making a fair comparison trivial to implement and report.

---

### Step 3: Training

```python
# YOLOv8 — example
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(data="pool_balls.yaml", epochs=50, imgsz=640, batch=16)

# RT-DETR — same API
from ultralytics import RTDETR
model = RTDETR("rtdetr-l.pt")
model.train(data="pool_balls.yaml", epochs=50, imgsz=640, batch=8)
```

Key augmentations specific to detection: mosaic, mixup, random scale, color jitter (all built into Ultralytics).

---

### Step 4: Evaluation Metrics

| Metric | Description |
|---|---|
| **mAP@50** | Main detection metric (IoU threshold = 0.5) |
| **mAP@50:95** | Stricter, averaged across IoU thresholds 0.5→0.95 |
| **Precision / Recall** | Per class and overall |
| **F1 Score** | Balance of precision and recall |
| **Inference speed (ms/image)** | Critical for architecture comparison |

---

## Part B: Table Retrieval

> **Goal:** Given a query pool table image, find the most visually similar image from the training set  
> **Evaluation:** Qualitative only (good and bad examples) — as per the spec

---

### Step 1: Feature Extraction

| Option | Method | Pros | Cons |
|---|---|---|---|
| A ✅ | **Global color histogram** (HSV space) | Fast, no model needed, ball colors are distinctive | Ignores spatial layout, sensitive to lighting |
| B ✅ | **CNN global embedding** (penultimate layer of Task 2 backbone) | Semantic features, robust, reuses existing work | Needs pretrained model (already have it) |
| C | **CLIP embeddings** (ViT) | Powerful semantic understanding, zero-shot | Large model, may be overkill |
| D | **Color histogram + CNN embedding** (concatenated) | Best of both worlds | More complex pipeline |

**Recommendation (moderate effort):** Compare **Option A** (baseline) vs **Option B** (reuse the Task 2 backbone — the backbone is already trained, you just extract features from the penultimate layer). This saves work and creates a coherent narrative with Task 2.

---

### Step 2: Similarity Metric

| Option | Metric | Best for |
|---|---|---|
| A ✅ | **Cosine similarity** | High-dimensional CNN embeddings |
| B | **L2 / Euclidean distance** | Lower-dimensional features |
| C ✅ | **Histogram intersection / Chi-squared** | Color histograms specifically |

---

### Step 3: Retrieval System

```
Query image
    → Extract features (histogram or CNN embedding)
    → Compute similarity to all N training images (brute-force, N=247)
    → Return top-K most similar images
```

| Option | Implementation | Pros | Cons |
|---|---|---|---|
| A ✅ | **Brute-force** (numpy dot products / distance) | Simple, sufficient for 247 images | Doesn't scale |
| B | **FAISS index** | Scales to millions of images | Overkill here |

---

### Step 4: Qualitative Evaluation (notebook)

Show in the notebook:
- Top-3 retrievals for **3–5 different query images**
- At least **1 failure case** with analysis of why it failed
- **Side-by-side comparison**: color histogram retrieval vs. CNN embedding retrieval

---

## Suggested Execution Order

```
Week 1-2
├── [ ] Download full dataset from Roboflow (COCO + YOLO formats)
├── [ ] Parse COCO JSON → extract ball counts per image
└── [ ] Build PyTorch Dataset class + DataLoader

Week 2-3
├── [ ] Task 2: Train Custom CNN (baseline)
├── [ ] Task 2: Fine-tune ResNet-50
└── [ ] Task 2: Fine-tune EfficientNet-B0

Week 3
├── [ ] Task 2: Evaluate all 3 models → build comparison table
└── [ ] Task 2: Export best model to .pth, write task2.py

Week 3-4
├── [ ] Task 3A: Fine-tune YOLOv8-s
├── [ ] Task 3A: Fine-tune RT-DETR-l
└── [ ] Task 3A: Compare mAP, precision/recall, speed

Week 4
├── [ ] Task 3B: Implement color histogram retrieval
├── [ ] Task 3B: Implement CNN embedding retrieval (reuse Task 2 backbone)
└── [ ] Task 3B: Build qualitative evaluation in notebook

Week 4-5 (buffer)
├── [ ] Write 3-page report
├── [ ] Polish notebook and scripts
└── [ ] Final review before June 22 deadline
```

---

## Key Reuse Opportunity

> The **pretrained CNN backbone from Task 2** can be directly reused as the feature extractor for Task 3B retrieval. Extract the global average pooled feature vector (before the classification head) and use it as the image embedding. This saves significant effort and makes for a coherent methodology story in the report.

---

## Grading Reminder

| Component | Weight |
|---|---|
| Task 1 | 40% |
| **Tasks 2 + 3** | **50%** |
| Presentation | 10% |

Tasks 2+3 are evaluated on: methodology, report quality, and results quality.
