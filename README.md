# CV: Task 2 & 3 - Pool Ball Detection and Counting

This repository contains the complete pipeline for a Computer Vision project focused on detecting and counting pool balls. It is divided into two main tasks:
* **Task 2:** Regression-based counting using CNNs.
* **Task 3:** Object detection using YOLO and Transformer-based models.

---

## Dependencies

Ensure you have Python 3.8+ installed. Install the required packages using the following command:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow ultralytics --quiet
```

## Datasets
The pipeline expects the datasets to be organized in an ```images``` folder located in the root directory. Within this folder, the data must be split into ```train```, ```valid```, and ```test``` subfolders. Both Task 2 and Task 3 utilize this structure.

## Task 2 Pipeline
### Step 1: Data Preparation & Cleaning
We are combining multiple datasets from Roboflow, so we run the merging script first. This script unifies all labels into a single ball class and drops noisy annotations (like "dots"). We are left with JSON files for the train/val/test splits.
```python
python ball_counting_pipeline.py merge --json-files [LIST JSON FILES] --output [NAME OF MERGED JSON FILE]
```
### Step 2: Transfer Learning
This step freezes the backbone of the selected pre-trained model and trains only a custom linear regression head. We plot the history plots (Loss and MAE curves).
```python
python ball_counting_pipeline.py train --model [MODEL_NAME]
```
***MODEL_NAME** can be any of the following options: ['convnext', 'mobilenet', 'efficientnet', 'shufflenet'].*

### Step 3: Fine-Tuning
Once the base head is stable, this step loads your best weights, unfreezes the entire architecture, and trains it using a much lower learning rate to help the model learn the specific textures of pool balls. We also plot the history plots (Loss and MAE curves) for fine-tuning.
```python
python ball_counting_pipeline.py finetune --model [MODEL_NAME]
```

### Step 4: Evaluation
This evaluates the model on the unseen test set and outputs the final MAE, RMSE, and $R^2$ scores. It generates a summary text file and a regression scatter plot.
```python
python ball_counting_pipeline.py evaluate --model [MODEL_NAME] [--ft]
```
***[--ft]** is a flag to indicate if we want to evaluate the fine-tuned model.*

## Task 3
Task 3 focuses on precise ball localization and detection and is completely self-contained.

All methodologies, model comparisons (YOLOv8-s vs. RT-DETR-l), and evaluations can be found and executed within the following Jupyter Notebook: ```task3_ball_detection_and_retrieval.ipynb```.
