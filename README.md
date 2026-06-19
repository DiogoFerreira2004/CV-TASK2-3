# CV: TASK 2

Task: Regression Problem optimized using MSE, evaluating outputs based on MAE, RMSE, and $R^2$ scores.

## Dependencies
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow
```

## Task 2 Pipeline
### Step 1: Data Preparation & Cleaning
We are combining multiple datasets from Roboflow, so we run the merging script first. This script unifies all labels into a single ball class and drops noisy annotations (like "dots"). We are left with JSON files for the train/val/test splits.
```python
python merge_jsons.py
```
### Step 2: Transfer Learning
This step freezes the backbone of the selected pre-trained model and trains only a custom linear regression head. We plot the history plots (Loss and MAE curves).
```python
python train_CNNs.py --model [MODEL_NAME]
```
***MODEL_NAME** can be any of the following options: ['convnext', 'mobilenet', 'efficientnet', 'shufflenet'].*

### Step 3: Fine-Tuning
Once the base head is stable, this step loads your best weights, unfreezes the entire architecture, and trains it using a much lower learning rate to help the model learn the specific textures of pool balls. We also plot the history plots (Loss and MAE curves) for fine-tuning.
```python
python fine_tune_CNNs.py --model [MODEL_NAME]
```

### Step 4: Evaluation
This evaluates the model on the unseen test set and outputs the final MAE, RMSE, and $R^2$ scores. It generates a summary text file and a regression scatter plot.
```python
python test_CNNs.py --model [MODEL_NAME] [--ft]
```
***[--ft]** is a flag to indicate if we want to evaluate the fine-tuned model.*
