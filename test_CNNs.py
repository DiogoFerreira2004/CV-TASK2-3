import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from dataset import PoolBallCountingDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, test_loader, model_name, ft_flag, device="cuda"):
    model.eval()
    
    all_true_counts = []
    all_pred_floats = []
    
    print(f"\n--- Evaluating {model_name.upper()} ---")
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)

            outputs = model(images).cpu().numpy().flatten()
            targets = targets.cpu().numpy().flatten()

            outputs = np.clip(outputs, 0, 16)
            
            all_true_counts.extend(targets)
            all_pred_floats.extend(outputs)

    y_true = np.array(all_true_counts)
    y_pred = np.array(all_pred_floats)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"Mean Absolute Error (MAE):  {mae:.4f}")
    print(f"Root Mean Squared (RMSE):   {rmse:.4f}")
    print(f"R² Score:                   {r2:.4f}")

    report_name = f"{model_name}_ft_metrics.txt" if ft_flag else f"{model_name}_base_metrics.txt"
    os.makedirs("eval", exist_ok=True)
    report_path = os.path.join("eval", report_name)
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name.upper()} | Fine-Tuned: {ft_flag}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Absolute Error (MAE):  {mae:.4f}\n")
        f.write(f"Root Mean Squared (RMSE):   {rmse:.4f}\n")
        f.write(f"R² Score:                   {r2:.4f}\n")
    print(f"Saved metrics summary to: {report_path}")

    plt.figure(figsize=(10, 8))
    
    plt.plot([0, 16], [0, 16], 'r--', lw=2, label='Perfect Prediction (y=x)')
    
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', edgecolor='black', s=60, label='Model Predictions')
    
    plt.title(f'Regression Performance: {model_name.upper()} {"[Fine Tuned]" if ft_flag else ""}', fontsize=16)
    plt.xlabel('Actual Number of Balls', fontsize=14)
    plt.ylabel('Predicted Number of Balls (Continuous)', fontsize=14)
    plt.xlim(-0.5, 16.5)
    plt.ylim(-0.5, 16.5)
    plt.xticks(range(0, 17, 2))
    plt.yticks(range(0, 17, 2))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    plot_name = f"{model_name}_ft_scatter.png" if ft_flag else f"{model_name}_base_scatter.png"
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", plot_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to: {plot_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pool Ball Regression Models")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['convnext', 'mobilenet', 'efficientnet', 'shufflenet'],
                        help="Choose the model architecture to evaluate.")
    parser.add_argument('--ft', action='store_true',
                        help="Include flag to evaluate fine-tuned models.")
    
    args = parser.parse_args()
    print(f"--- Initializing Regression Eval Pipeline for: {args.model.upper()} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    BATCH_SIZE = 32

    test_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = PoolBallCountingDataset("images/test/test_annotations.json", "images/test/", test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    if args.model == 'convnext':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)

    elif args.model == 'mobilenet':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)

    elif args.model == 'efficientnet':
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    elif args.model == 'shufflenet':
        model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)

    weight_file = f"{args.model}_best_ft_model.pth" if args.ft else f"{args.model}_best_model.pth"
    try:
        model.load_state_dict(torch.load(weight_file))
        print(f"Successfully loaded weights from {weight_file}")
    except FileNotFoundError:
        print(f"ERROR: Could not find weight file '{weight_file}'. Did you train this model yet?")
        exit(1)

    model = model.to(device)

    evaluate_model(model, test_loader, args.model, args.ft, device)