import os
import json
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

def evaluate_model(model, test_loader, model_name, ft_flag, output_dir, device="cuda"):
    model.eval()
    
    all_true_counts = []
    all_pred_floats = []

    output_json_data = {}
    
    print(f"\n--- Evaluating {model_name.upper()} ---")
    with torch.no_grad():
        for images, targets, paths in test_loader:
            images = images.to(device)

            outputs = model(images).cpu().numpy().flatten()
            targets = targets.cpu().numpy().flatten()

            outputs = np.clip(outputs, 0, 16)
            
            all_true_counts.extend(targets)
            all_pred_floats.extend(outputs)

            rounded_preds = np.round(outputs).astype(int)
            for i in range(len(paths)):
                output_json_data[paths[i]] = int(rounded_preds[i])

    y_true = np.array(all_true_counts)
    y_pred = np.array(all_pred_floats)

    y_true_int = y_true.astype(int)
    y_pred_int = np.round(y_pred).astype(int)

    exact_matches = np.sum(y_true_int == y_pred_int)
    exact_accuracy = (exact_matches / len(y_true_int)) * 100.0

    absolute_errors = np.abs(y_true_int - y_pred_int)
    off_by_one_matches = np.sum(absolute_errors <= 1)
    off_by_one_accuracy = (off_by_one_matches / len(y_true_int)) * 100.0

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    report_name = f"{model_name}_ft_metrics.txt" if ft_flag else f"{model_name}_base_metrics.txt"
    os.makedirs("eval", exist_ok=True)
    report_path = os.path.join("eval", report_name)

    with open(report_path, "w") as f:
        f.write(f"Model: {model_name.upper()} | Fine-Tuned: {ft_flag}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Absolute Error (MAE):  {mae:.4f}\n")
        f.write(f"Root Mean Squared (RMSE):   {rmse:.4f}\n")
        f.write(f"R^2 Score:                   {r2:.4f}\n")
        f.write(f"Exact Accuracy:             {exact_accuracy:.2f}%\n")
        f.write(f"Off-by-One Accuracy:        {off_by_one_accuracy:.2f}%\n")

    json_filename = f"{model_name}_ft_predictions.json" if ft_flag else f"{model_name}_base_predictions.json"
    json_path = os.path.join(output_dir, json_filename)
    
    with open(json_path, 'w') as json_file:
        json.dump(output_json_data, json_file, indent=4)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pool Ball Regression Models")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['convnext', 'mobilenet', 'efficientnet', 'shufflenet'],
                        help="Choose the model architecture to evaluate.")
    parser.add_argument('--ft', action='store_true',
                        help="Include flag to evaluate fine-tuned models.")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 32

    test_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = PoolBallCountingDataset("images/test/test_annotations.json", "images/test/", test_transforms, return_path=True)
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

    model_name = f"{args.model}_best_ft_model.pth" if args.ft else f"{args.model}_best_model.pth"
    model_path = os.path.join("models", model_name)

    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"ERROR: Could not find weight file '{model_path}'. Did you train this model yet?")
        exit(1)

    model = model.to(device)

    evaluate_model(model, test_loader, args.model, args.ft, device)