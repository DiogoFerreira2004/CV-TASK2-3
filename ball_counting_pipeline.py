"""Complete pool-ball counting pipeline.

Commands:
    python pool_ball_pipeline.py merge
    python pool_ball_pipeline.py train --model shufflenet
    python pool_ball_pipeline.py finetune --model shufflenet
    python pool_ball_pipeline.py evaluate --model shufflenet [--ft]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Keep the original NumPy seed and automatically use a GPU when one is available.
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHOICES = ["convnext", "mobilenet", "efficientnet", "shufflenet"]

BATCH_SIZE = 32
TRAIN_EPOCHS = 50
FINE_TUNE_EPOCHS = 20
PATIENCE = 7


class PoolBallCountingDataset(Dataset):
    """Load COCO images and use the annotation count as the regression target."""

    def __init__(self, json_path, img_dir, transforms=None, return_path=False):
        self.img_dir = img_dir
        self.transforms = transforms
        self.return_path = return_path

        with open(json_path, "r") as file:
            self.coco_data = json.load(file)

        self.images = self.coco_data["images"]

        # Count the annotations belonging to each image.
        self.image_id_to_count = {img["id"]: 0 for img in self.images}
        for annotation in self.coco_data["annotations"]:
            if annotation["image_id"] in self.image_id_to_count:
                self.image_id_to_count[annotation["image_id"]] += 1

        # Match the original cleaning rule by excluding counts above 16.
        self.valid_images = []
        for image in self.images:
            count = self.image_id_to_count[image["id"]]
            if count <= 16:
                self.valid_images.append(image)

        removed_count = len(self.images) - len(self.valid_images)
        if removed_count > 0:
            print(f"Removed {removed_count} noisy images from {json_path}")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, index):
        image_info = self.valid_images[index]
        file_name = image_info["file_name"]
        image_path = os.path.join(self.img_dir, file_name)

        image = Image.open(image_path).convert("RGB")
        count = float(self.image_id_to_count[image_info["id"]])
        count_tensor = torch.tensor([count], dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)

        if self.return_path:
            return image, count_tensor, file_name

        return image, count_tensor


def get_transforms():
    """Return the same training and validation/test transforms as before."""
    train_transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_test_transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transforms, val_test_transforms


def create_model(model_name, freeze_backbone=False):
    """Create a pretrained model and replace its classifier with one output.

    This shared function removes the repeated model-head definitions from the
    original training, fine-tuning, and evaluation scripts.
    """
    if model_name == "convnext":
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        if freeze_backbone:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)

    elif model_name == "mobilenet":
        model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )
        if freeze_backbone:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)

    elif model_name == "efficientnet":
        model = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.DEFAULT
        )
        if freeze_backbone:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x2_0(
            weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT
        )
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 1)
        if freeze_backbone:
            for parameter in model.fc.parameters():
                parameter.requires_grad = True

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def create_train_validation_loaders():
    """Build the data loaders shared by training and fine-tuning."""
    train_transforms, val_test_transforms = get_transforms()

    train_dataset = PoolBallCountingDataset(
        "images/train/train_annotations.json",
        "images/train/",
        train_transforms,
    )
    validation_dataset = PoolBallCountingDataset(
        "images/valid/valid_annotations.json",
        "images/valid/",
        val_test_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    return train_loader, validation_loader


def epoch_iter(dataloader, model, loss_fn, optimizer=None, is_train=True):
    """Run one training or validation epoch and return MSE and MAE."""
    if is_train:
        assert optimizer is not None, "When training, please provide an optimizer."
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    predictions = []
    labels = []

    for images, targets in tqdm(dataloader, leave=False):
        images = images.to(DEVICE)
        targets = targets.view(-1, 1).to(DEVICE)

        outputs = model(images)
        loss = loss_fn(outputs, targets)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions.extend(outputs.detach().cpu().numpy().flatten())
        labels.extend(targets.detach().cpu().numpy().flatten())

    average_loss = total_loss / len(dataloader.dataset)
    mae = mean_absolute_error(labels, predictions)
    return average_loss, mae


def train_model(
    model,
    model_name,
    num_epochs,
    train_dataloader,
    validation_dataloader,
    loss_fn,
    optimizer,
    patience,
    ft_flag=False,
):
    """Train with validation-based checkpointing and early stopping."""
    train_history = {"loss": [], "mae": []}
    validation_history = {"loss": [], "mae": []}
    best_validation_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_mae = epoch_iter(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            is_train=True,
        )
        print(f"Train loss (MSE): {train_loss:.3f} \t Train MAE: {train_mae:.3f}")

        with torch.no_grad():
            validation_loss, validation_mae = epoch_iter(
                validation_dataloader,
                model,
                loss_fn,
                is_train=False,
            )
        print(
            f"Val loss (MSE):   {validation_loss:.3f} "
            f"\t Val MAE:   {validation_mae:.3f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            checkpoint_name = (
                f"{model_name}_best_ft_model.pth"
                if ft_flag
                else f"{model_name}_best_model.pth"
            )
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("models", checkpoint_name))
            print(" -> Validation loss improved. Model saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    "Early stopping triggered! "
                    f"No improvement for {patience} epochs."
                )
                break

        train_history["loss"].append(train_loss)
        train_history["mae"].append(train_mae)
        validation_history["loss"].append(validation_loss)
        validation_history["mae"].append(validation_mae)

    return train_history, validation_history


def plot_training_history(train_history, validation_history, model_name):
    """Save the original two-panel loss and MAE history plot."""
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.title(f"{model_name.upper()} - Loss Function (MSE)")
    plt.plot(train_history["loss"], label="Train")
    plt.plot(validation_history["loss"], label="Validation")
    plt.ylabel("MSE")
    plt.legend(loc="best")

    plt.subplot(2, 1, 2)
    plt.title(f"{model_name.upper()} - Mean Absolute Error (MAE)")
    plt.plot(train_history["mae"], label="Train")
    plt.plot(validation_history["mae"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")

    plt.tight_layout()
    plt.legend(loc="best")

    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", f"{model_name}_training_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()


def run_training(model_name, fine_tune=False):
    """Run either transfer learning or full-model fine-tuning."""
    train_loader, validation_loader = create_train_validation_loaders()

    # Fine-tuning starts from the same regression-head architecture and then
    # loads the best transfer-learning checkpoint before unfreezing all layers.
    model = create_model(model_name, freeze_backbone=True).to(DEVICE)

    if fine_tune:
        checkpoint_path = os.path.join("models", f"{model_name}_best_model.pth")
        model.load_state_dict(torch.load(checkpoint_path))
        for parameter in model.parameters():
            parameter.requires_grad = True
        epochs = FINE_TUNE_EPOCHS
        learning_rate = 1e-5
    else:
        epochs = TRAIN_EPOCHS
        learning_rate = 1e-3

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-2,
    )
    criterion = nn.MSELoss()

    train_history, validation_history = train_model(
        model,
        model_name,
        epochs,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        PATIENCE,
        ft_flag=fine_tune,
    )

    plot_name = model_name.upper() + " [Fine Tuned]" if fine_tune else model_name
    plot_training_history(train_history, validation_history, plot_name)


def evaluate_model(model, test_loader, model_name, ft_flag, output_dir):
    """Evaluate a checkpoint and save metrics, predictions, and a scatter plot."""
    model.eval()

    all_true_counts = []
    all_predicted_counts = []
    output_json_data = []

    with torch.no_grad():
        for images, targets, paths in test_loader:
            images = images.to(DEVICE)

            outputs = model(images).cpu().numpy().flatten()
            targets = targets.cpu().numpy().flatten()

            # Pool-ball counts are constrained to the dataset's valid range.
            outputs = np.clip(outputs, 0, 16)
            all_true_counts.extend(targets)
            all_predicted_counts.extend(outputs)

            rounded_predictions = np.round(outputs).astype(int)
            for index, path in enumerate(paths):
                output_json_data.append(
                    {
                        "image_path": f"images/test/{path}",
                        "num_balls": int(rounded_predictions[index]),
                    }
                )

    y_true = np.array(all_true_counts)
    y_pred = np.array(all_predicted_counts)
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

    report_name = (
        f"{model_name}_ft_metrics.txt"
        if ft_flag
        else f"{model_name}_base_metrics.txt"
    )
    os.makedirs("eval", exist_ok=True)
    with open(os.path.join("eval", report_name), "w") as report:
        report.write(f"Model: {model_name.upper()} | Fine-Tuned: {ft_flag}\n")
        report.write("-" * 40 + "\n")
        report.write(f"Mean Absolute Error (MAE):  {mae:.4f}\n")
        report.write(f"Root Mean Squared (RMSE):   {rmse:.4f}\n")
        report.write(f"R^2 Score:                   {r2:.4f}\n")
        report.write(f"Exact Accuracy:             {exact_accuracy:.2f}%\n")
        report.write(f"Off-by-One Accuracy:        {off_by_one_accuracy:.2f}%\n")

    json_filename = (
        f"{model_name}_ft_predictions.json"
        if ft_flag
        else f"{model_name}_base_predictions.json"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, json_filename), "w") as json_file:
        json.dump(output_json_data, json_file, indent=4)

    plt.figure(figsize=(10, 8))
    plt.plot(
        [0, 16],
        [0, 16],
        "r--",
        lw=2,
        label="Perfect Prediction (y=x)",
    )
    plt.scatter(
        y_true,
        y_pred,
        alpha=0.5,
        color="blue",
        edgecolor="black",
        s=60,
        label="Model Predictions",
    )
    title_suffix = " [Fine Tuned]" if ft_flag else ""
    plt.title(
        f"Regression Performance: {model_name.upper()}{title_suffix}",
        fontsize=16,
    )
    plt.xlabel("Actual Number of Balls", fontsize=14)
    plt.ylabel("Predicted Number of Balls (Continuous)", fontsize=14)
    plt.xlim(-0.5, 16.5)
    plt.ylim(-0.5, 16.5)
    plt.xticks(range(0, 17, 2))
    plt.yticks(range(0, 17, 2))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)

    plot_name = (
        f"{model_name}_ft_scatter.png"
        if ft_flag
        else f"{model_name}_base_scatter.png"
    )
    os.makedirs("plots", exist_ok=True)
    plt.savefig(
        os.path.join("plots", plot_name),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def run_evaluation(model_name, fine_tuned=False):
    """Load the requested checkpoint and evaluate it on the test split."""
    _, test_transforms = get_transforms()
    test_dataset = PoolBallCountingDataset(
        "images/test/test_annotations.json",
        "images/test/",
        test_transforms,
        return_path=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    model = create_model(model_name)
    checkpoint_name = (
        f"{model_name}_best_ft_model.pth"
        if fine_tuned
        else f"{model_name}_best_model.pth"
    )
    checkpoint_path = os.path.join("models", checkpoint_name)

    try:
        model.load_state_dict(torch.load(checkpoint_path))
    except FileNotFoundError:
        print(
            f"ERROR: Could not find weight file '{checkpoint_path}'. "
            "Did you train this model yet?"
        )
        return

    model = model.to(DEVICE)
    evaluate_model(
        model,
        test_loader,
        model_name,
        fine_tuned,
        "json_output",
    )


def merge_coco_jsons(json_files, output_file):
    """Merge COCO files, discard dot annotations, and use one ball category."""
    print("Starting the unified merge process...")

    merged_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "ball", "supercategory": "none"}],
        "info": {"description": "Merged 8-Ball Pool Dataset - Single Class"},
    }

    image_id_offset = 0
    annotation_id_offset = 0

    for file_path in json_files:
        print(f"Processing: {file_path}")
        with open(file_path, "r") as file:
            data = json.load(file)

        local_category_names = {}
        for category in data.get("categories", []):
            local_category_names[category["id"]] = category["name"].lower().strip()

        local_to_merged_image_ids = {}
        for image in data.get("images", []):
            old_image_id = image["id"]
            new_image_id = old_image_id + image_id_offset
            local_to_merged_image_ids[old_image_id] = new_image_id

            image["id"] = new_image_id
            merged_data["images"].append(image)

        for annotation in data.get("annotations", []):
            category_name = local_category_names.get(annotation["category_id"], "")
            if category_name == "dot":
                continue

            annotation["id"] += annotation_id_offset
            annotation["image_id"] = local_to_merged_image_ids[
                annotation["image_id"]
            ]
            annotation["category_id"] = 1
            merged_data["annotations"].append(annotation)

        if data.get("images"):
            image_id_offset = max(
                image["id"] for image in merged_data["images"]
            ) + 1
        if data.get("annotations") and merged_data["annotations"]:
            annotation_id_offset = max(
                annotation["id"] for annotation in merged_data["annotations"]
            ) + 1

    with open(output_file, "w") as file:
        json.dump(merged_data, file, indent=4)

    print(f"\nSuccess! Merged into --> {output_file}")
    print(f"Total Images: {len(merged_data['images'])}")
    print(f"Total Annotations: {len(merged_data['annotations'])}")
    print(
        "Unique Categories: "
        f"{[category['name'] for category in merged_data['categories']]}"
    )


def build_argument_parser():
    """Define one command-line interface for the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Pool Ball Counting: data preparation, training, and evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge the original test COCO annotation files.",
    )
    merge_parser.add_argument(
        "--json-files",
        nargs="+",
        default=[
            "images/test/_annotations.coco (2).json",
            "images/test/_annotations.coco (3).json",
            "images/test/_annotations.coco (4).json",
            "images/test/_annotations.coco.json",
        ],
        help="COCO JSON files to merge.",
    )
    merge_parser.add_argument(
        "--output",
        default="images/test/test_annotations.json",
        help="Path for the merged COCO JSON file.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a regression head with a frozen backbone.",
    )
    train_parser.add_argument("--model", required=True, choices=MODEL_CHOICES)

    fine_tune_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune all layers from the best base checkpoint.",
    )
    fine_tune_parser.add_argument("--model", required=True, choices=MODEL_CHOICES)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a base or fine-tuned checkpoint.",
    )
    evaluate_parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    evaluate_parser.add_argument(
        "--ft",
        action="store_true",
        help="Evaluate the fine-tuned checkpoint.",
    )

    return parser


def main():
    """Dispatch the selected pipeline command."""
    args = build_argument_parser().parse_args()

    if args.command == "merge":
        merge_coco_jsons(args.json_files, args.output)
    elif args.command == "train":
        run_training(args.model)
    elif args.command == "finetune":
        run_training(args.model, fine_tune=True)
    elif args.command == "evaluate":
        run_evaluation(args.model, fine_tuned=args.ft)


if __name__ == "__main__":
    main()
