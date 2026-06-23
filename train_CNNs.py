import os
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
np.random.seed(42)
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from dataset import PoolBallCountingDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def epoch_iter(dataloader, model, loss_fn, optimizer=None, is_train=True):
    if is_train:
        assert optimizer is not None, "When training, please provide an optimizer."

    num_batches = len(dataloader)

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    preds = []
    labels = []

    for batch, (X, y) in enumerate(tqdm(dataloader, leave=False)):
        X = X.to(device)

        y = y.view(-1, 1).to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * X.size(0)

        preds.extend(pred.detach().cpu().numpy().flatten())
        labels.extend(y.detach().cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader.dataset)
    mae = mean_absolute_error(labels, preds)
    
    return avg_loss, mae

def train(model, model_name, num_epochs, train_dataloader, validation_dataloader, loss_fn, optimizer, patience, ft_flag=False):
    train_history = {'loss': [], 'mae': []}
    val_history = {'loss': [], 'mae': []}
    best_val_loss = np.inf
    epochs_no_improve = 0
    
    for t in range(num_epochs):
        print(f"\nEpoch {t+1}/{num_epochs}")
        
        train_loss, train_mae = epoch_iter(train_dataloader, model, loss_fn, optimizer, is_train=True)
        print(f"Train loss (MSE): {train_loss:.3f} \t Train MAE: {train_mae:.3f}")
        
        with torch.no_grad():
            val_loss, val_mae = epoch_iter(validation_dataloader, model, loss_fn, is_train=False)
            print(f"Val loss (MSE):   {val_loss:.3f} \t Val MAE:   {val_mae:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_path = f"{model_name}_best_ft_model.pth" if ft_flag else f"{model_name}_best_model.pth"
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", model_path)
            torch.save(model.state_dict(), model_path)
            print(" -> Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break

        train_history["loss"].append(train_loss)
        train_history["mae"].append(train_mae)

        val_history["loss"].append(val_loss)
        val_history["mae"].append(val_mae)

    return train_history, val_history

def plotTrainingHistory(train_history, val_history, model_name):
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.title(f'{model_name.upper()} - Loss Function (MSE)')
    plt.plot(train_history['loss'], label='Train')
    plt.plot(val_history['loss'], label='Validation')
    plt.ylabel('MSE')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.title(f'{model_name.upper()} - Mean Absolute Error (MAE)')
    plt.plot(train_history['mae'], label='Train')
    plt.plot(val_history['mae'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    
    plt.tight_layout()
    plt.legend(loc='best')
    
    plot_name=f"{model_name}_training_history.png"
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", plot_name)
    plt.savefig(plot_path, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pool Ball Counter Models")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['convnext', 'mobilenet', 'efficientnet', 'shufflenet'],
                        help="Choose the model architecture to train.")
    
    args = parser.parse_args()

    BATCH_SIZE = 32
    EPOCHS = 50
    PATIENCE = 7 

    train_transforms = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PoolBallCountingDataset("images/train/train_annotations.json", "images/train/", train_transforms)
    valid_dataset = PoolBallCountingDataset("images/valid/valid_annotations.json", "images/valid/", val_test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)

    if args.model == 'convnext':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        for param in model.features.parameters(): param.requires_grad = False
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)

    elif args.model == 'mobilenet':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        for param in model.features.parameters(): param.requires_grad = False
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)

    elif args.model == 'efficientnet':
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        for param in model.features.parameters(): param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    elif args.model == 'shufflenet':
        model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT)
        for param in model.parameters(): param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 1)
        for param in model.fc.parameters(): param.requires_grad = True 

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.MSELoss()

    custom_train_history, custom_val_history = train(model, args.model, EPOCHS, train_loader, valid_loader, criterion, optimizer, PATIENCE)
    plotTrainingHistory(custom_train_history, custom_val_history, args.model)