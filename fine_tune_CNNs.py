import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from dataset import PoolBallCountingDataset
from train_CNNs import epoch_iter, train, plotTrainingHistory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pool Ball Counter Models")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['convnext', 'mobilenet', 'efficientnet', 'shufflenet'],
                        help="Choose the model architecture to train.")
    
    args = parser.parse_args()
    print(f"--- Initializing Fine Tuning Pipeline for: {args.model.upper()} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine Tuning on: {device}")

    BATCH_SIZE = 32
    FINE_TUNE_EPOCHS = 20
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
    model.load_state_dict(torch.load(f"{args.model}_best_model.pth"))

    for param in model.parameters():
        param.requires_grad = True

    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    criterion = nn.MSELoss()

    custom_train_history, custom_val_history = train(model, args.model, FINE_TUNE_EPOCHS, train_loader, valid_loader, criterion, optimizer_ft, PATIENCE, ft_flag=True)
    model_title=args.model.upper()+" [Fine Tuned]"
    plotTrainingHistory(custom_train_history, custom_val_history, model_title)

    print("Tuning Complete!")