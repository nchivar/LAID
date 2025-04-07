# general imports
import argparse
import torch
import torch.nn as nn
import math
import time
from matplotlib import pyplot as plt
from torch.amp import autocast, GradScaler
from torchmetrics.classification import BinaryF1Score, BinaryAUROC
from torchvision.models import (shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights,
                                mobilenet_v3_small, MobileNet_V3_Small_Weights,
                                mnasnet0_5, MNASNet0_5_Weights,
                                squeezenet1_1, SqueezeNet1_1_Weights,
                                mobilenet_v2, MobileNet_V2_Weights,
                                regnet_y_400mf, RegNet_Y_400MF_Weights,
                                efficientnet_b0, EfficientNet_B0_Weights)

# project imports
from ref_models.Ladevic import CNNModel
from ref_models.Mulki import MobileNetV2Classifier
from dataset.dataset_loader import TinyGenImage


# -------------------- GLOBAL VARS  -------------------- #
MODELS = ['ShuffleNet', 'MobileNetV3', 'MNASNet', 'SqueezeNet', 'MobileNetV2', 'RegNet', 'EfficientNet',
          'Ladevic', 'Mulki']  # reference methods
TRAIN_PATIENCE = 10

# -------------------- ARGUMENTS -------------------- #
parser = argparse.ArgumentParser()

# hyper-parameters
parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("-lr", "--learning_rate", type=float,  default=1e-4, help="Learning rate for the optimizer")
parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='Weight decay for optimizer')

# device
parser.add_argument("-c", "--cuda", type=bool, default=False, help="Use Cuda GPU for training if available")

# dataset
parser.add_argument("--train_data", type=str, default='dataset/preprocessed/images/train',
                    help="Path of train split for dataset")
parser.add_argument("--val_data",  type=str,  default='dataset/preprocessed/images/val',
                    help="Path of val split for dataset")

# model
parser.add_argument("-m", "--model", required=True, type=str, choices=MODELS,
                    help= f"Model to train on AIGC detection: {MODELS}")

# output
parser.add_argument('--output_model', required=True, type=str, help = "File path of trained model")
parser.add_argument('--output_plot', required=True, type=str, help = "File path of model loss plot")

args = parser.parse_args()

# pretraining summary
if args.cuda and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("-" * 64, flush=True)
print(f"Pretraining Summary:", flush=True)
print(f"Number of Epochs: {args.epochs}", flush=True)
print(f"Batch Size: {args.batch_size}", flush=True)
print(f"Learning Rate: {args.learning_rate}", flush=True)
print(f"Weight Decay: {args.weight_decay}", flush=True)
print(f"Device: {device}", flush=True)
print(f"Model: {args.model}", flush=True)
print("-" * 64, flush=True)


def load_model():

    if args.model == 'ShuffleNet':
        model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)

    elif args.model == 'MobileNetV3':
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)

    elif args.model == 'MNASNet':
        model = mnasnet0_5(weights=MNASNet0_5_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    elif args.model == 'SqueezeNet':
        model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))

    elif args.model == 'MobileNetV2':
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    elif args.model == 'RegNet':
        model = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, 2)

    elif args.model == 'EfficientNet':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    elif args.model == 'Ladevic':
        model = CNNModel()

    elif args.model == 'Mulki':
        model = MobileNetV2Classifier()

    else:
        return None

    return model

def load_data():

    train_dataset = TinyGenImage(data_dir=args.train_data)
    val_dataset = TinyGenImage(data_dir=args.val_data)
    train_batches = math.ceil(len(train_dataset) / args.batch_size)
    val_batches = math.ceil(len(val_dataset) / args.batch_size)

    print("-" * 64, flush=True)
    print(f"Number of training batches: ", train_batches, flush=True)
    print(f"Number of validation batches: ", val_batches, flush=True)
    print("-" * 64, flush=True)

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_data, val_data


def train(model, train_data, val_data, optimizer, scheduler, loss_fn):
    scaler = GradScaler()
    f1 = BinaryF1Score().to(device)
    auc = BinaryAUROC().to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range (args.epochs):

        print (f"\nEpoch {epoch}/{args.epochs}")

        # training
        model.train()
        train_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0
        train_all_preds = []
        train_all_targets = []

        for batch_idx, (images, labels) in enumerate(train_data):

            start_time = time.time()

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # accumulate batch loss
            train_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)  # if output is logits for multi-class
            train_correct_predictions += preds.eq(labels).sum().item()
            train_total_predictions += labels.size(0)

            # Collect for metrics
            probs = torch.softmax(outputs.detach(), dim=1)[:, 1]
            train_all_preds.append(probs)
            train_all_targets.append(labels.detach())

            # logging
            end_time = time.time()
            if (batch_idx) % 10 == 0:
                print(f"Completed training batch {batch_idx}/{len(train_data)} - Total Time: {end_time-start_time: 2f}")

        # validation
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        val_all_preds = []
        val_all_targets = []

        for batch_idx, (images, labels) in enumerate(val_data):

            start_time = time.time()

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            # accumulate batch loss
            val_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)  # if output is logits for multi-class
            val_correct_predictions += preds.eq(labels).sum().item()
            val_total_predictions += labels.size(0)

            # Collect for metrics
            probs = torch.softmax(outputs.detach(), dim=1)[:, 1]
            val_all_preds.append(probs)
            val_all_targets.append(labels.detach())

            # logging
            end_time = time.time()
            if (batch_idx) % 10 == 0:
                print(f"Completed validation batch {batch_idx}/{len(val_data)}- Total Time: {end_time-start_time: 2f}")

        # results - train
        train_loss = train_loss / train_total_predictions
        train_losses.append(train_loss)
        train_acc = 100. * train_correct_predictions / train_total_predictions
        train_all_preds = torch.cat(train_all_preds)
        train_all_targets = torch.cat(train_all_targets)
        train_f1 = f1(train_all_preds, train_all_targets).item()
        train_auc = auc(train_all_preds, train_all_targets).item()

        # results - val
        val_loss = val_loss / val_total_predictions
        val_losses.append(val_loss)
        val_acc = 100. * val_correct_predictions / val_total_predictions
        val_all_preds = torch.cat(val_all_preds)
        val_all_targets = torch.cat(val_all_targets)
        val_f1 = f1(val_all_preds, val_all_targets).item()
        val_auc = auc(val_all_preds, val_all_targets).item()

        # print
        print(f"\nEpoch {epoch}/{args.epochs} Stats -- Train Loss : {train_loss:.2f} | Train Accuracy : {train_acc:2f} | "
              f"Train F1 : {train_f1: 2f} | Train AUC : {train_auc:2f} | Val Loss : {val_loss:.2f} "
              f"| Val Accuracy : {val_acc:2f} | Val F1 : {val_f1: 2f} | Val AUC : {val_auc:2f} |")

        scheduler.step(val_loss)

        # plot saving
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss', marker='o')
        plt.plot(val_losses, label='Validation Loss', marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.output_plot)
        plt.close()

        # model saving
        if val_loss < best_val_loss:
            patience_counter = 0 # reset patience back to 0
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, args.output_model)
            print("Model checkpoint updated")
        # else:
        #     if patience_counter < TRAIN_PATIENCE:
        #         patience_counter+=1
        #     else:
        #         print(f'Early stopping at epoch: {epoch}')
        #         return # exit program if training is failing to make progress



if __name__ == "__main__":
    print("Starting Training...")

    model = load_model()
    model.to(device)
    train_data, val_data = load_data()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # because lower loss is better
        factor=0.1,  # reduce LR by 10×
        patience=5,  # wait 5 epochs of no improvement
        verbose=True  # print when LR is reduced
    )
    loss_fn = nn.CrossEntropyLoss()

    if device == 'cuda':
        loss_fn.cuda()

    train(model=model,
          train_data=train_data,
          val_data=val_data,
          optimizer=optimizer,
          scheduler=scheduler,
          loss_fn=loss_fn)

