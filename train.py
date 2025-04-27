# general imports
import os
import gc
import argparse
import torch
import torch.nn as nn
import math
import time
import shutil
import sys
import matplotlib
matplotlib.use('Agg')  # Must go before importing pyplot!
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
from dataset_util.dataset_loader import SampledGenImage
from dataset_util.gen_dataset import generate_dataset


# -------------------- GLOBAL VARS  -------------------- #
MODELS = ['ShuffleNet', 'MobileNetV3', 'MNASNet', 'SqueezeNet', 'MobileNetV2', 'RegNet', 'EfficientNet',
          'Ladevic', 'Mulki']  # reference methods
MODALITIES = ['img', 'freq']
TRAIN_PATIENCE = 10
DATASET_LOCAL = 'dataset'

# -------------------- ARGUMENTS -------------------- #
parser = argparse.ArgumentParser()

# hyper-parameters
parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("-lr", "--learning_rate", type=float,  default=1e-4, help="Learning rate for the optimizer")
parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='Weight decay for optimizer')

# dataset
parser.add_argument("-d", "--data_dir", type=str, default='../825/GenImage/', help='Directory of GenImage')
parser.add_argument('--train_image_count', type=int, default=100000, help='Number of training images')
parser.add_argument('--val_image_count', type=int, default=12500, help='Number of validation images')

# model
parser.add_argument("-m", "--model", required=True, type=str, choices=MODELS,
                    help= f"Model to train on AIGC detection: {MODELS}")
parser.add_argument("-dm", "--modality", required=True, type=str, choices=MODALITIES,
                    help=f'modality of input data, one of :{MODALITIES}')
parser.add_argument("-mc", "--model_checkpoint", type=str, default=None,
                    help='Previous model checkpoint for continued model training')
# output
parser.add_argument('--output_dir', required=True, type=str, help = "Directory of outputs")
parser.add_argument('--output_model', required=True, type=str, help = "File path of trained model")
parser.add_argument('--output_plot', required=True, type=str, help = "File path of model loss plot")

# device
parser.add_argument("-c", "--cuda", type=bool, default=False, help="Use Cuda GPU for training if available")

args = parser.parse_args()

# pretraining summary
if args.cuda and torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

print("-" * 64, flush=True)
print(f"Pretraining Summary:", flush=True)
print(f"Number of Epochs: {args.epochs}", flush=True)
print(f"Batch Size: {args.batch_size}", flush=True)
print(f"Learning Rate: {args.learning_rate}", flush=True)
print(f"Weight Decay: {args.weight_decay}", flush=True)
print(f'Data Directory: {args.data_dir}', flush=True)
print(f'Train Image Count: {args.train_image_count}', flush=True)
print(f'Val Image Count: {args.val_image_count}', flush=True)
print(f"Model: {args.model}", flush=True)
print(f"Modality: {args.modality}", flush=True)
print(f"Output Root: {args.output_dir}", flush=True)
print(f"Output Model: {args.output_model}", flush=True)
print(f"Output Loss Plot: {args.output_plot}", flush=True)
print(f"Device: {device}", flush=True)
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

    # if dataset has not been previously generated, create dataset
    if not os.path.exists(DATASET_LOCAL):
        print(f'Generating dataset...', flush=True)
        generate_dataset(args.data_dir, DATASET_LOCAL, args.train_image_count, args.val_image_count)

    # determine dataset subdirectory based on modality
    subdir = 'img' if args.modality == 'img' else 'spec'
    data_dir = os.path.join(DATASET_LOCAL, subdir)

    print('Dataset found, beginning data loading...', flush=True)

    # Initialize datasets
    train_dataset = SampledGenImage(data_dir=os.path.join(data_dir, 'train'))
    val_dataset = SampledGenImage(data_dir=os.path.join(data_dir, 'val'))

    train_batches = math.ceil(len(train_dataset) / args.batch_size)
    val_batches = math.ceil(len(val_dataset) / args.batch_size)

    if (len(train_dataset) != args.train_image_count) or (len(val_dataset) != args.val_image_count):
        print("-" * 64, flush=True)
        print('Dataset is incorrectly sized (likely from previous generation), please delete the \'./dataset\' folder and re-run.')
        print("-" * 64, flush=True)
        sys.exit()

    print("-" * 64, flush=True)
    print(f'Number of training images:{len(train_dataset)}')
    print(f'Number of validation images:{len(val_dataset)}')
    print(f"Number of training batches: ", train_batches, flush=True)
    print(f"Number of validation batches: ", val_batches, flush=True)
    print("-" * 64, flush=True)

    train_data = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True,
                                             prefetch_factor=4,
                                             persistent_workers=True)
    val_data = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           persistent_workers=True)
    return train_data, val_data

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cuda')  # or 'cpu' if needed

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from {path}")



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
        train_preds = []
        train_labels = []

        batch_timer_start = time.time()  # Start timing before the loop

        for batch_idx, (images, labels) in enumerate(train_data):

            if (batch_idx) % 10 == 0:
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
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs.detach(), dim=1)[:, 1]  # For AUC
            train_preds.append(probs.detach().cpu())
            train_labels.append(labels.detach().cpu())

            train_correct_predictions += preds.eq(labels).sum().item()
            train_loss += loss.item() * images.size(0)

            # logging
            if batch_idx % 10 == 0 and batch_idx != 0:
                batch_timer_end = time.time()
                print(f"Completed training batch {batch_idx + 1}/{len(train_data)} - "
                      f"Time for last 10 batches: {batch_timer_end - batch_timer_start:.2f} seconds")
                batch_timer_start = time.time()  # Reset timer for next 10 batches

            # del images, labels, outputs, probs, preds, loss
            torch.cuda.empty_cache()

        # training epoch stats
        train_loss = train_loss / args.train_image_count
        train_losses.append(train_loss)
        train_acc = 100. * train_correct_predictions / args.train_image_count
        train_all_preds = torch.cat(train_preds)
        train_all_targets = torch.cat(train_labels)
        train_f1 = f1( train_all_preds, train_all_targets.int()).item()
        train_auc = auc( train_all_preds, train_all_targets.int()).item()

        # validation
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_preds = []
        val_labels = []

        batch_timer_start = time.time()  # Start timing before the loop

        for batch_idx, (images, labels) in enumerate(val_data):

            start_time = time.time()
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type=device):
                outputs = model(images)
                loss = loss_fn(outputs, labels)


            # accumulate batch loss
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs.detach(), dim=1)[:, 1]  # For AUC
            val_preds.append(probs.detach().cpu())
            val_labels.append(labels.detach().cpu())

            val_correct_predictions += preds.eq(labels).sum().item()
            val_loss += loss.item() * images.size(0)

            # logging
            if batch_idx % 10 == 0 and batch_idx!=0:
                batch_timer_end = time.time()
                print(f"Completed validation batch {batch_idx + 1}/{len(val_data)} - "
                      f"Time for last 10 batches: {batch_timer_end - batch_timer_start:.2f} seconds")
                batch_timer_start = time.time()  # Reset timer for next 10 batches


        # results - val
        val_loss = val_loss / args.val_image_count
        val_losses.append(val_loss)
        val_acc = 100. * val_correct_predictions / args.val_image_count
        val_all_preds = torch.cat(val_preds)
        val_all_targets = torch.cat(val_labels)
        val_f1 = f1(val_all_preds, val_all_targets.int()).item()
        val_auc = auc(val_all_preds, val_all_targets.int()).item()

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
        plt.savefig(os.path.join(args.output_dir, args.output_plot))
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
            }, os.path.join(args.output_dir, args.output_model))
            print("Model checkpoint updated")
        else:
            if patience_counter < TRAIN_PATIENCE:
                patience_counter+=1
            else:
                print(f'Early stopping at epoch: {epoch}')
                return # exit program if training is failing to make progress

        # cleanup
        gc.collect()
        torch.cuda.empty_cache()
        f1.reset()
        auc.reset()



if __name__ == "__main__":
    print("Loading Model...", flush=True)
    model = load_model()
    model.to(device)
    print("-" * 64, flush=True)

    print("Loading Data...", flush=True)
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

    print("Creating output directory...", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print("-" * 64, flush=True)

    if args.model_checkpoint is not None:
        print("Loading model from checkpoint...", flush=True)
        load_checkpoint(args.model_checkpoint, model, optimizer)
        print("-" * 64, flush=True)

    print("Starting Training...", flush=True)
    try:
        train(model=model,
              train_data=train_data,
              val_data=val_data,
              optimizer=optimizer,
              scheduler=scheduler,
              loss_fn=loss_fn)
    except Exception as e:
        print(f"[FATAL] Training failed before starting: {e}", flush=True)

