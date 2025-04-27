# general imports
import argparse
import os
import torch
import torch.nn as nn
from torch.amp import autocast
from torchmetrics.classification import BinaryF1Score, BinaryAUROC
from torchvision.models import (shufflenet_v2_x0_5,
                                mobilenet_v3_small,
                                mnasnet0_5,
                                squeezenet1_1,
                                mobilenet_v2,
                                regnet_y_400mf,
                                efficientnet_b0)

# project imports
from ref_models.Ladevic import CNNModel
from ref_models.Mulki import MobileNetV2Classifier
from dataset_util.dataset_loader import SampledGenImage
from dataset_util.attacked_dataset_loader import AttackedGenImage

# FLOPs/Params
from thop import profile, clever_format

# -------------------- ARGUMENTS -------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cuda", type=bool, default=False, help="Use CUDA if available")
parser.add_argument("--test_data_img", type=str, required=True, help="Path to spatial test data")
parser.add_argument("--test_data_spec", type=str, required=True, help="Path to spectral test data")
parser.add_argument("-m", "--models_dir", required=True, type=str, help="Trained models path")

# ensemble flags
parser.add_argument("--best_img_model", type=str, help="Path to first model (e.g., spatial)")
parser.add_argument("--best_spec_model", type=str, help="Path to second model (e.g., frequency)")

args = parser.parse_args()
device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
ATTACKS = [
    'crop', 'blur', 'noise', 'compress', 'combined'
]

# -------------------- LOAD MODEL -------------------- #
def load_model(checkpoint_path):
    if 'ShuffleNet' in checkpoint_path:
        model = shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif 'MobileNetV3' in checkpoint_path:
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    elif 'MNASNet' in checkpoint_path:
        model = mnasnet0_5(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif'SqueezeNet' in checkpoint_path:
        model = squeezenet1_1(weights=None)
        model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1))
    elif 'MobileNetV2' in checkpoint_path:
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif 'RegNet' in checkpoint_path:
        model = regnet_y_400mf(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif 'EfficientNet' in checkpoint_path:
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif 'Ladevic' in checkpoint_path:
        model = CNNModel()
    elif 'Mulki' in checkpoint_path:
        model = MobileNetV2Classifier()
    else:
        raise ValueError("Unknown model type")

    # load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loading checkpoint: {checkpoint_path}")
    return model

# -------------------- LOAD DATA -------------------- #
def load_data():
    test_dataset_img = SampledGenImage(data_dir=args.test_data_img)
    test_dataset_spec = SampledGenImage(data_dir=args.test_data_spec)
    attack_dataset = AttackedGenImage(data_dir=args.test_data_img)  # only apply attacks on raw image


    print("-" * 64, flush=True)
    print(f'Number of spatial test images:{len(test_dataset_img)}')
    print(f'Number of spectral test images:{len(test_dataset_spec)}')
    print(f'Number of attacked images:{len(attack_dataset)}')
    print("-" * 64, flush=True)

    test_loader_img = torch.utils.data.DataLoader(test_dataset_img,
                                                  batch_size=512,
                                                  shuffle=True,
                                                  num_workers=4,
                                                  pin_memory=True,
                                                  prefetch_factor=4,
                                                  persistent_workers=True
                                                  )
    test_loader_spec = torch.utils.data.DataLoader(test_dataset_spec,
                                                   batch_size=512,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   pin_memory=True,
                                                   prefetch_factor=4,
                                                   persistent_workers=True
                                                   )
    attack_loader = torch.utils.data.DataLoader(attack_dataset,
                                                batch_size=512,
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True,
                                                prefetch_factor=4,
                                                persistent_workers=True
                                                )
    return test_loader_img, test_loader_spec, attack_loader

def parse_flops(flops_str):
    units = {'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12}
    unit = flops_str[-1]
    if unit in units:
        return float(flops_str[:-1]) / 1e9 * units[unit]
    return float(flops_str) / 1e9

# -------------------- TEST CLEAN -------------------- #
@torch.no_grad()
def test(model, dataloader):
    model.eval()
    f1 = BinaryF1Score().to(device)
    auc = BinaryAUROC().to(device)

    total = 0
    correct = 0
    all_preds = []
    all_targets = []

    # FLOPs and Params from one sample
    for example_input, _ in dataloader:
        example_input = example_input.to(device)
        example_input = example_input[:1]
        break

    flops, params = profile(model, inputs=(example_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    # Run evaluation loop
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        with autocast(device_type=device):
            outputs = model(images)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_preds.append(probs)
        all_targets.append(labels)

    acc = 100. * correct / total
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1_score = f1(all_preds, all_targets).item()
    auc_score = auc(all_preds, all_targets).item()

    # Efficiency metrics
    if 'M' in params:
        total_params_mil = float(params.replace('M', ''))
    elif 'K' in params:
        total_params_mil = float(params.replace('K', '')) / 1e3
    else:
        total_params_mil = float(params) / 1e6

    total_flops_g = parse_flops(flops)

    acc_per_mparam = acc / total_params_mil
    acc_per_gflop = acc / total_flops_g

    print("-" * 64)
    print(f"Test Accuracy      : {acc:.2f}%")
    print(f"F1 Score           : {f1_score:.4f}")
    print(f"AUROC              : {auc_score:.4f}")
    print(f"Total Parameters   : {params}")
    print(f"Total FLOPs        : {flops}")
    print(f"Acc / MParam       : {acc_per_mparam:.4f}% per MParam")
    print(f"Acc / GFLOP        : {acc_per_gflop:.4f}% per GFLOP")
    print("-" * 64)

# -------------------- TEST ATTACKED -------------------- #
@torch.no_grad()
def test_attacked(model, dataloader):
    model.eval()
    f1 = BinaryF1Score().to(device)
    auc = BinaryAUROC().to(device)

    # Only attack-specific storage
    attack_correct = {}
    attack_total = {}
    attack_preds = {}
    attack_targets = {}

    for images, labels, attacks in dataloader:
        images, labels = images.to(device), labels.to(device)

        with autocast(device_type=device):
            outputs = model(images)

        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        for i in range(len(attacks)):
            attack = attacks[i]
            attack_name = attack if isinstance(attack, str) else attack.lower()

            if attack_name not in attack_correct:
                attack_correct[attack_name] = 0
                attack_total[attack_name] = 0
                attack_preds[attack_name] = []
                attack_targets[attack_name] = []

            attack_correct[attack_name] += (preds[i] == labels[i]).item()
            attack_total[attack_name] += 1
            attack_preds[attack_name].append(probs[i].unsqueeze(0))
            attack_targets[attack_name].append(labels[i].unsqueeze(0))

    # Only print attack-wise metrics
    print("-" * 64)

    for attack in ATTACKS:
        if attack not in attack_total or attack_total[attack] == 0:
            continue

        attack_preds_tensor = torch.cat(attack_preds[attack])
        attack_targets_tensor = torch.cat(attack_targets[attack])

        attack_acc = 100. * attack_correct[attack] / attack_total[attack]
        attack_f1 = f1(attack_preds_tensor, attack_targets_tensor).item()
        attack_auc = auc(attack_preds_tensor, attack_targets_tensor).item()

        print(f"Attack: {attack.upper()}")
        print(f"  Accuracy    : {attack_acc:.2f}%")
        print(f"  F1 Score    : {attack_f1:.4f}")
        print(f"  AUROC       : {attack_auc:.4f}")
        print("-" * 64)

# -------------------- MAIN -------------------- #
if __name__ == "__main__":

    print("Loading Data...", flush=True)
    test_data_img, test_data_spec, attack_data= load_data()

    models = []
    for filename in os.listdir(args.models_dir):
        if filename.endswith('.pth'):
           models.append(os.path.join(args.models_dir, filename))

    print(f"Models to be tested: {models}")
    print("Starting Testing")

    # print("-" * 64)
    # print(f"Clean Images")
    # print("-" * 64)
    # for checkpoint in models:
    #     model = load_model(checkpoint)
    #     model.to(device)
    #     if 'img' in checkpoint:
    #         test(model, test_data_img)
    #     elif 'freq' in checkpoint:
    #         test(model, test_data_spec)

    print(f"Attacked Images")
    print("-" * 64)
    for checkpoint in models:
        model = load_model(checkpoint)
        model.to(device)
        if 'img' in checkpoint:
            test_attacked(model, attack_data)


