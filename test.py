# general imports
import argparse
import os
import torch
import numpy as np
import torch.nn as nn
from torch.amp import autocast
import timm
from torchmetrics.classification import BinaryF1Score, BinaryAUROC
from torchvision.models import (shufflenet_v2_x0_5,
                                mobilenet_v3_small,
                                mnasnet0_5,
                                squeezenet1_1,
                                mobilenet_v2,
                                regnet_y_400mf,
                                efficientnet_b0)

from thop import profile, clever_format  # FLOPs/Params

# project imports
from models.Ladevic import CNNModel
from models.Mulki import MobileNetV2Classifier
from dataset_util.dataset_loader import SampledGenImage
from dataset_util.attacked_dataset_loader import AttackedGenImage
from dataset_util.gen_dataset import convert_to_freq, convert_to_tensor

# -------------------- ARGUMENTS -------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models_dir", required=True, type=str, help="Trained models path")
parser.add_argument("--test_data_img", type=str, required=True, help="Path to spatial test data")
parser.add_argument("--test_data_spec", type=str, required=True, help="Path to spectral test data")
parser.add_argument('--attack', type=bool, default=True, help = "apply adversarial testing on top of clean testing")
parser.add_argument("-c", "--cuda", type=bool, default=False, help="Use CUDA if available")

args = parser.parse_args()
device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
ATTACKS = [
    'crop', 'blur', 'noise', 'compress', 'combined'
]

# load model conditionally based on command-line argument and adjust final FC layer to output 2 outputs (for our task)
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
    elif 'EdgeNeXt' in checkpoint_path:
        model = timm.create_model('edgenext_xx_small', pretrained=False, num_classes=2)
    elif 'MobileViTV1' in checkpoint_path:
        model = timm.create_model('mobilevit_xxs', pretrained=False, num_classes=2)
    elif 'MobileViTV2' in checkpoint_path:
        model = timm.create_model('mobilevitv2_050', pretrained=False, num_classes=2)
    elif 'FastViT' in checkpoint_path:
        model = timm.create_model('fastvit_t8', pretrained=False, num_classes=2)
    else:
        raise ValueError("Unknown model type")

    # load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loading checkpoint: {checkpoint_path}")
    return model


# setup dataloaders to be passed to models for testing
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

# helper method to convert flops to GFLOPs
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

    # FLOPs and params from one sample
    for example_input, _ in dataloader:
        example_input = example_input.to(device)
        example_input = example_input[:1]
        break

    flops, params = profile(model, inputs=(example_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    # evaluation loop
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

    # efficiency metrics
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
def test_attacked(spatial_model, spectral_model, dataloader):
    spatial_model.eval()
    spectral_model.eval()
    f1 = BinaryF1Score().to(device)
    auc = BinaryAUROC().to(device)

    # attack-specific storage
    img_attack_correct, img_attack_total, img_attack_preds, img_attack_targets = {}, {}, {},{}
    freq_attack_correct, freq_attack_total, freq_attack_preds, freq_attack_targets = {}, {}, {},{}
    fusion_attack_correct, fusion_attack_total, fusion_attack_preds, fusion_attack_targets = {}, {}, {}, {}

    for images, labels, attacks in dataloader:

        # send images to device
        img_images = images.to(device)  # spatial
        image_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # spectral
        freq_images = []
        for img in image_np:
            freq_np = convert_to_freq(img, is_tensor=True)
            freq_tensor = convert_to_tensor(freq_np, is_tensor=True)
            freq_images.append(freq_tensor)
        freq_images = torch.stack(freq_images).to(device)  # spectral

        # send labels to device
        labels = labels.to(device)

        # capture model outputs
        with autocast(device_type=device):
            img_outputs = spatial_model(img_images)
            freq_outputs = spectral_model(freq_images)

        # predictions
        img_preds = torch.argmax(img_outputs, dim=1)
        img_probs = torch.softmax(img_outputs, dim=1)[:, 1]
        freq_preds = torch.argmax(freq_outputs, dim=1)
        freq_probs = torch.softmax(freq_outputs, dim=1)[:, 1]

        # for each attacked image, collect scores/metrics
        for i in range(len(attacks)):
            attack = attacks[i]
            attack_name = attack if isinstance(attack, str) else attack.lower()

            # initialize empty metric dicts as necessary
            if attack_name not in img_attack_correct:
                img_attack_correct[attack_name] = 0
                img_attack_total[attack_name] = 0
                img_attack_preds[attack_name] = []
                img_attack_targets[attack_name] = []
            if attack_name not in freq_attack_correct:
                freq_attack_correct[attack_name] = 0
                freq_attack_total[attack_name] = 0
                freq_attack_preds[attack_name] = []
                freq_attack_targets[attack_name] = []
            if attack_name not in fusion_attack_correct:
                fusion_attack_correct[attack_name] = 0
                fusion_attack_total[attack_name] = 0
                fusion_attack_preds[attack_name] = []
                fusion_attack_targets[attack_name] = []

            # update metrics

            # spatial
            img_attack_correct[attack_name] += (img_preds[i] == labels[i]).item()
            img_attack_total[attack_name] += 1
            img_attack_preds[attack_name].append(img_probs[i].unsqueeze(0))
            img_attack_targets[attack_name].append(labels[i].unsqueeze(0))

            # spectral
            freq_attack_correct[attack_name] += (freq_preds[i] == labels[i]).item()
            freq_attack_total[attack_name] += 1
            freq_attack_preds[attack_name].append(freq_probs[i].unsqueeze(0))
            freq_attack_targets[attack_name].append(labels[i].unsqueeze(0))

            # fusion
            correct = (img_preds[i] == labels[i]) or (freq_preds[i] == labels[i])
            fusion_attack_correct[attack_name] += correct
            fusion_attack_total[attack_name] += 1
            fusion_prob = torch.max(img_probs[i], freq_probs[i])
            fusion_attack_preds[attack_name].append(fusion_prob.unsqueeze(0))
            fusion_attack_targets[attack_name].append(labels[i].unsqueeze(0))

    # print attack-wise metrics
    print("-" * 64)

    for attack in ATTACKS:
        if attack not in img_attack_total or img_attack_total[attack] == 0:
            continue

        # spatial
        img_preds_tensor = torch.cat(img_attack_preds[attack])
        img_targets_tensor = torch.cat(img_attack_targets[attack])
        img_acc = 100. * img_attack_correct[attack] / img_attack_total[attack]
        img_f1 = f1(img_preds_tensor, img_targets_tensor).item()
        img_auc = auc(img_preds_tensor, img_targets_tensor).item()

        # spectral
        freq_preds_tensor = torch.cat(freq_attack_preds[attack])
        freq_targets_tensor = torch.cat(freq_attack_targets[attack])
        freq_acc = 100. * freq_attack_correct[attack] / freq_attack_total[attack]
        freq_f1 = f1(freq_preds_tensor, freq_targets_tensor).item()
        freq_auc = auc(freq_preds_tensor, freq_targets_tensor).item()

        # fusion
        fusion_preds_tensor = torch.cat(fusion_attack_preds[attack])
        fusion_targets_tensor = torch.cat(fusion_attack_targets[attack])
        fusion_acc = 100. * fusion_attack_correct[attack] / fusion_attack_total[attack]
        fusion_f1 = f1(fusion_preds_tensor, fusion_targets_tensor).item()
        fusion_auc = auc(fusion_preds_tensor, fusion_targets_tensor).item()

        print(f"Attack: {attack.upper()}")
        print(f"  [Spatial Setting] Acc: {img_acc:.2f}% | F1: {img_f1:.4f} | AUC: {img_auc:.4f}")
        print(f"  [Spectral Setting] Acc: {freq_acc:.2f}% | F1: {freq_f1:.4f} | AUC: {freq_auc:.4f}")
        print(f"  [Fusion Setting] Acc: {fusion_acc:.2f}% | F1: {fusion_f1:.4f} | AUC: {fusion_auc:.4f}")
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

    print("-" * 64)
    print(f"Clean Images")
    print("-" * 64)
    for checkpoint in models:
        model = load_model(checkpoint)
        model.to(device)
        if 'img' in checkpoint:
            test(model, test_data_img)
        elif 'freq' in checkpoint:
            test(model, test_data_spec)

    # only test attacking models if command-line argument is set
    if args.attack:
        print(f"Attacked Images")
        print("-" * 64)

        attack_models = set()
        for checkpoint in models:
            if checkpoint.endswith('.pth'):
                parts = checkpoint.split('/')[1]
                parts = parts.split('_')
                if len(parts) >= 3:
                    attack_models.add(parts[0])

        for model in attack_models:
            spatial_path = os.path.join(args.models_dir, f'{model}_img_checkpoint.pth')
            spectral_path = os.path.join(args.models_dir, f'{model}_freq_checkpoint.pth')
            if os.path.exists(spatial_path) and os.path.exists(spectral_path):
                spatial_model = load_model(spatial_path)
                spectral_model = load_model(spectral_path)
                spatial_model.to(device)
                spectral_model.to(device)
                test_attacked(spatial_model, spectral_model, attack_data)
            else:
                print(f"Missing both spatial and spectral model for [{model.upper()}]")


