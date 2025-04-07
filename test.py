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
from dataset.dataset_loader import TinyGenImage

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

# -------------------- LOAD MODEL -------------------- #
def load_model(checkpoint_path):
    if 'shufflenet' in checkpoint_path:
        model = shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif 'mobilenetv3' in checkpoint_path:
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    elif 'mnasnet' in checkpoint_path:
        model = mnasnet0_5(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif'squeezenet' in checkpoint_path:
        model = squeezenet1_1(weights=None)
        model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1))
    elif 'mobilenetv2' in checkpoint_path:
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif 'regnet' in checkpoint_path:
        model = regnet_y_400mf(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif 'efficientnet' in checkpoint_path:
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif 'ladevic' in checkpoint_path:
        model = CNNModel()
    elif 'mulki' in checkpoint_path:
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
    test_dataset_img = TinyGenImage(data_dir=args.test_data_img)
    test_loader_img = torch.utils.data.DataLoader(test_dataset_img, batch_size=256, shuffle=False)
    test_dataset_spec = TinyGenImage(data_dir=args.test_data_spec)
    test_loader_spec = torch.utils.data.DataLoader(test_dataset_spec, batch_size=256, shuffle=False)
    return test_loader_img, test_loader_spec

def parse_flops(flops_str):
    units = {'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12}
    unit = flops_str[-1]
    if unit in units:
        return float(flops_str[:-1]) / 1e9 * units[unit]
    return float(flops_str) / 1e9

# -------------------- TEST SINGLE DOMAIN -------------------- #
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

@torch.no_grad()
def test_fusion(img_model, spec_model, img_data, spec_data, fusion_type):

    f1 = BinaryF1Score().to(device)
    auc = BinaryAUROC().to(device)

    total, correct = 0, 0

    for (imgs, img_labels), (spec_imgs, spec_labels) in zip(img_data, spec_data):
        assert torch.equal(img_labels, spec_labels), "Label mismatch between spatial and frequency sets"

        imgs, spec_imgs, labels = imgs.to(device), spec_imgs.to(device), img_labels.to(device)

        with autocast(device_type=device):
            img_out = img_model(imgs)
            spec_out = spec_model(spec_imgs)

            # Combine probabilities
            img_prob = torch.softmax(img_out, dim=1)
            spec_prob = torch.softmax(spec_out, dim=1)
            combined = (img_prob + spec_prob) / 2

        preds = torch.argmax(combined, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total

    print("-" * 64)
    print(f"{fusion_type} Accuracy: {acc:.2f}%")
    print("-" * 64)

# -------------------- MAIN -------------------- #
if __name__ == "__main__":
    print("Running Testing")

    test_data_img, test_data_spec= load_data()

    models = []
    for filename in os.listdir(args.models_dir):
        if filename.endswith('.pth'):
           models.append(os.path.join(args.models_dir, filename))

    print(f"Models to be tested: {models}")


    # # single domain
    # print("-" * 64)
    # print(f"SINGLE DOMAIN")
    # print("-" * 64)
    # for checkpoint in models:
    #     model = load_model(checkpoint)
    #     model.to(device)
    #     if 'img' in checkpoint:
    #         test(model, test_data_img)
    #     elif 'spec' in checkpoint:
    #         test(model, test_data_spec)

    # fusion
    print("-" * 64)
    print(f"FUSION")
    print("-" * 64)
    best_img_model = load_model('outputs/img_regnet.pth').to(device)
    best_spec_model = load_model('outputs/spec_mulki.pth').to(device)

    for checkpoint in models:

        # MSDI/CMBF
        if 'img' in checkpoint:
            img_model = load_model(checkpoint).to(device)
            spec_model = load_model(checkpoint.replace("img", "spec")).to(device)
            test_fusion(img_model, spec_model, test_data_img, test_data_spec, 'MSDI')
            test_fusion(img_model, best_spec_model, test_data_img, test_data_spec, 'CMBF')

        # CMBS
        elif 'spec' in checkpoint:
            spec_model = load_model(checkpoint).to(device)
            test_fusion(best_img_model, spec_model, test_data_img, test_data_spec, 'CMBS')

