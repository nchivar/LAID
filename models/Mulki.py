import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2Classifier, self).__init__()

        # load pretrained MobileNetV2 backbone
        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # extract only the feature layers (exclude classifier)
        self.features = mobilenet.features  # output: (B, 1280, 8, 8) for 256×256 input

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(1280 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
