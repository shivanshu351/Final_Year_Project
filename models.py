import torch
import torch.nn as nn
from torchvision import models

# ANN Model
class ANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# YOLOv7-style Model
class YOLOv7Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),

            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        return x


class YOLOv7Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = YOLOv7Backbone()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            pooled = self.avgpool(features)
            in_features = pooled.view(1, -1).size(1)

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.avgpool(features)
        flattened = self.flatten(pooled)
        return self.classifier(flattened)


# ViT with Custom Head
class ViTClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(ViTClassifier, self).__init__()
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(layer in name for layer in [
                'encoder.layers.encoder_layer_11',
                'encoder.layers.encoder_layer_10',
                'encoder.layers.encoder_layer_9',
                'encoder.layers.encoder_layer_8',
                'heads'
            ]):
                param.requires_grad = True

        class CustomHead(nn.Module):
            def __init__(self, in_features, num_classes, dropout_rate):
                super().__init__()
                self.fc1 = nn.Linear(in_features, 512)
                self.ln1 = nn.LayerNorm(512)
                self.dropout1 = nn.Dropout(dropout_rate)
                self.fc2 = nn.Linear(512, 256)
                self.ln2 = nn.LayerNorm(256)
                self.dropout2 = nn.Dropout(dropout_rate / 2)
                self.fc3 = nn.Linear(256, num_classes)
                self.gelu = nn.GELU()

            def forward(self, x):
                identity = x
                x = self.fc1(x)
                x = self.ln1(x)
                x = self.gelu(x)
                x = self.dropout1(x)
                x = self.fc2(x)
                x = self.ln2(x)
                x = self.gelu(x)
                x = self.dropout2(x)
                x = self.fc3(x + identity[:, :256])
                return x

        self.model.heads = CustomHead(768, num_classes, dropout_rate)

    def forward(self, x):
        return self.model(x)
