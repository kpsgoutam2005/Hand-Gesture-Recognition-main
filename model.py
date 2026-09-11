import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class GestureNet(nn.Module):
    def __init__(self):
        super(GestureNet, self).__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Input: (1, 224, 224)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32x224x224
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x112x112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x56x56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x28x28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256x28x28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256x14x14
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*14*14, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, Config.NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def initialize_model(device):
    model = GestureNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    return model, criterion, optimizer, scheduler
