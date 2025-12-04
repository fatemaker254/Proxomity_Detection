import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallHandCNN(nn.Module):
    def __init__(self):
        super(SmallHandCNN, self).__init__()
        # Input: 3 x 64 x 64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)   # 64->32->16->8

        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)  # binary output (hand or no-hand)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # raw logit

if __name__ == "__main__":
    model = SmallHandCNN()
    dummy = torch.randn(1, 3, 64, 64)
    out = model(dummy)
    print("Output shape:", out.shape)
