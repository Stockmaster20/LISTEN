import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class ICAMC_FE(nn.Module):
    def __init__(self, classes=11):
        super(ICAMC_FE, self).__init__()
        dr = 0.2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 8), padding='same')
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 4), padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 8), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))
        self.dropout = nn.Dropout(dr)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(1, 8), padding='same')
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.flatten(x)

        return x
    
class ICAMC(nn.Module):
    def __init__(self, seq_length=128, classes=11):
        super(ICAMC, self).__init__()
        self.dr = 0.2
        self.encoder = ICAMC_FE()
        self.dense = nn.Sequential(
            nn.Linear(64*seq_length, 128),
            nn.ReLU(),
            nn.Dropout(self.dr),
            nn.Linear(128, classes),)
        
    def forward(self, input):
        x = self.encoder(input)
        x = self.dense(x)

        return x
    
if __name__ == "__main__":
    model = ICAMC(seq_length=128, classes=11)
    x = torch.randn(1, 1, 2, 128)
    out = model(x)
    print(out.shape)