import torch
import torch.nn as nn
import numpy as np

class GRUModel_FE(nn.Module):
    def __init__(self, classes = 11):
        super(GRUModel_FE, self).__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=128, num_layers=2, batch_first=True)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]

        return x

class GRUModel(nn.Module):
    def __init__(self, classes=11):
        super(GRUModel, self).__init__()
        self.encoder = GRUModel_FE()
        self.fc = nn.Linear(128, classes)
        
    def forward(self, input):
        x = self.encoder(input)
        x = self.fc(x)

        return x
    
if __name__ == "__main__":
    model = GRUModel(classes=11)
    x = torch.randn(1, 2, 128)
    out = model(x)
    print(out.shape)