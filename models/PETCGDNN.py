import time
import torch
import torch.nn as nn
import numpy as np

class PET(nn.Module):
    def __init__(self, frame_length=128):
        super(PET, self).__init__()
        self.p1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(frame_length*2, 1),
        )

    def forward(self, x):
        p1_x = self.p1(x)
        sin_x = torch.sin(p1_x)
        cos_x = torch.cos(p1_x)

        x11 = x[:, :, 0] * cos_x
        x12 = x[:, :, 1] * sin_x
        x21 = x[:, :, 0] * sin_x
        x22 = x[:, :, 1] * cos_x

        y1 = x11 + x12
        y2 = x21 - x22
        y1 = torch.unsqueeze(y1, 2)
        y2 = torch.unsqueeze(y2, 2)

        x2 = torch.cat([y1, y2], dim=2)
        x2 = torch.transpose(x2, 1, 2)
        x2 = torch.unsqueeze(x2, 1)

        return x2


class PETCGDNN_FE(nn.Module):
    def __init__(self, frame_length=1024, hidden_size=128):
        super(PETCGDNN_FE, self).__init__()
        self.features = nn.Sequential(
            PET(frame_length=frame_length),
            nn.Conv2d(1, 75, kernel_size=(2, 8), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Conv2d(75, 25, kernel_size=(1, 5), padding='valid'),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(input_size=25, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        x = torch.transpose(x,2,1)
        x = self.features(x)
        x = torch.squeeze(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.gru(x)
        x = x[:, -1, :]

        return x
    
class PETCGDNN(nn.Module):
    def __init__(self, frame_length=128, classes=26):
        super(PETCGDNN, self).__init__()
        self.encoder = PETCGDNN_FE(frame_length)
        self.classifier = nn.Linear(128, classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = PETCGDNN(classes=26, frame_length=128).to(device)
    x = torch.rand((400, 2, 128)).to(device)
    y = model(x)
    print(y.shape)