from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
from collections import Counter
from scipy.interpolate import interp1d
import h5py
import pandas as pd
from scipy.signal import stft

import os

class Getdata_RML2016A(Dataset):
    def __init__(self, data, label, transform = None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.transform = transform
        print("shape of all data:", self.X.shape)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        return x, y
        
    def __len__(self):
        return(self.X.shape[0])

class RML2016B(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = samples
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx])
        _,_,stp = stft(x[0,:],1.0,'blackman',31,30,128)
        # length = 512
        # x = x.T
        # _, _, stp = stft(x[0,:], 1.0, 'blackman', 61, 60, 64)
        return torch.Tensor(x), torch.Tensor(np.expand_dims(stp[:32,:],0)), self.label[idx], self.SNR[idx]
    

class Getdata_RML2016A_snr(Dataset):
    def __init__(self, data, label, snr, transform = None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.snr = snr
        self.transform = transform
        print("shape of all data:", self.X.shape)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        # x = x.T
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        snr = self.snr[index]
        return x, y, snr
        
    def __len__(self):
        return(self.X.shape[0])