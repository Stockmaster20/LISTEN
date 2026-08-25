import numpy as np
import time
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from SMC.dataset import Getdata_RML2016A_snr
from SMC.models.LISTEN import LISTEN, SPCPLoss
from SMC.models.AWN import AWN
from sklearn.metrics import confusion_matrix
import os
from SMC.scheduler import PolynomialLR
import tqdm
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from thop import profile
import random

device = torch.device("cuda:0")

batchsize = 512
start_epoch = 0
training_epoch = 100

def set_random_seed(seed, deterministic=False):
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(3407)


start = time.time()

dataset = './SMC/2016A/'

X_test = np.load(dataset + 'X_test.npy')
Y_test = np.load(dataset + 'Y_test.npy')
snr_test = np.load(dataset + 'snr_test.npy')

print(set(Y_test), snr_test)

all_mods = {'8PSK': 0, 'AM-DSB': 1, 'AM-SSB': 2, 'BPSK': 3, 'CPFSK': 4, 'GFSK':5, 'PAM4':6, 'QAM16':7, 'QAM64':8, 'QPSK':9, 'WBFM':10}
map_func = np.vectorize(lambda x: all_mods.get(x, x))

Y_test = map_func(Y_test.reshape(-1))

test_dataset = Getdata_RML2016A_snr(X_test, Y_test, snr_test)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True)

num_class = len(set(Y_test))

end = time.time()
print("load dataset time: {:.3f} s".format(end - start))

"""
=== test ===
"""
checkpoint = torch.load(r'./model_valAcc_0.627.pth')
model = LISTEN(in_channels=2, num_stages=3, num_classes=num_class, feature_dim=32).to(device=device)
model.load_state_dict(checkpoint)
print("Load Pretrained Model Successful!")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量: {total_params}")

CrossLoss = nn.CrossEntropyLoss()

correct = torch.zeros(1).squeeze().to(device=device)
correct_ = list(0. for i in range(num_class))
epochs = []
train_losses = []
train_accs = []
val_losses = []
val_accs = []
val_kap = []
best_acc = 0

print(set(snr_test))

snr_list = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

snr_type = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

model.eval()

val_accs = []
val_kap = []
dim = 128
all_predicts = torch.empty(0, 1).to(device)
all_targets = torch.empty(0).to(device)
all_fea = torch.empty(0, dim).to(device)
for snr in set(snr_test):
    print("=== SNR: {} dB ===".format(snr_list[snr]))
    X_test_snr = X_test[snr_test==snr]
    Y_test_snr = Y_test[snr_test==snr]
    snr_test_snr = snr_test[snr_test==snr]
    test_dataset = Getdata_RML2016A_snr(X_test_snr, Y_test_snr, snr_test_snr)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True)
    batch_predicts = torch.empty(0, 1).to(device)
    batch_targets = torch.empty(0).to(device)
    batch_fea = torch.empty(0, dim).to(device)
    with torch.no_grad():
        for _ , (data, targets) in enumerate(test_dataloader):
            
            data, targets = data.to(device=device).float().squeeze(1), targets.to(device=device).long()

            outputs, _, _, _ = model(data)
            predicts = outputs.argmax(dim=1, keepdim=True)
            batch_targets = torch.cat([batch_targets, targets])
            batch_predicts = torch.cat([batch_predicts, predicts], dim=0)
    
        correct_ = batch_predicts.eq(batch_targets.view_as(batch_predicts)).sum().item()
        kappa = cohen_kappa_score(batch_targets.cpu().numpy(), batch_predicts.cpu().numpy())
        accuracy_ = correct_/float(X_test_snr.shape[0])
        print("val_acc:", accuracy_)
        print(f"Cohen's Kappa: {kappa:.4f}")
    
    all_predicts = torch.cat([all_predicts, batch_predicts], dim=0)
    all_targets = torch.cat([all_targets, batch_targets], dim=0)
    
    val_kap.append(kappa)
    val_accs.append(accuracy_)

print(val_accs, np.mean(val_accs))
print(val_kap, np.mean(val_kap))

end = time.time()
print("train time: {:.3f} s".format(end - start))