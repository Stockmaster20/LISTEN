import numpy as np
import time
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from model import CLDNN, TransNet_AP, LSTMModel, GRUModel, MobileNet, ICAMC, SCF_CNN_16, SCF_CNN
from models.MCLDNN import MCLDNN
from models.PETCGDNN import PETCGDNN
from models.TLDNN import TLDNN
from models.FEAT import FEAT
from models.MCFormer import ViTClassifier
from models.IQformer import IQFormer
from models.AWN import AWN
from models.MCT import ModelFor1024
from models.LISTEN import LISTEN, SPCPLoss
from dataset import Getdata_RML2016A_snr
from sklearn.metrics import confusion_matrix
import os
from scheduler import PolynomialLR
import random
import tqdm
import matplotlib.pyplot as plt
device = torch.device("cuda:0")

batchsize = 512
start_epoch = 0
training_epoch = 200

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(2026)

start = time.time()

dataset = './SMC/2016A/'

X_train = np.load(dataset + 'X_train.npy') 
X_test = np.load(dataset + 'X_test.npy')

Y_train = np.load(dataset + 'Y_train.npy')
Y_test = np.load(dataset + 'Y_test.npy')

snr_train = np.load(dataset + 'snr_train.npy')
snr_test = np.load(dataset + 'snr_test.npy')

X_val = np.load(dataset + 'X_val.npy')
Y_val = np.load(dataset + 'Y_val.npy')
snr_val = np.load(dataset + 'snr_val.npy')

all_mods = {'8PSK': 0, 'AM-DSB': 1, 'AM-SSB': 2, 'BPSK': 3, 'CPFSK': 4, 'GFSK':5, 'PAM4':6, 'QAM16':7, 'QAM64':8, 'QPSK':9, 'WBFM':10}

map_func = np.vectorize(lambda x: all_mods.get(x, x))

Y_train = map_func(Y_train.reshape(-1))
Y_test = map_func(Y_test.reshape(-1))
Y_val = map_func(Y_val.reshape(-1))

num_class = len(set(Y_train))

num_workers = min(os.cpu_count(), 4)
train_dataset = Getdata_RML2016A_snr(X_train, Y_train, snr_train)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True)

test_dataset = Getdata_RML2016A_snr(X_val, Y_val, snr_val)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True)

end = time.time()
print("load dataset time: {:.3f} s".format(end - start))


model = LISTEN(in_channels=2, num_stages=3, num_classes=num_class, feature_dim=32).to(device=device)

for name, param in model.named_parameters():
    if 'rsna_embed' in name:
        continue
    if 'weight' in name and param.dim() >= 2:
        nn.init.kaiming_normal_(param.data)
        
optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3, weight_decay=1e-5)
scheduler = PolynomialLR(optimizer, max_iter=training_epoch, power=0.9)
scaler = torch.amp.GradScaler('cuda')
NUM_ACCUMULATION_STEPS = 8
CrossLoss = nn.CrossEntropyLoss()
criterion = SPCPLoss(gamma=0.3, lambda_s=0.005, beta_l=0.05).to(device)

correct = torch.zeros(1).squeeze().to(device=device)
correct_ = list(0. for i in range(num_class))
epochs = []
train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_acc = 0

save_path = './SMC/backbone/LISTEN/RML2016A/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
start = time.time()

for epoch in range(start_epoch+1, training_epoch, 1):
    model.train()
    with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
        start = time.time()
        for idx, (data, target, _) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            data, target = data.to(device=device).float().squeeze(1), target.to(device=device).long()
            with torch.amp.autocast('cuda', enabled=False):

                output, S_k, L_k, X_feat = model(data)
                loss, loss_cls, loss_recon = criterion(output, target, S_k, L_k, X_feat)
                losstr = loss

            scaler.scale(losstr).backward()
            losstr = losstr / NUM_ACCUMULATION_STEPS
            
            if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (idx + 1 == len(train_dataloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            predict_ = output.argmax(dim=1, keepdim=True)
            correct = predict_.eq(target.view_as(predict_)).sum().item() 

            accuracy = correct/len(data)
            tepoch.set_postfix(loss=losstr.item(), accuracy='{:.3f}'.format(accuracy))
            end = time.time()
        print("train time: {:.3f} s".format(end - start))
    if (epoch + 1) % 5 == 0:
        scheduler.step()
    epochs.append(epoch)
    train_losses.append(losstr.item())
    train_accs.append(accuracy)
    
    model.eval()
    all_predicts = torch.empty(0, 1).to(device=device)
    all_targets = torch.empty(0).to(device=device)
    with torch.no_grad():
        for _ , (data, targets, snr_label) in enumerate(test_dataloader):
            
            data, targets, snr_label = data.to(device=device).float().squeeze(1), targets.to(device=device).long(), snr_label.to(device=device).long()

            outputs, _, _, _ = model(data)

            loss = CrossLoss(outputs, targets)
            predicts = outputs.argmax(dim=1, keepdim=True)
            all_targets = torch.cat([all_targets, targets])
            all_predicts = torch.cat([all_predicts, predicts], dim=0)
    
        correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
        accuracy_ = correct_/float(len(test_dataset))
        print("val_acc:", accuracy_)
    
    val_losses.append(loss.item())
    val_accs.append(accuracy_)
        
    if accuracy_ > best_acc:
        best_acc = accuracy_
        torch.save(model.state_dict(), save_path + f'model_epoch{epoch+1}_valAcc_{accuracy_:.3f}.pth')
    else:
        torch.save(model.state_dict(), save_path + 'model_epoch{}.pth'.format(epoch+1))

end = time.time()
print("train time: {:.3f} s".format(end - start))

np.savetxt(save_path + 'train_acc.txt',train_accs)
np.savetxt(save_path + 'train_loss.txt',train_losses)
np.savetxt(save_path + 'val_acc.txt',val_accs)
np.savetxt(save_path + 'val_loss.txt',val_losses)
                
plt.subplot(121)
plt.plot(epochs, train_losses, color = 'b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# 绘制 acc 曲线
plt.subplot(122)
plt.plot(epochs, train_accs, color = 'r')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(save_path + 'train_curve.png', dpi=600, bbox_inches='tight')
plt.close()
