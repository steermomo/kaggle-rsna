#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# jupyter nbconvert --to python 03_train_submit.ipynb
# 在命令行下将本文件转为python文件 挂在tmux下运行 网络不稳定 在notebook内训练会丢失结果


# In[2]:


import os
from os import path
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import sys
from albumentations import Compose, ShiftScaleRotate, Resize
import albumentations as alb
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt

from sklearn.metrics import log_loss


# In[ ]:


dir_csv = dir_dcm = '/home/jupyter/rsna/source_data'
dir_train_img = '/home/jupyter/rsna/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x'
dir_test_img = '/home/jupyter/rsna/rsna-train-stage-1-images-png-224x/stage_1_test_png_224x'


# In[3]:


saved = 'saved'
train = pd.read_csv(f'{saved}/train.csv')
test = pd.read_csv(f'{saved}/test.csv')


# In[4]:


class IntracranialDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):
        
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.png')
        img = cv2.imread(img_name)   
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']   
            
        if self.labels:
            
            labels = torch.tensor(
                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}    
        
        else:      
            return {'image': img}
            


# In[5]:


n_classes = 6
n_epochs = 100
batch_size = 5*7


# In[7]:


# Data loaders

transform_train = Compose([
    alb.HorizontalFlip(),
    alb.VerticalFlip(),
    alb.RandomRotate90(),
    alb.GridDistortion(),
    ShiftScaleRotate(),
    alb.Resize(512, 512),
    ToTensor()
])

transform_test= Compose([
    alb.Resize(512, 512),
    ToTensor()
])

train_dataset = IntracranialDataset(
    csv_file=f'{saved}/train.csv', path=dir_train_img, transform=transform_train, labels=True)

val_dataset = IntracranialDataset(
    csv_file=f'{saved}/val.csv', path=dir_train_img, transform=transform_test, labels=True)

test_dataset = IntracranialDataset(
    csv_file=f'{saved}/test.csv', path=dir_test_img, transform=transform_test, labels=False)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# In[8]:



# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='a+'
        self.file = open(file, mode)
        self.file.write('\n----\n')

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


# In[ ]:


device = torch.device("cuda:0")

# model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
# model.fc = torch.nn.Linear(2048, n_classes)
model = torchvision.models.resnet34(pretrained=True)
model.fc = torch.nn.Linear(512, n_classes)

model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
plist = [{'params': model.parameters(), 'lr': 2e-5}]
optimizer = optim.Adam(plist, lr=2e-5)

# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


# In[1]:


bst_model = None
log = Logger()
log.open(path.join(saved, 'log.txt'))


# In[ ]:


check_point = None


# In[ ]:


ckpt_path = f'{saved}/checkpoint.pth'
if path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    optimizer.load_state_dict(ckpt['optim']),
    epoch_start = ckpt['epoch']
    model.load_state_dict(ckpt['state'])
    
    log.write(f'resume from epoch {epoch_start}\n')
else:
    epoch_start = 0


for epoch in range(epoch_start, n_epochs):
    
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    model.train()    
    tr_loss = 0
    

    tk0 = data_loader_train
    trn_len = len(data_loader_train)
    for step, batch in enumerate(tk0):
        log.write(f'\r{step:05d} / {trn_len}')
        optimizer.zero_grad()
        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        print(f' loss: {loss.item():.4f}', end='')
        
        tr_loss += loss.item()

        optimizer.step()
        
        if step % 100 == 0:
            # 训练一个epoc太久
            save_state = {
                'optim': optimizer.state_dict(),
                'epoch': epoch+1,
                'state': model.state_dict(),
            }
            with open(ckpt_path, 'wb') as save_file:
                torch.save(save_state, save_file)
    
    epoch_loss = tr_loss / len(data_loader_train)
    log.write('Training Loss: {:.4f}'.format(epoch_loss))

    
        
    if epoch % 1 == 0:
        save_state = {
            'optim': optimizer.state_dict(),
            'epoch': epoch+1,
            'state': model.state_dict(),
        }
        with open(ckpt_path, 'wb') as save_file:
            torch.save(save_state, save_file)
        
    if epoch < 20:
        # do val from epoch 20
        continue
        
    if epoch % 5 != 0:
        continue
        
    # do val
    val_pred = np.zeros((len(val_dataset) * n_classes, 1))
    val_true = np.zeros((len(val_dataset) * n_classes, 1))
    val_len = len(data_loader_val)
    log.write(f'Epoch {epoch}, val\n')
    for val_step, val_batch in enumerate(data_loader_val):
        log.write(f'\r{val_step:05d} / {val_len}')
        with torch.no_grad():
            inputs = val_batch["image"]
            labels = val_batch["labels"]

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
            val_pred[(val_step * batch_size * n_classes):((val_step + 1) * batch_size * n_classes)] = torch.sigmoid(
            outputs).detach().cpu().reshape((len(inputs) * n_classes, 1))
            
            val_true[(val_step * batch_size * n_classes):((val_step + 1) * batch_size * n_classes)] = labels.cpu().reshape((len(inputs) * n_classes, 1))
            
    
#     val_pred = val_pred.reshape(len(val_dataset), n_classes)
#     val_true = val_true.reshape(len(val_dataset), n_classes)
    
#     loss = log_loss(val_true, val_pred, sample_weight=([1, 1, 1, 1, 1, 2] * len(val_dataset)))
    loss = log_loss(val_true, val_pred, sample_weight=([2, 1, 1, 1, 1, 1,] * len(val_dataset)))
    
    save_state = {
            'optim': optimizer.state_dict(),
            'epoch': epoch+1,
            'state': model.state_dict(),
            'val_loss': loss,
        }
    with open(f'{saved}/checkpoint_{epoch}.pth', 'wb') as save_file:
        torch.save(save_state, save_file)
    log.write(f'loss: {loss}\n')
        
    
        

    


# In[ ]:


# Inference

print('\nInference')
for param in model.parameters():
    param.requires_grad = False

model.eval()

test_pred = np.zeros((len(test_dataset) * n_classes, 1))

for i, x_batch in enumerate(data_loader_test):
    print(f'\r {i} / {len(data_loader_test)}', end='')
    x_batch = x_batch["image"]
    x_batch = x_batch.to(device, dtype=torch.float)
    
    with torch.no_grad():
        
        pred = model(x_batch)
        
        test_pred[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = torch.sigmoid(
            pred).detach().cpu().reshape((len(x_batch) * n_classes, 1))


# In[ ]:


# Submission

submission =  pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))
submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
submission.columns = ['ID', 'Label']

submission.to_csv('submission.csv', index=False)
submission.head()


# In[3]:


get_ipython().system('kaggle competitions submit -f submission.csv -m from_gcp rsna-intracranial-hemorrhage-detection')

