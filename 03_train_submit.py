#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# jupyter nbconvert --to python 03_train_submit.ipynb
# 在命令行下将本文件转为python文件 挂在tmux下运行 网络不稳定 在notebook内训练会丢失结果


# In[1]:


import os
from os import path
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
# from albumentations import Compose, ShiftScaleRotate, Resize
import albumentations as alb
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
from matplotlib import pyplot as plt

from apex import amp

from sklearn.metrics import log_loss

import pretrainedmodels
from efficientnet_pytorch import EfficientNet

from fastai2.torch_basics      import *
from fastai2.data.all          import *
from fastai2.test              import *
from fastai2.medical.imaging   import *


# In[ ]:


dir_csv = dir_dcm = '/home/jupyter/rsna/source_data'
# dir_train_img = '/home/jupyter/rsna/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x'
# dir_test_img = '/home/jupyter/rsna/rsna-train-stage-1-images-png-224x/stage_1_test_png_224x'

# dir_train_img = '/home/jupyter/rsna/rsna-train-stage-1-images_fastai_subdural_windows/stage_1_train_npy'
# dir_test_img  = '/home/jupyter/rsna/rsna-train-stage-1-images_fastai_subdural_windows/stage_1_test_npy'


# In[2]:


saved = 'saved'
fold = 0
fold = 1

# train = pd.read_csv(f'{saved}/train_fold{fold}.csv')


# train = pd.read_csv(f'{saved}/train.csv')
# test = pd.read_csv(f'{saved}/test.csv')


# In[ ]:


class IntracranialDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None, subDset=1):
        
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels
        self.subDset = subDset

    def __len__(self):
        return len(self.data) // self.subDset

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.png')
        img = cv2.imread(img_name)   
        if img is None:
            print(img_name)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']   
            
        if self.labels:
            
            labels = torch.tensor(
                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}    
        
        else:      
            return {'image': img}
            

            
bins = torch.load('sample_bin.pt')
mean = 0.4104
std = 0.2989
class DicomDataset(Dataset):

    def __init__(self, csv_file, labels, transform=None, subDset=1):
        
#         self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels
        self.subDset = subDset

    def __len__(self):
        return len(self.data) // self.subDset

    def __getitem__(self, idx):
        
#         img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.png')
#         img = cv2.imread(img_name)   
        
        dcm = Path(self.data.fname[idx]).dcmread()
        img = dcm.hist_scaled_px(bins)
#         img = img.unsqueeze(-1).numpy()
#         print(img.shape)
        while img.shape[0] != 512:
            idx = np.random.randint(0, len(self.data))
            dcm = Path(self.data.fname[idx]).dcmread()
            img = dcm.hist_scaled_px(bins)
        if img is None:
            print(img_name)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']   
        
#         img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img -= mean
        img /= std
            
        if self.labels:
            
            labels = torch.tensor(
                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}    
        
        else:      
            return {'image': img}
            


# In[4]:


n_classes = 6
n_epochs = 100

batch_size = 6*7 # se_resnext50_32x4d 224*224

batch_size = 6*7*3 * 2 # se_resnext50_32x4d 128*128

# batch_size = 6*7*2*7 # 16GB se_resnext50_32x4d 128*128  fp16
# resize_size = (128, 128)

batch_size = 6*7*2*4 # 16GB se_resnext50_32x4d 164*164  fp16
resize_size = (164, 164)

batch_size = 30 # 16GB se_resnext50_32x4d 164*164  fp16
resize_size = (224, 224)

# batch_size = 550 # 16GB resnet34 224*224  fp16
# resize_size = (224, 224)

val_batch_size = batch_size * 3


# 先 128*128 训练，早期收敛
# 再用 224*224 训练

# In[34]:


# Data loaders

transform_train = alb.Compose([
    alb.HorizontalFlip(),
    alb.VerticalFlip(),
    alb.RandomRotate90(),
    alb.GridDistortion(),
    
    alb.ShiftScaleRotate(),
#     alb.Resize(*resize_size),
#     alb.Normalize()
#     ToTensor()
])

transform_test= alb.Compose([
#     alb.Resize(512, 512),
#     alb.Resize(*resize_size),
    ToTensor()
])


# In[ ]:





def train_aug(image):
    if np.random.rand() < 0.5:
        image = torch.flip(image, [0])
    if np.random.rand() < 0.5:
        image = torch.flip(image, [1])
        
    rot90 = np.random.randint(0, 4)
    image = torch.rot90(image, rot90)
    
#     image = torch.nn.functional.interpolate(image, *resize_size, mode='linear', align_corners=True)
    
    return {
        'image': image
    }

def test_aug(image):
#     image = torch.nn.functional.interpolate(image, *resize_size, mode='linear', align_corners=True)
    return {
        'image': image
    }

# train_dataset = IntracranialDataset(
#     csv_file=f'{saved}/train.csv', path=dir_train_img, transform=transform_train, labels=True)

# val_dataset = IntracranialDataset(
#     csv_file=f'{saved}/val.csv', path=dir_train_img, transform=transform_test, labels=True)

# test_dataset = IntracranialDataset(
#     csv_file=f'{saved}/test.csv', path=dir_test_img, transform=transform_test, labels=False)



# 过拟合问题, 每次用1/5控制
train_dataset = DicomDataset(
    csv_file=f'{saved}/train_fold{fold}.csv', transform=train_aug, labels=True, subDset=5)

val_dataset = DicomDataset(
    csv_file=f'{saved}/val_fold{fold}.csv', transform=test_aug, labels=True)

test_dataset = DicomDataset(
    csv_file=f'{saved}/test.csv', transform=test_aug, labels=False)

# train_dataset = DicomDataset(
#     csv_file=f'{saved}/train_fold{fold}.csv', transform=transform_train, labels=True, subDset=5)

# val_dataset = DicomDataset(
#     csv_file=f'{saved}/val_fold{fold}.csv', transform=transform_test, labels=True)

# test_dataset = DicomDataset(
#     csv_file=f'{saved}/test.csv', transform=transform_test, labels=False)


# pin_memory 加速
data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8, pin_memory=True)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8, pin_memory=True)


# In[ ]:





# In[ ]:



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


# model = torchvision.models.resnet34(pretrained=True)
# model.fc = torch.nn.Linear(512, n_classes)
# model_name = 'resnet34'


#(last_linear): Linear(in_features=2048, out_features=1000, bias=True)
model = pretrainedmodels.se_resnext50_32x4d()
model.layer0.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
model.last_linear = torch.nn.Linear(2048, n_classes)
model_name = 'se_resnext50_32x4d'



# model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=n_classes)
# model_name = 'efficientnet-b0'

model.to(device)

# criterion = torch.nn.BCEWithLogitsLoss()
plist = [{'params': model.parameters(), 'lr': 1e-3}]
optimizer = optim.Adam(plist, lr=1e-3)

# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


# In[ ]:


bst_model = None
log = Logger()
log.open(path.join(saved, 'log.txt'))


# In[ ]:


check_point = None


# In[ ]:


ckpt_path = f'{saved}/{model_name}_fold{fold}_checkpoint.pth'
amp_ckpt_path = f'{saved}/{model_name}_fold{fold}_amp_checkpoint.pt'

# amp_ckpt_path = f'{saved}/{model_name}_amp_checkpoint_{11}.pt'

opt_level = 'O1'

# 'val_loss': loss

if path.exists(amp_ckpt_path):
    print(f'===> load {amp_ckpt_path}')
    checkpoint = torch.load(amp_ckpt_path)
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    epoch_start = checkpoint['epoch'] + 1

elif path.exists(ckpt_path):
    print(f'===> load {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    optimizer.load_state_dict(ckpt['optim']),
    epoch_start = ckpt['epoch'] + 1
    model.load_state_dict(ckpt['state'])
    
    # Initialization
    
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    
    log.write(f'resume from epoch {epoch_start}\n')
else:
    epoch_start = 0
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    
print(f'training from epoch {epoch_start}')
    
# amp_ckpt_path = f'{saved}/{model_name}_fold{fold}_amp_checkpoint.pt'


# In[ ]:


def do_eval():
    print(f'==> eval')
    model.eval()
#     val_pred = np.zeros((len(val_dataset) * n_classes, 1))
#     val_true = np.zeros((len(val_dataset) * n_classes, 1))
    val_len = len(data_loader_val)
    log.write(f'Epoch {epoch} fold {fold}, val\n')
    
    val_loss = 0.
    
    tbar = tqdm(data_loader_val, ascii=True)
    for val_step, val_batch in enumerate(tbar):
#         log.write(f'\r{val_step:05d} / {val_len}')
        with torch.no_grad():
            inputs = val_batch["image"]
            labels = val_batch["labels"]

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
    val_loss /= val_len
        
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'amp': amp.state_dict(),
        'val_loss': loss
    }
    torch.save(checkpoint, f'{saved}/{model_name}_fold{fold}_amp_checkpoint_{epoch}.pt')
    log.write(f'epoch {epoch} - val loss: {loss}\n')


# In[ ]:


torch.backends.cudnn.benchmark = True

weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0]).cuda()
def criterion(y_pred,y_true):
    return F.binary_cross_entropy_with_logits(y_pred,
                                  y_true,
                                  weights.repeat(y_pred.shape[0],1))

for epoch in range(epoch_start, n_epochs):
    
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)
    
#     do_eval()
    
    model.train()    
    tr_loss = 0
    

#     tk0 = data_loader_train
    tbar = tqdm(data_loader_train, ascii=True)
    trn_len = len(data_loader_train)
    for step, batch in enumerate(tbar):
#         log.write(f'\r{step:05d} / {trn_len}')
        optimizer.zero_grad()
        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
#         loss.backward()
        optimizer.step()
        
#         print(f' loss: {loss.item():.4f}', end='')
        
        tr_loss += loss.item()

        tbar.set_description(f'loss: {loss.item():.4f}')
        
        if (step+1) % 100 == 0:
            # 训练一个epoc太久                   
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'amp': amp.state_dict()
            }
            torch.save(checkpoint, amp_ckpt_path)
    
    epoch_loss = tr_loss / len(data_loader_train)
    log.write(f'\nTraining Loss: {epoch_loss:.4f} fold {fold} \n')

    
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'amp': amp.state_dict()
    }
    torch.save(checkpoint, amp_ckpt_path)
        
#     if epoch < 5:
#         # do val from epoch 20
#         continue
        
    if (epoch + 1) % 2 == 0:
        do_eval()
 
        
    
        

    

