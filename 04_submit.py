#!/usr/bin/env python
# coding: utf-8

# In[2]:


# jupyter nbconvert --to python 04_submit.ipynb
# 在命令行下将本文件转为python文件 挂在tmux下运行 网络不稳定 在notebook内训练会丢失结果


# In[3]:


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


# In[4]:


use_cpu = False


# In[5]:


dir_csv = dir_dcm = '/home/jupyter/rsna/source_data'
dir_train_img = '/home/jupyter/rsna/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x'
dir_test_img = '/home/jupyter/rsna/rsna-train-stage-1-images-png-224x/stage_1_test_png_224x'


# In[6]:


saved = 'saved'
train = pd.read_csv(f'{saved}/train.csv')
test = pd.read_csv(f'{saved}/test.csv')


# In[7]:


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
            


# In[8]:


n_classes = 6
n_epochs = 100
batch_size = 5*7 * 3


# In[9]:


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

test_dataset = IntracranialDataset(
    csv_file=f'{saved}/test.csv', path=dir_test_img, transform=transform_test, labels=False)

data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# In[10]:



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


# In[11]:


if use_cpu:
    device = torch.device("cpu")
else:
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


# In[12]:


bst_model = None
log = Logger()
log.open(path.join(saved, 'log.txt'))


# In[13]:


check_point = None


# In[14]:


ckpt_path = f'{saved}/checkpoint.pth'

if use_cpu:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
else:
    ckpt = torch.load(ckpt_path, map_location='cuda:0')

optimizer.load_state_dict(ckpt['optim']),
epoch_start = ckpt['epoch']
model.load_state_dict(ckpt['state'])
    


# In[ ]:


def sharpen(p,t=0):
    if t!=0:
        return p**t
    else:
        return p

def cls_do_eval(nets, batch_data, augment=[], t=0):
    num_augment = 0
    probability_label = None
    for net in nets:
        if 1: #  null
            logit =  net(batch_data)  #net(input)
            probability = torch.sigmoid(logit)
            if probability_label is None:
                probability_label = sharpen(probability,0)
            else:
                probability_label += sharpen(probability,0)
            num_augment+=1

        if 'flip_lr' in augment:
            logit = net(torch.flip(batch_data,dims=[3]))
            probability  = torch.sigmoid(logit)

            probability_label += sharpen(probability, t)
            num_augment+=1
            
        if 'flip_ud' in augment:
            logit = net(torch.flip(batch_data,dims=[2]))
            probability = torch.sigmoid(logit)

            probability_label += sharpen(probability, t)
            num_augment+=1
        
    probability_label = probability_label/num_augment
    return probability_label


# In[18]:


augment = ['null', 'flip_lr','flip_ud']


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
        
#         pred = model(x_batch)
        pred = cls_do_eval([model], x_batch, augment)
        test_pred[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = pred.detach().cpu().reshape((len(x_batch) * n_classes, 1))
        
#         test_pred[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = torch.sigmoid(
#             pred).detach().cpu().reshape((len(x_batch) * n_classes, 1))


# In[ ]:


# Submission

submission =  pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))
submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
submission.columns = ['ID', 'Label']

submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:


#!kaggle competitions submit -f submission.csv -m from_gcp rsna-intracranial-hemorrhage-detection

