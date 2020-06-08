import numpy as np
import pandas as pd
import collections
import datetime
import gc
import glob
import logging
import math
import operator
import os 
import pickle
import random
import re
import sklearn
import scipy.signal
import scipy.stats as stats
import seaborn as sns
import string
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from contextlib import contextmanager
from collections import Counter, defaultdict, OrderedDict
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_log_error, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.nn import CrossEntropyLoss, MSELoss
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import tensorflow as tf

from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt


# https://github.com/pranjaldatta/Denoising-Autoencoder-in-Pytorch/blob/master/DenoisingAutoencoder.ipynb
class DenoisingAutoEncoder(nn.Module):
  def __init__(self, in_out_channels):
      super(DenoisingAutoEncoder, self).__init__()
      self.encoder=nn.Sequential(
                    nn.Linear(in_out_channels,256),
                    nn.ReLU(True),
                    nn.Linear(256,128),
                    nn.ReLU(True),
                    nn.Linear(128,64),
                    nn.ReLU(True)
                    )
      
      self.decoder=nn.Sequential(
                    nn.Linear(64,128),
                    nn.ReLU(True),
                    nn.Linear(128,256),
                    nn.ReLU(True),
                    nn.Linear(256,in_out_channels),
                    # nn.Sigmoid(),
                    )
    
  def forward(self,x):
      x=self.encoder(x)
      x=self.decoder(x)   
      return x


class DenoisingDataset:
    def __init__(self, spec_array):
        self.spec_array = spec_array

    def __len__(self):
        return len(self.spec_array)

    def __getitem__(self, item):
        light_intensity = self.spec_array[item]

        return {
            'light_intensity': torch.tensor(light_intensity, dtype=torch.float32),
        }


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def loss_fn(logits, targets):
    loss_fct = nn.MSELoss()
    loss = loss_fct(logits, targets)
    return loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    y_true = []
    y_pred = []
    for bi, d in enumerate(tk0):

        light_intensity = d["light_intensity"].to(device, dtype=torch.float32)

        model.zero_grad()
        inputs = light_intensity.reshape(-1, 512)
        outputs = model(inputs)

        loss = loss_fn(outputs, inputs)
        loss.backward()
        optimizer.step()

        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        y_pred.append(outputs)

        losses.update(loss.item(), light_intensity.size(0))
        tk0.set_postfix(loss=losses.avg)

    y_pred_cat = np.concatenate(y_pred, 0)
    return y_pred_cat


def test_fn(data_loader, model, device):
    model.eval() 
    preds = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            light_intensity = d["light_intensity"].to(device, dtype=torch.float32)
            outputs = model(light_intensity.reshape(-1, 512))
            # outputs = torch.sigmoid(outputs).cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
            preds.append(outputs)
    
    return preds


file_path = '/content/drive/My Drive/offline_competition/atma05/'

df_train = pd.read_csv(os.path.join(file_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(file_path, 'test.csv'))

EXP_ID = 'expXX_DAE'
device = 'cuda'
EPOCHS = 100
SEED = 718
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 128

n_train = len(df_train)
df_train = df_train.append(df_test).reset_index(drop=True)

spec_meta_df = unpickle('/content/drive/My Drive/offline_competition/atma05/spectrum_metafeatures.pkl')
spec_meta_df['light_intensity'] = spec_meta_df['light_intensity'].apply(lambda x: padding(x))

df_train = pd.merge(df_train, spec_meta_df, on='spectrum_filename')

spec_array = scipy.signal.savgol_filter(np.stack(df_train['light_intensity'].values), 5, 2, deriv=0, axis=1).reshape(-1, 1, 512) # なめらかにしただけ   

# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75116#442741
spec_array /= spec_array.std(axis=2).reshape(-1, 1, 1)

del df_test
gc.collect()


train_dataset = DenoisingDataset(spec_array=spec_array)
train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=False, 
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=0, pin_memory=True)
    
del train_dataset
gc.collect()


# training part
model = DenoisingAutoEncoder(512)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, EPOCHS + 1):
    print("Starting {} epoch...".format(epoch))
    y_pred_cat = train_fn(train_loader, model, optimizer, device)
    torch.save(model.state_dict(), os.path.join('/content/drive/My Drive/offline_competition/atma05', 'DAE_model.pth'))


# predict part
test_dataset = DenoisingDataset(spec_array=spec_array)
    
test_loader = torch.utils.data.DataLoader(
            test_dataset, shuffle=False,
            batch_size=256,
            num_workers=0, pin_memory=True)

test_preds = test_fn(test_loader, model, device)

denoising_auto_encoded_intensity = np.concatenate(test_preds)
denoising_auto_encoded_intensity.shape


# check part
spec_array_test = spec_array[n_train:]

spec = spec_array_test[1]
qqq = denoising_auto_encoded_intensity[n_train:][1]

plt.plot(spec.reshape(-1), color='r')
plt.plot(qqq.reshape(-1))

# to_pickle('/content/drive/My Drive/offline_competition/atma05/denoising_auto_encoded_intensity.pkl', denoising_auto_encoded_intensity)
