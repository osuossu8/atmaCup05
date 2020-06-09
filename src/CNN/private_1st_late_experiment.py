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
import matplotlib.pyplot as plt
from tqdm import tqdm


def loss_fn(logits, targets):
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(logits, targets)
    return loss


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, dilation=1, dropout=0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, filter_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.model(x)


class SimpleModel(nn.Module):
    def __init__(self, in_channels, n_features, hidden_channels=64, out_dim=1):
        super().__init__()
        self.filters = [3, 5, 7, 21, 51, 101]
        for filter_size in self.filters:
            setattr(
                self,
                f"seq{filter_size}", 
                nn.Sequential(
                    Conv1dBlock(in_channels, hidden_channels, filter_size, dropout=0.1),
                    Conv1dBlock(hidden_channels, hidden_channels, filter_size, dropout=0.1),
                ),
            )
        self.cont = nn.Sequential(
            nn.Linear(10, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.last_linear = nn.Sequential(
            nn.Linear(hidden_channels*(len(self.filters)+1), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_dim),
            # nn.Sigmoid()
        )

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, in_seq, in_cont):
        outs = []
        for filter_size in self.filters:
            out = getattr(self, f"seq{filter_size}")(in_seq)
            out, _ = torch.max(out, -1)
            outs.append(out)

        outs.append(self.cont(in_cont))
        out = torch.cat(outs, axis=1)
        out = self.last_linear(out)
        return out


class OpticsDataset:
    def __init__(self, df, spec_array, indices=None):
        if indices is not None:
            self.meta_feature = df[num_cols].iloc[indices].values
            self.target = df.iloc[indices].target.values
            self.spectrum_filename = df.iloc[indices].spectrum_filename.values
            self.spec_array = spec_array[indices]
        else:
            self.meta_feature = df[num_cols].values
            self.target = df.target.values
            self.spectrum_filename = df.spectrum_filename.values
            self.spec_array = spec_array

        self.df = df

    def __len__(self):
        return len(self.target)

    def process_data(self, item, target, spectrum_filename):

        light_intensity = self.spec_array[item]
        targets = [1] if target == 1 else [0]

        return {
            'light_intensity' : light_intensity,
            'targets' : targets,
        }

    def __getitem__(self, item):
        data = self.process_data(
            item,
            self.target[item], 
            self.spectrum_filename[item],
        )

        return {
            'meta_feature': torch.tensor(self.meta_feature[item], dtype=torch.float32),
            'light_intensity': torch.tensor(data["light_intensity"], dtype=torch.float32),
            'targets': torch.tensor(data["targets"], dtype=torch.float32),
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


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    y_true = []
    y_pred = []
    for bi, d in enumerate(tk0):

        meta_feature = d["meta_feature"].to(device, dtype=torch.float32)
        light_intensity = d["light_intensity"].to(device, dtype=torch.float32)
        targets = d["targets"].to(device, dtype=torch.float32)

        model.zero_grad()
        outputs = model(light_intensity.reshape(-1, 4, 512), meta_feature)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        targets = targets.float().cpu().detach().numpy()

        y_true.append(targets)
        y_pred.append(outputs)

        losses.update(loss.item(), light_intensity.size(0))
        tk0.set_postfix(loss=losses.avg)

    y_true_cat = np.concatenate(y_true, 0)
    y_pred_cat = np.concatenate(y_pred, 0)
    avg_precision = average_precision_score(y_true_cat, y_pred_cat)
    print(f'train pr_score : {avg_precision}')


def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    
    y_true = []
    y_pred = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):

            meta_feature = d["meta_feature"].to(device, dtype=torch.float32)
            light_intensity = d["light_intensity"].to(device, dtype=torch.float32)
            targets = d["targets"].to(device, dtype=torch.float32)

            outputs = model(light_intensity.reshape(-1, 4, 512), meta_feature)
            loss = loss_fn(outputs, targets)

            outputs = torch.sigmoid(outputs).cpu().detach().numpy()
            targets = targets.float().cpu().detach().numpy()

            y_true.append(targets)
            y_pred.append(outputs)

            losses.update(loss.item(), light_intensity.size(0))
            tk0.set_postfix(loss=losses.avg)

        y_true_cat = np.concatenate(y_true, 0)
        y_pred_cat = np.concatenate(y_pred, 0)
        avg_precision = average_precision_score(y_true_cat, y_pred_cat)
        print(f'valid pr_score : {avg_precision}')
    return avg_precision, losses.avg


def test_fn(data_loader, model, device):
    model.eval()
    preds = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            meta_feature = d["meta_feature"].to(device, dtype=torch.float32)
            light_intensity = d["light_intensity"].to(device, dtype=torch.float32)

            outputs = model(light_intensity.reshape(-1, 4, 512), meta_feature)
            outputs = torch.sigmoid(outputs).cpu().detach().numpy()
            preds.append(outputs)
    return preds


file_path = '/content/drive/My Drive/offline_competition/atma05/'

df_train = pd.read_csv(os.path.join(file_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(file_path, 'test.csv'))

EXP_ID = 'expXX_cnn'
device = 'cuda'
EPOCHS = 30
fold_id = 0
SEED = 718
num_folds = 5
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 128
min_loss = 999
best_score = -999
best_epoch = 0
patience = 5
p = 0
num_cols = ['exc_wl', 'params0', 'params1', 'params2', 'params3',
            'params4', 'params5', 'params6', 'rms', 'beta']


kf = StratifiedKFold(n_splits = num_folds, random_state = SEED)
splits = list(kf.split(X=df_train, y=df_train[['target']]))

n_train = len(df_train)
df_train = df_train.append(df_test).reset_index(drop=True)

train_idx = splits[fold_id][0]
val_idx = splits[fold_id][1]

fitting = pd.read_csv(os.path.join(file_path, 'fitting__fixed.csv'))
spec_meta_df = unpickle('/content/drive/My Drive/offline_competition/atma05/spectrum_metafeatures.pkl')
spec_meta_df['light_intensity'] = spec_meta_df['light_intensity'].apply(lambda x: padding(x))

df_train = pd.merge(df_train, fitting, on='spectrum_id')
df_train = pd.merge(df_train, spec_meta_df, on='spectrum_filename')

DAE_intensity = unpickle('/content/drive/My Drive/offline_competition/atma05/denoising_auto_encoded_intensity.pkl')

residual = (np.stack(df_train['light_intensity'].values) - DAE_intensity)

spec_array = np.stack([
            scipy.signal.savgol_filter(np.stack(df_train['light_intensity'].values), 5, 2, deriv=0, axis=1),  # なめらかにしただけ   
            scipy.signal.savgol_filter(np.stack(df_train['light_intensity'].values), 5, 2, deriv=1, axis=1),  # 1次微分
            scipy.signal.savgol_filter(np.stack(df_train['light_intensity'].values), 5, 2, deriv=2, axis=1),  # 2次微分
            residual, # 生波形と DAE で生成したものの残差
        ], axis=1)

# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75116#442741
spec_array /= spec_array.std(axis=2).reshape(-1, 4, 1)


del df_test
gc.collect()


scaler = sklearn.preprocessing.StandardScaler()
for c in num_cols:
    col = df_train[c].values.reshape(-1, 1)
    df_train[c] = scaler.fit_transform(col)


train_dataset = OpticsDataset(df=df_train, spec_array=spec_array, indices=train_idx,)
train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, 
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=0, pin_memory=True)

val_dataset = OpticsDataset(df=df_train, spec_array=spec_array, indices=val_idx,)
val_loader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, 
            batch_size=VALID_BATCH_SIZE,
            num_workers=0, pin_memory=True)
    
del train_dataset, val_dataset
gc.collect()


model = SimpleModel(4, 512)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)


for epoch in range(1, EPOCHS + 1):
    print("Starting {} epoch...".format(epoch))
    train_fn(train_loader, model, optimizer, device, scheduler)
    score, val_loss = eval_fn(val_loader, model, device)
    scheduler.step(val_loss)

    if val_loss < min_loss:
        min_loss = val_loss
        best_score = score
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join('/content/drive/My Drive/offline_competition/atma05', '{}_fold{}.pth'.format(EXP_ID, fold_id)))
        print("save model at score={} on epoch={}".format(best_score, best_epoch))
        p = 0
            
    if p > 0: 
        print(f'val loss is not updated while {p} epochs of training')
    p += 1
    if p > patience:
        print(f'Early Stopping')
        break
    print("best score={} on epoch={}".format(best_score, best_epoch))


df_test = df_train[n_train:].reset_index(drop=True)
spec_array_test = spec_array[n_train:]
df_test['target'] = 0
test_dataset = OpticsDataset(df=df_test, spec_array=spec_array_test)
    
test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=256,
            num_workers=0, 
            pin_memory=True
        )

test_preds = test_fn(test_loader, model, device)


