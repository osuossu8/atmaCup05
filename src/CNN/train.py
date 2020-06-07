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
from torch.utils import model_zoo
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import tensorflow as tf

from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')


file_path = '../input/atma5-data/'

df_train = pd.read_csv(os.path.join(file_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(file_path, 'test.csv'))

EXP_ID = 'exp1'
device = 'cuda'
EPOCHS = 5
fold_id = 0
SEED = 718
num_folds = 5
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 128
min_loss = 999
best_score = -999
best_epoch = 0
patience = 4
p = 0

kf = StratifiedKFold(n_splits = num_folds, random_state = SEED)
splits = list(kf.split(X=df_train, y=df_train[['target']]))

n_train = len(df_train)
df_train = df_train.append(df_test).reset_index(drop=True)

train_idx = splits[fold_id][0]
val_idx = splits[fold_id][1]

# fitting = pd.read_csv(os.path.join(file_path, 'fitting__fixed.csv'))
fitting = pd.read_csv(os.path.join(file_path, 'fitting.csv'))
spec_meta_df = unpickle('../input/atma05-spectrum-features/spectrum_based_features.pkl')
spec_meta_df['light_intensity'] = spec_meta_df['light_intensity'].apply(lambda x: padding(x))

df_train = pd.merge(df_train, fitting, on='spectrum_id')
df_train = pd.merge(df_train, spec_meta_df, on='spectrum_filename')


del df_test
gc.collect()


train_dataset = OpticsDataset(df=df_train, indices=train_idx,)
train_loader = torch.utils.data.DataLoader(
                train_dataset, shuffle=True, 
                batch_size=TRAIN_BATCH_SIZE,
                num_workers=0, pin_memory=True)

val_dataset = OpticsDataset(df=df_train, indices=val_idx,)
val_loader = torch.utils.data.DataLoader(
                val_dataset, shuffle=False, 
                batch_size=VALID_BATCH_SIZE,
                num_workers=0, pin_memory=True)

del train_dataset, val_dataset
gc.collect()


model = SimpleModel(4, 512)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-5) 


for epoch in range(1, EPOCHS + 1):

    print("Starting {} epoch...".format(epoch))

    train_fn(train_loader, model, optimizer, device, scheduler)
    score, val_loss, y_true, oof_preds = eval_fn(val_loader, model, device)
    scheduler.step(val_loss)

    if val_loss < min_loss:
        min_loss = val_loss
        best_score = score
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join('{}_fold{}.pth'.format(EXP_ID, fold_id)))
        print("save model at score={} on epoch={}".format(best_score, best_epoch))
        p = 0

    if p > 0: 
        print(f'val loss is not updated while {p} epochs of training')
    p += 1
    if p > patience:
        print(f'Early Stopping')
        break

    print("best score={} on epoch={}".format(best_score, best_epoch))
