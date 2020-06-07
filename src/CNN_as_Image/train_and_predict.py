import numpy as np
import pandas as pd
import albumentations
from albumentations.pytorch import ToTensorV2
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
import scipy.stats as stats
import seaborn as sns
import string
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from torchvision.transforms import functional as F
from contextlib import contextmanager
from collections import Counter, defaultdict, OrderedDict
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_log_error, roc_auc_score
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
import warnings
warnings.filterwarnings('ignore')


from PIL import Image
import matplotlib.pyplot as plt
from keras.applications.densenet import preprocess_input as preprocess_input_dense


def to_pickle(filename, obj):
    with open(filename, mode='wb') as f:
        pickle.dump(obj, f)

        
def unpickle(filename):
    with open(filename, mode='rb') as fo:
        p = pickle.load(fo)
    return p 


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
kaeru_seed = 718
seed_everything(seed=kaeru_seed)

file_path = '../input/atma5-data/'

df_train = pd.read_csv(os.path.join(file_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(file_path, 'test.csv'))


y = df_train['target'].values


fold_id = 0
SEED = 718
num_folds = 5

kf = StratifiedKFold(n_splits = num_folds, random_state = SEED)
splits = list(kf.split(X=df_train, y=df_train[['target']]))

n_train = len(df_train)
df_train = df_train.append(df_test).reset_index(drop=True)

train_idx = splits[fold_id][0]
val_idx = splits[fold_id][1]


fitting = pd.read_csv(os.path.join(file_path, 'fitting.csv'))
df_train = pd.merge(df_train, fitting, on='spectrum_id')


del df_test
gc.collect()


TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8

train_dataset = ImageDataset(
                    df=df_train[:n_train].iloc[train_idx],
                    y=y[train_idx], training=True)
    
train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=0, 
            pin_memory=True)

val_dataset = ImageDataset(
                    df=df_train[:n_train].iloc[val_idx],
                    y=y[val_idx], training=True)
    
val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=VALID_BATCH_SIZE,
            num_workers=0, 
            pin_memory=True)
    
del train_dataset, val_dataset
gc.collect()


device = 'cuda'
EPOCHS = 30

model = AtmaOpticsImageModel()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-4) 


EXP_ID = 'exp_EX_image'

min_loss = 999
best_score = -999
best_epoch = 0
patience = 5
p = 0
for epoch in range(1, EPOCHS + 1):

    print("Starting {} epoch...".format(epoch))

    train_fn(train_loader, model, optimizer, device, scheduler)
    score, val_loss = eval_fn(val_loader, model, device)
    scheduler.step(val_loss)

    if val_loss < min_loss:
        min_loss = val_loss
        best_score = score
        best_epoch = epoch
        torch.save(model.state_dict(), '{}_fold{}.pth'.format(EXP_ID, fold_id))
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
df_test['target'] = 0

test_dataset = ImageDataset(
                    df=df_test,
                    y=None, training=False)
    
test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=64,
            num_workers=0, 
            pin_memory=True)

test_preds = test_fn(test_loader, model, device)


sub = pd.read_csv(os.path.join(file_path, 'atmaCup5__sample_submission.csv'))
sub.target = np.concatenate(test_preds)

