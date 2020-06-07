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
import matplotlib.pyplot as plt

from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')



file_path = '/content/drive/My Drive/offline_competition/atma05/'
df_train = pd.read_csv(os.path.join(file_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(file_path, 'test.csv'))
y = df_train['target'].values
fold_id = 0
SEED = 718
num_folds = 5

kf = StratifiedKFold(n_splits = num_folds, random_state = SEED)
splits = list(kf.split(X=df_train, y=df_train[['target']]))

stratified_groupk = True
if stratified_groupk:
    df_train['chip_id'], _ = pd.factorize(df_train['chip_id'])
    kf = stratified_group_k_fold(X=df_train, y=df_train['target'],
                                 groups=df_train['chip_id'], 
                                 k = num_folds, seed=SEED)
    splits = [(np.array(trn_idx), np.array(val_idx)) for (trn_idx, val_idx) in kf]

n_train = len(df_train)
df_train = df_train.append(df_test).reset_index(drop=True)
train_idx = splits[fold_id][0]
val_idx = splits[fold_id][1]
fitting = pd.read_csv(os.path.join(file_path, 'fitting__fixed.csv'))
df_train = pd.merge(df_train, fitting, on='spectrum_id')
del df_test
gc.collect()


num_features = ['exc_wl', 'layout_a', 'layout_x', 'layout_y', 'pos_x'] + list(fitting.columns[1:]) \
               + ['inclination', 'intensity_max', 'intensity_min', 'intensity_skew', 'intensity_kurt'] \
               + ['k4_means_cluster'] \
               + tsfresh_feature_cols_A \
               + X_PCA_cols \
               + W_NMF_cols \
               + ['peak_near_sum_light_intensity'] \
               + ['peak_near_sum_dft_polar_light_intensity'] \
               + log_fftn_peak_around_light_intensity_cols \
               + log_fftn_peak_around_dft_polar_light_intensity_cols \


X = df_train[num_features]

feature_name = list(X.columns)

cat_model = Catboost()

cat_models, oof_preds_cat, test_cat_preds, feature_importance_cat, evals_results_cat = cat_model.cv(
            y, X[:n_train], X[n_train:], feature_name, splits, cat_config
        )
