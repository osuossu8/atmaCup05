import numpy as np
import pandas as pd
import albumentations
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
    

def image_to_tensor(image, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(image / (255. if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def imshow(images, title=None):
    images = images.numpy().transpose((1, 2, 0))  # (h, w, c)
    images = np.clip(images, 0, 1)
    plt.imshow(images)


def extract_features(df, training):

    model = get_model(pretrained=True)
    model = model.cuda()
    model.eval()

    # register hook to access to features in forward pass
    features = []
    filename = []
    def hook(module, input, output):
        N,C,H,W = output.shape
        output = output.reshape(N,C,-1)
        features.append(output.mean(dim=2).cpu().detach().numpy())
    handle = model._modules.get(layer_name).register_forward_hook(hook)

    dataset = ImageDataset(df, training=training)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    for i_batch, d in tqdm(enumerate(loader), total=len(loader)):
        inputs = d['image'].cuda()
        _ = model(inputs.cuda())
        filename.append(d['spectrum_filename'])

    features_cat = np.concatenate(features)
    filename_cat = np.concatenate(filename, 0)
    
    features = pd.DataFrame(features_cat, columns=[f'image_features_{i}' for i in range(1024)])
    features['spectrum_filename'] = filename_cat
    
    handle.remove()
    del model

    return features

    
kaeru_seed = 718
seed_everything(seed=kaeru_seed)


file_path = '../input/atma5-data/'

df_train = pd.read_csv(os.path.join(file_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(file_path, 'test.csv'))


y = df_train['target'].values

print(df_train.target.value_counts())

fold_id = 0
SEED = 718
num_folds = 5

kf = StratifiedKFold(n_splits = num_folds, random_state = SEED)
splits = list(kf.split(X=df_train, y=df_train[['target']]))


n_train = len(df_train)
df_train = df_train.append(df_test).reset_index(drop=True)

train_idx = splits[fold_id][0]
val_idx = splits[fold_id][1]

print(len(train_idx), len(val_idx))

fitting = pd.read_csv(os.path.join(file_path, 'fitting.csv'))
df_train = pd.merge(df_train, fitting, on='spectrum_id')


del df_test
gc.collect()


model_name = 'densenet121'
layer_name = 'features'
get_model = getattr(torchvision.models, model_name)

features_train = extract_features(df_train[:n_train], training=True)
features_test = extract_features(df_train[n_train:], training=False)

features_df = pd.concat([features_train, features_test]).reset_index(drop=True)

# to_pickle('densenet121_generated_image_features_1024.pkl', features_df)




