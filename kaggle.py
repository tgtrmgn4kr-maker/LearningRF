import pandas as pd
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import hashlib
import os
import tarfile
import zipfile
import requests




def download(name, cache_dir=os.path.join('..', 'data')):
    """Download a file inserted into DATA_HUB, return the local filename."""

    url, sha1_hash = DATA_HUB[name]

    os.makedirs(cache_dir, exist_ok=True)

    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache

    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)

    return fname

def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce'
    )
DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
    )

def download_all():
    """Download all files in the DATA_HUB."""
    for name in DATA_HUB:
        download(name)

def log_rmse(net, features, labels):
    "為了在取對數時穩定該值，將小於1的數值取1"
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            loss_value = loss(net(X), y)
            loss_value.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_kaggle_data(k, i, X, y):
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            pass

def main():

    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))
    print(train_data.shape)
    print(test_data.shape)
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
    '''
    (1460, 81) 樣本數、特徵數
    (1459, 80)
       Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice
    0   1          60       RL         65.0       WD        Normal     208500
    1   2          20       RL         80.0       WD        Normal     181500
    2   3          60       RL         68.0       WD        Normal     223500
    3   4          70       RL         60.0       WD      Abnormal     140000
    '''

    # Delete the Id feature bcs it's just a number that doesn't provide any information.
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # After standardizing the data, we can fill the missing values in the
    # remaining features with their mean value.
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # Now all features are numeric and have been normalized/scaled.

    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(all_features.shape)
    # (2919, 331) 特徵數增加到331個
    # 將每個離散值標籤化為獨熱編碼（one-hot encoding）
    # 這是一種常用的處理離散數據的方法，其特點是每個類別都有自己的特徵向量，
    # 且只有一個元素為1，其他元素為0。

    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(
        train_data.SalePrice.values.reshape(-1, 1), # type: ignore
        dtype=torch.float32
    )
    global loss
    loss = nn.MSELoss()
    in_features = train_features.shape[1]

    net = nn.Sequential(
        nn.Linear(in_features, 1)
    )

if __name__ == '__main__':
    main()

