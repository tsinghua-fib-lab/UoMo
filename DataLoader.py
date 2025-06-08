import numpy as np
import torch as th
import json
import torch
import datetime
import copy
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split



class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def cal_percentage(data, percentage):
    return torch.quantile(data, percentage)
    
def clip_data(data, threshold):
    return torch.where(data > threshold, threshold, data)

def data_load_single(args, dataset): 

    folder_path = './dataset64time/24_{}_{}.json'.format(dataset,args.task)

    #folder_path = '../autodl-tmp/1_12/{}_{}{}.json'.format(dataset,args.task, args.length0)
    f = open(folder_path,'r')
    data_all = json.load(f)

    X_train = torch.tensor(data_all['X_train'][0]).unsqueeze(1)

    X_test = torch.tensor(data_all['X_test'][0]).unsqueeze(1)
    X_val = torch.tensor(data_all['X_val'][0]).unsqueeze(1)

    # X_train = torch.cat((X_train,X_test,X_val),dim = 0)
    print('Data_length', X_train.shape)
    X_train = X_train.numpy()
    X_test = X_test.numpy()
    X_val = X_val.numpy()

    clip_datas_train = np.percentile(X_train, 99.99)
    #clip_datas_train = cal_percentage(X_train.view(-1,1),0.95)
    #X_train = clip_data(X_train,clip_datas_train)/clip_datas_train



    X_train = np.clip(X_train, 0, clip_datas_train) / clip_datas_train
    # X_train = torch.tensor(X_train)
    X_test = np.clip(X_test, 0, clip_datas_train) / clip_datas_train
    X_val = np.clip(X_val, 0, clip_datas_train) / clip_datas_train

    args.seq_len = X_train.shape[2]
    H, W = X_train.shape[3], X_train.shape[4]
    args.info =(X_train.shape[2],X_train.shape[3],X_train.shape[4])


    X_train_ts = torch.tensor(data_all['timestamps']['train'])
    X_test_ts = torch.tensor(data_all['timestamps']['test'])
    X_val_ts = torch.tensor(data_all['timestamps']['val'])
    # X_train_ts = torch.cat((X_train_ts, X_test_ts, X_val_ts), dim=0)


    # my_scaler = MinMaxNormalization()
    # MAX = max(torch.max(X_train).item(), torch.max(X_test).item(), torch.max(X_val).item())
    # MIN = min(torch.min(X_train).item(), torch.min(X_test).item(), torch.min(X_val).item())
    # my_scaler.fit(np.array([MIN, MAX]))

    # 首先划分训练集和临时集（验证集 + 测试集）
    # train_idx, temp_idx = train_test_split(np.arange(len(X_train)), test_size=0.3, random_state=42)
    # 然后将临时集划分为验证集和测试集
    # val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    my_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data = X_train
    my_scaler.fit(train_data.reshape(-1,1))

    # 对所有子集进行标准化
    # 对所有子集进行标准化
    data_scaled = my_scaler.transform(X_train.reshape(-1,1)).reshape(X_train.shape)
    data = [[data_scaled[i], X_train_ts[i]] for i in range(X_train.shape[0])]

    data_scaled_test = my_scaler.transform(X_test.reshape(-1,1)).reshape(X_test.shape)
    data_test = [[data_scaled_test[i], X_test_ts[i]] for i in range(X_test.shape[0])]

    data_scaled_val = my_scaler.transform(X_val.reshape(-1,1)).reshape(X_val.shape)
    data_val = [[data_scaled_val[i], X_val_ts[i]] for i in range(X_val.shape[0])]

    # 创建子集
    train_dataset = MyDataset(data)
    val_dataset = MyDataset(data_test)
    test_dataset = MyDataset(data_val)


    batch_size = args.batch_size_taxibj
        # if H + W < 48:
        #     batch_size *= 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return  train_loader, test_loader, val_loader, my_scaler

def data_load_mix(args, data_list):
    data_all = []

    for data in data_list:
        data_all += data

    data_all = th.utils.data.DataLoader(data_all, batch_size=args.batch_size, shuffle=True)

    return data_all


def data_load(args):

    data_all = []
    test_data_all = []
    val_data_all = []
    my_scaler_all = []
    my_scaler_all = {}

    for dataset_name in args.dataset.split('*'):
        data, test_data, val_data, my_scaler = data_load_single(args,dataset_name)
        data_all.append([dataset_name, data])
        test_data_all.append(test_data)
        val_data_all.append(val_data)
        my_scaler_all[dataset_name] = my_scaler

    data_all = [(name,i) for name, data in data_all for i in data]
    random.seed(1111)
    random.shuffle(data_all)
    
    return data_all, test_data_all, val_data_all, my_scaler_all


def data_load_main(args):

    data, test_data, val_data, scaler = data_load(args)

    return data, test_data, val_data, scaler

