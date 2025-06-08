import numpy as np
import torch as th
import json
import torch
import random
from sklearn.preprocessing import  MinMaxScaler
from torch.utils.data import Dataset, DataLoader



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def data_load_single(args, dataset): 

    folder_path = './dataset64time/24_{}_{}.json'.format(dataset,args.task)
    folder_path_u = './dataset64time/{}_user64.json'.format(dataset)
    folder_path_p = './dataset64time/{}_poi64.json'.format(dataset)

    f = open(folder_path,'r')
    data_all = json.load(f)

    X_train = torch.tensor(data_all['X_train'][0]).unsqueeze(1)
    X_test = torch.tensor(data_all['X_test'][0]).unsqueeze(1)
    X_val = torch.tensor(data_all['X_val'][0]).unsqueeze(1)

    print('Data_length', X_train.shape)

    X_train = X_train.numpy()
    X_test = X_test.numpy()
    X_val = X_val.numpy()
    clip_datas_train = np.percentile(X_train, 99.99)

    X_train = np.clip(X_train, 0, clip_datas_train) / clip_datas_train
    X_test = np.clip(X_test, 0, clip_datas_train) / clip_datas_train
    X_val = np.clip(X_val, 0, clip_datas_train) / clip_datas_train


    # --------------------------------------------
    f_u = open(folder_path_u, 'r')
    data_all_u = json.load(f_u)

    X_train_u = torch.tensor(data_all_u['X_train'][0]).unsqueeze(1)
    X_test_u = torch.tensor(data_all_u['X_test'][0]).unsqueeze(1)
    X_val_u = torch.tensor(data_all_u['X_val'][0]).unsqueeze(1)

    X_train_u = X_train_u.numpy()
    X_test_u = X_test_u.numpy()
    X_val_u = X_val_u.numpy()


    clip_datas_train_u = np.percentile(X_train_u, 99.99)

    X_train_u = np.clip(X_train_u, 0, clip_datas_train_u) / clip_datas_train_u
    X_test_u = np.clip(X_test_u, 0, clip_datas_train_u) / clip_datas_train_u
    X_val_u = np.clip(X_val_u, 0, clip_datas_train_u) / clip_datas_train_u




    
    # --------------------------------------------
    f_p = open(folder_path_p, 'r')
    data_all_p= json.load(f_p)

    X_train_p = torch.tensor(data_all_p['X_train']).unsqueeze(1)
    X_test_p = torch.tensor(data_all_p['X_test']).unsqueeze(1)
    X_val_p = torch.tensor(data_all_p['X_val']).unsqueeze(1)


    clip_datas_train_p = X_train_p.max()
    data_p = X_train_p/clip_datas_train_p
    data_p_val = X_val_p / clip_datas_train_p
    data_p_test = X_test_p / clip_datas_train_p
    print('POI_length', data_p.shape)


    # --------------------------------------------


    args.seq_len = X_train.shape[2]
    H, W = X_train.shape[3], X_train.shape[4]
    args.info =(X_train.shape[2],X_train.shape[3],X_train.shape[4])

    X_train_ts = torch.tensor(data_all['timestamps']['train'])
    X_test_ts = torch.tensor(data_all['timestamps']['test'])
    X_val_ts = torch.tensor(data_all['timestamps']['val'])


    my_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data = X_train
    my_scaler.fit(train_data.reshape(-1,1))

    my_scaler0 = MinMaxScaler(feature_range=(-1, 1))
    train_data0 = X_train_u
    my_scaler0.fit(train_data0.reshape(-1,1))


    data_scaled = my_scaler.transform(X_train.reshape(-1,1)).reshape(X_train.shape)
    data_u = my_scaler0.transform(X_train_u.reshape(-1, 1)).reshape(X_train_u.shape).astype(np.float32)
    data = [[data_scaled[i], X_train_ts[i], data_u[i], data_p[i]] for i in range(X_train.shape[0])]


    data_scaled_test = my_scaler.transform(X_test.reshape(-1,1)).reshape(X_test.shape)
    data_u_test = my_scaler0.transform(X_test_u.reshape(-1, 1)).reshape(X_test_u.shape).astype(np.float32)
    data_test = [[data_scaled_test[i], X_test_ts[i], data_u_test[i], data_p_test[i]] for i in range(X_test.shape[0])]


    data_scaled_val = my_scaler.transform(X_val.reshape(-1,1)).reshape(X_val.shape)
    data_u_val = my_scaler0.transform(X_val_u.reshape(-1, 1)).reshape(X_val_u.shape).astype(np.float32)
    data_val = [[data_scaled_val[i], X_val_ts[i], data_u_val[i], data_p_val[i]] for i in range(X_val.shape[0])]



    train_dataset = MyDataset(data)
    val_dataset = MyDataset(data_test)
    test_dataset = MyDataset(data_val)

    batch_size = args.batch_size

    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return  train_loader, test_loader, val_loader, my_scaler




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

