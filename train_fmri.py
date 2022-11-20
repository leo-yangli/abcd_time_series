import os
import sys
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from scipy import stats
import torch.nn.functional as F
import math
import pandas as pd
from collections import Counter
import scipy
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import argparse

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2+1, 1)

    def forward(self, x, age=0):
        x = torch.nn.functional.layer_norm(x, x.shape[1:])
        output, hidden = self.lstm(x)
        o = output.mean(1)
        o = torch.hstack([o, age.repeat(o.shape[0]).view(-1,1)])
        x = torch.nn.functional.layer_norm(x, x.shape[1:])
        o = self.fc(o)
        return o


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, parts, prediction_target, task):
        raw_labels = []
        self.dataset_root = dataset_root
        for part in parts:
            raw_labels.append(pd.read_csv(f'{self.dataset_root}/intersection_part{part}_new.csv'))
        self.task = task
        self.raw_label = pd.concat(raw_labels)
        self.prediction_target = prediction_target
        self.var = self.raw_label[prediction_target].var()
        self.size = self.raw_label.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        id = self.raw_label.iloc[index]['subjectkey']
        if self.task == 'mid':
            X1 = np.load(f'{self.dataset_root}/MID/{id}.npy', allow_pickle=True)
            X1 = X1.reshape(1, -1, X1.shape[0], X1.shape[1])
            X = np.concatenate([X1])
        elif self.task == 'nback':    
            X2 = np.load(f'{self.dataset_root}/NBACK/{id}.npy', allow_pickle=True)
            X2 = X2.reshape(1, -1, X2.shape[0], X2.shape[1])
            X = np.concatenate([X2])
        elif self.task == 'sst':
            X3 = np.load(f'{self.dataset_root}/SST/{id}.npy', allow_pickle=True)
            X3 = X3.reshape(1, -1, X3.shape[0], X3.shape[1])
            X = np.concatenate([X3])
        elif self.task == 'rest':
            X4 = np.load(f'{self.dataset_root}/REST/{id}.npy', allow_pickle=True)
            X4 = X4.reshape(1, -1, X4.shape[0], X4.shape[1])
            X = np.concatenate([X4])
        else:
            X1 = np.load(f'{self.dataset_root}/MID/{id}.npy', allow_pickle=True)
            X2 = np.load(f'{self.dataset_root}/NBACK/{id}.npy', allow_pickle=True)
            X3 = np.load(f'{self.dataset_root}/SST/{id}.npy', allow_pickle=True)
            X4 = np.load(f'{self.dataset_root}/REST/{id}.npy', allow_pickle=True)
            l = min(X1.shape[0], X2.shape[0], X3.shape[0], X4.shape[0])
            X1 = X1[:l].reshape(1, -1, l, X1.shape[1])
            X2 = X2[:l].reshape(1, -1, l, X2.shape[1])
            X3 = X3[:l].reshape(1, -1, l, X3.shape[1])
            X4 = X4[:l].reshape(1, -1, l, X4.shape[1])
            X = np.concatenate([X1, X2, X3, X4], axis=0)
        X = X.reshape(-1, X.shape[-2], X.shape[-1])
        y = self.raw_label.iloc[index][self.prediction_target]
        age = np.float32(self.raw_label.iloc[index]['interview_age'])
        return np.float32(X), np.float32(y).repeat(X.shape[0]), age 


def train_one(epoch):
    global first_plot
    model.train()
    loss_list = []
    tbar = tqdm(train_dataloader, desc='Epoch {} Training'.format(epoch))
    preds, trues = [], []
    for i, (img,  label, age) in enumerate(tbar):
        model.zero_grad()
        img = img[0]
        label = label.reshape(-1, 1)
        if torch.isnan(label[0][0]):
            continue
        img, label, age = img.to(device), label.to(device), age.to(device)
        prediction = model(img, age)
        loss = torch.nn.MSELoss()(label, prediction)
        (loss).backward()
        optimizer.step()
        loss_list.append(loss.item() / train_dataset.var)
        tbar.set_postfix({'mse/var': np.mean(loss_list)})

        trues.append(label[0].item())
        preds.append(prediction.mean().item())
    logging.info(f'Train {epoch} MSE: {np.mean(loss_list)} corr: {stats.pearsonr(preds, trues)[0]}')
    return np.mean(loss_list)


def val_one(epoch):
    model.eval()
    loss_list = []
    with torch.no_grad():
        tbar = tqdm(test_dataloader, desc='Epoch {} Test'.format(epoch))
        preds, trues = [], []
        for i, (img,  label, age) in enumerate(tbar):
            if img.shape[1] == 0:
                continue
            if len(img.shape) == 4:
                img = img.reshape(-1, img.shape[2], img.shape[3])
            label = label.reshape(-1)

            if torch.isnan(label[0]):
                continue
            img, label, age = img.to(device), label.to(device), age.to(device)
            prediction = model(img, age)
            loss = torch.nn.MSELoss()(label[0], prediction.mean())
            
            trues.append(label[0].item())
            preds.append(prediction.mean().item())

            loss_list.append(loss.item() / test_dataset.var)
            tbar.set_postfix({'mse/var': np.mean(loss_list)})

        print('corr', stats.pearsonr(preds, trues)[0])
        print('mae', np.mean(np.abs(np.array(trues) - np.array(preds))))
        logging.info(f'Test {epoch} MSE: {np.mean(loss_list)} corr: {stats.pearsonr(preds, trues)[0]}')
        global bestmse, bestcorr, bestepoch
        if  np.mean(loss_list) < bestmse:
            bestmse = np.mean(loss_list)
            bestcorr = stats.pearsonr(preds, trues)[0]
            bestepoch = epoch
        logging.info(f'best {bestmse}/{bestcorr} @ {bestepoch}')

    return np.mean(loss_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ABCD LSTM')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--data_root', type=str, default=1.0, metavar='LR',
                        help='dataset folder')
    parser.add_argument('--task', type=str, help='Task: mid, nback, sst, rest, all')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--target', type=str, 
                        help='Prediction target: nihtbx_cryst_uncorrected, nihtbx_fluidcomp_uncorrected, nihtbx_totalcomp_uncorrected')
    parser.add_argument('--test_fold', type=int,
                        help='five folds, choose from 0-4')
    args = parser.parse_args()         
    hidden_dim = 80
    lstm_layers = 2
    bestmse = 1000
    bestcorr = 0
    bestepoch = -1
    test_fold = args.test_fold   
    train_folds = [i for i in range(5) if i != test_fold]   
    task = args.task
    target = args.target
    device = args.device
    train_dataset = Dataset(args.data_root, train_folds, target, task)
    test_dataset = Dataset(args.data_root, [test_fold], target, task)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dir = f"logs/fmri_{lstm_layers}_{hidden_dim}"
    Path(dir).mkdir(parents=True, exist_ok=True)
    log_file = f'{dir}/{test_fold}_{task}_{target}.txt'
    logging.basicConfig(handlers=[
                            logging.FileHandler(log_file, mode='w', encoding=None, delay=False),
                            logging.StreamHandler()
                        ],
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    logging.info(log_file)
    
    model = LSTMModel(input_dim=352, hidden_dim=hidden_dim, lstm_layers=lstm_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [8, 16, 30], verbose=True)
    for epoch in range(args.epochs):
        train_one(epoch)
        val_one(epoch)
        scheduler.step()



