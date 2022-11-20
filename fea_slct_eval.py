import torch
import numpy as np
import torch.nn as nn
import os
from scipy import stats
import torch.nn.functional as F
import math
import pandas as pd
from torch.utils.data import DataLoader
import os
from torch.utils.data import DataLoader
import sys 
import argparse
import logging
from tqdm import tqdm

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2+1,  1)
        self.qz_loga = nn.Parameter(torch.FloatTensor(input_dim))
        self.qz_loga.data.normal_(1, 1e-2)
        self.temperature = max(0.01, math.exp(-0.05*20))

    def forward(self, x, age=0, z=None):
        if z is None:
            z = self.sample_z(sample=True)
        x = z * x
        x = torch.nn.functional.layer_norm(x, x.shape[1:])
        output, hidden = self.lstm(x)
        o = output.mean(1)
        o = torch.hstack([o,age.repeat(o.shape[0]).view(-1,1)])
        o = self.fc(o)
        return o

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def reg(self):
        """Expected L0 norm under the stochastic gates"""
        q0 = self.cdf_qz(0)
        return (1 - q0).sum()

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        return eps

    def sample_z(self, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample: # 
            eps = self.get_eps(self.input_dim)
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(self.qz_loga)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def floatTensor(self, size):
        return torch.FloatTensor(size).to(self.qz_loga.device)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, parts, prediction_target, task):
        
        self.dataset_root = dataset_root
        raw_labels = []
        for part in parts:
            raw_labels.append(pd.read_csv(f'{dataset_root}/intersection_part{part}_new.csv'))
        self.task = task
        self.raw_label = pd.concat(raw_labels)
        self.prediction_target = prediction_target
        self.var = self.raw_label[self.prediction_target].var()
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
            # X = np.concatenate([X1, X3,], axis=-1)
        X = X.reshape(-1, X.shape[-2], X.shape[-1])
        y = self.raw_label.iloc[index][self.prediction_target]
        age = np.float32(self.raw_label.iloc[index]['interview_age'])
        return np.float32(X), np.float32(y).repeat(X.shape[0]), age 


def train_one(epoch, prune_rate):
    global first_plot
    model.train()
    # gradually tune temperature
    model.temperature = max(0.01, math.exp(-0.05*epoch))
    loss_list = []
    tbar = tqdm(train_dataloader, desc=f'Prune_rate: {prune_rate} - Finetuning Train'.format(epoch))
    imgs, labels = None, None
    for i, (img,  label, age) in enumerate(tbar):
        train_samp = model.sample_z(sample=False)
        prune_idx = torch.sort(train_samp)[1][:int(train_samp.shape[0] * prune_rate)]
        train_samp[prune_idx] = 0
        if torch.isnan(label[0][0]):
            continue
        model.zero_grad()
        img = img[0]
        label = label.reshape(-1, 1)
        img, label, age = img.to(device), label.to(device), age.to(device)
        prediction = model(img, age, train_samp)
        loss = torch.nn.MSELoss()(label, prediction)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item() / train_dataset.var)
    return np.mean(loss_list)


def val_one(epoch, prune_rate):
    model.eval()
    loss_list = []
    with torch.no_grad():
        tbar = tqdm(test_dataloader, desc=f'Prune_rate: {prune_rate} - Finetuning Test'.format(epoch))
        preds, trues = [], []
        train_samp = model.sample_z(sample=False)
        train_samp2 = train_samp
        prune_idx = torch.sort(train_samp)[1][:int(train_samp.shape[0] * prune_rate)]
        train_samp[prune_idx] = 0
        for i, (img,  label, age) in enumerate(tbar):
            label = label.reshape(-1)
            if label.shape[0] == 0 or torch.isnan(label[0]):
                continue
            if img.shape[1] == 0:
                continue
            if len(img.shape) == 4:
                img = img.reshape(-1, img.shape[2], img.shape[3])
            img, label, age = img.to(device), label.to(device), age.to(device)
            prediction = model(img, age, train_samp)
            loss = torch.nn.MSELoss()(label[0], prediction.mean())
            trues.append(label[0].item())
            preds.append(prediction.mean().item())
            loss_list.append(loss.item() / test_dataset.var)
        mse = np.mean(loss_list)

        logging.info(f'mse {np.mean(loss_list)}')  
        corr = stats.pearsonr(preds, trues)[0]
        os.makedirs(f'logs/fea_selc_{task}/{train_dataset.prediction_target}/{test_fold}/', exist_ok = True)
        pd.DataFrame(train_samp2.cpu().numpy()).to_csv(f'logs/fea_selc_{task}/{train_dataset.prediction_target}/{test_fold}/prune={prune_idx.shape[0]}_{mse:.2f}_{corr:.2f}', header=False, index=False,)
    return np.mean(loss_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ABCD LSTM Feature Selection Finetuning')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--data_root', type=str, default=1.0, metavar='LR',
                        help='dataset folder')
    parser.add_argument('--task', type=str, help='Task')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--target', type=str, 
                        help='Prediction target: one of nihtbx_cryst_uncorrected, nihtbx_fluidcomp_uncorrected, nihtbx_totalcomp_uncorrected')
    parser.add_argument('--test_fold', type=int,
                        help='five folds, choose from 0-4')
    parser.add_argument('--regcoef', type=float, default=0.25,
                        help='regularization strength')
    parser.add_argument('--model_path', type=str, 
                        help='the path of the model')
    
    args = parser.parse_args()
    device = args.device
    target = args.target
    task = args.task
    test_fold = args.test_fold
    train_folds = [i for i in range(5) if i != test_fold]
    regcoef = args.regcoef
    folder = f'logs/fea_selc_{task}/{test_fold}_{target}_{regcoef}_finetune'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    logging.basicConfig(handlers=[
                        logging.FileHandler(f'{folder}/log.txt', mode='a', encoding=None, delay=False),
                        logging.StreamHandler()
                    ],
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

    train_dataset = Dataset(args.data_root, train_folds, target, task) 
    test_dataset = Dataset(args.data_root, [test_fold], target, task)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = LSTMModel(input_dim=352, hidden_dim=80, lstm_layers=2).to(device)
    model.load_state_dict(torch.load(f'{args.model_path}'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20], verbose=False)
    iter = 0
    # test on different pruining rate
    for p in np.arange(0., 1, 0.1):
        train_one(20, p)
        val_one(20, p)