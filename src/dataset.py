import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
#===========統計特徵=============
def compute_stat_features(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    max_ = data.max(axis=0)
    min_ = data.min(axis=0)
    rms = np.sqrt((data ** 2).mean(axis=0))
    
    return np.concatenate([mean, std, max_, min_, rms])  # shape (30,)

class SwingDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        unique_id = row['unique_id']
        path = os.path.join(self.data_dir, f"{unique_id}.txt")
        data = np.loadtxt(path)
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        ts_tensor = torch.tensor(data.T, dtype=torch.float32)
        stat_tensor = torch.tensor(compute_stat_features(data), dtype=torch.float32)

        gender = row['gender'] - 1
        hand = row['hold racket handed'] - 1
        players = row['play years']
        level = row['level'] - 2
        mode = row['mode'] - 1

        return ts_tensor, stat_tensor, gender, hand, players, level, mode

class TestSwingDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row['unique_id']
        mode = row['mode'] - 1
        path = os.path.join(self.data_dir, f"{uid}.txt")
        data = np.loadtxt(path)
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        ts_tensor = torch.tensor(data.T, dtype=torch.float32)
        stat_tensor = torch.tensor(compute_stat_features(data), dtype=torch.float32)
        return uid, ts_tensor, stat_tensor, mode

def collate_fn(batch):
    ts_data, stats_data, gender, hand, players, level, mode = zip(*batch)
    ts_padded = pad_sequence([d.permute(1, 0) for d in ts_data], batch_first=True, padding_value=0).permute(0, 2, 1)
    
    return (
        ts_padded,
        torch.stack(stats_data),
        torch.tensor(gender, dtype=torch.float32),
        torch.tensor(hand, dtype=torch.float32),
        torch.tensor(players, dtype=torch.long),
        torch.tensor(level, dtype=torch.long),
        torch.tensor(mode, dtype=torch.long),
    )

def test_collate(batch):
    uids, ts, stat, mode = zip(*batch)
    ts_padded = pad_sequence([t.permute(1, 0) for t in ts], batch_first=True).permute(0, 2, 1)

    return uids, ts_padded, torch.stack(stat), torch.tensor(mode)

def get_dataloader(input_df_or_path, data_dir, batch_size=32, shuffle=True):
    if isinstance(input_df_or_path, str):
        df = pd.read_csv(input_df_or_path)
    else:
        df = input_df_or_path.copy()

    dataset = SwingDataset(df, data_dir)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def get_test_dataloader(csv_path, data_dir, batch_size=32):
    df = pd.read_csv(csv_path)
    dataset = TestSwingDataset(df, data_dir)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collate)
