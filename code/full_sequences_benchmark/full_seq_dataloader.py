import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader
import pandas as pd
import numpy as np
from ast import literal_eval

class SequencesDataset(Dataset):
    def __init__(self, seq, clvs, pad_letter='X', pad_before=6, pad_after=4):
        self.pad_letter = pad_letter
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seq = seq
        self.clvs = clvs

    def __getitem__(self, idx):
        padded_seq = [self.pad_letter] * self.pad_before + list(self.seq[idx]) + [self.pad_letter] * self.pad_after
        seq_len = len(self.seq[idx])
        window_size = self.pad_before + self.pad_after
        windows = [''.join(padded_seq[i:i+window_size]) for i in range(seq_len + 1)]
        n_targets = torch.zeros(seq_len + 1)
        c_targets = torch.zeros(seq_len + 1)
        for clv in self.clvs[idx]:
            N, C = clv
            n_targets[N - 1] = 1
            c_targets[C - 1] = 1
        return windows, n_targets, c_targets, self.clvs[idx]

    def __len__(self):
        return len(self.seq)

def load_sequence_dataset(path, pad_letter='X', pad_before=6, pad_after=4):
    df = pd.read_csv(path)
    seq = df['protein'].values
    clvs = df['cleavages'].apply(literal_eval).values
    return SequencesDataset(seq, clvs, pad_letter, pad_before, pad_after)