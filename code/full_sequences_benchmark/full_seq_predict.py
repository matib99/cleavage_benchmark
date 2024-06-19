import torch
import json
import sys
import os
from full_seq_dataloader import load_sequence_dataset
from tqdm import tqdm
import numpy as np

sys.path.append('./code/')
from models import BiLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


dataset_path = './data/benchmark/lt2_windows__cvs_gt2.csv'

model_c_json_path = './code/run_jsons/c_bilstm.json'
model_n_json_path = './code/run_jsons/n_bilstm.json'

model_c_params = './params/models/c_BiLSTM.pt'
model_n_params = './params/models/n_BiLSTM.pt'

results_path = '/home/matib99/cleavage_benchmark/data/benchmark/preds/'
os.system(f"mkdir -p {results_path}")

print("Loading dataset and model json configs")

dataset = load_sequence_dataset(dataset_path)

model_c_json = json.load(open(model_c_json_path))
model_n_json = json.load(open(model_n_json_path))

if 'batch_norm' not in model_c_json:
    model_c_json['batch_norm'] = False
if 'batch_norm' not in model_n_json:
    model_n_json['batch_norm'] = False

vocab = torch.load("./params/vocab.pt").to(DEVICE)

model_c_conf = {
    "vocab_size": len(vocab),
    "embedding_dim": model_c_json['embedding_dim'],
    "rnn_size1": model_c_json['rnn_size1'],
    "rnn_size2": model_c_json['rnn_size2'],
    "hidden_size": model_c_json['linear_size1'],
    "dropout": model_c_json['dropout'],
    "out_neurons": model_c_json['out_neurons'],
    "seq_len": model_c_json['seq_len'],
    "batch_norm": model_c_json['batch_norm'],
}

model_n_conf = {
    "vocab_size": len(vocab),
    "embedding_dim": model_n_json['embedding_dim'],
    "rnn_size1": model_n_json['rnn_size1'],
    "rnn_size2": model_n_json['rnn_size2'],
    "hidden_size": model_n_json['linear_size1'],
    "dropout": model_n_json['dropout'],
    "out_neurons": model_n_json['out_neurons'],
    "seq_len": model_n_json['seq_len'],
    "batch_norm": model_n_json['batch_norm'],
}

print("Loading models")

model_c = BiLSTM(**model_c_conf).to(DEVICE)
model_n = BiLSTM(**model_n_conf).to(DEVICE)

model_c.load_state_dict(torch.load(model_c_params))
model_n.load_state_dict(torch.load(model_n_params))

model_c.eval()
model_n.eval()

print("Predicting")

n_preds_list = []
c_preds_list = []

for data in tqdm(dataset):
    windows, _, _, clvs = data
    windows = torch.tensor([vocab(list(w)) for w in windows]).to(DEVICE)
    n_preds = model_n(windows)
    c_preds = model_c(windows)
    n_preds_list.append(n_preds.detach().cpu().numpy())
    c_preds_list.append(c_preds.detach().cpu().numpy())

print("Saving predictions")
np.save(f'{results_path}/c_bilstm.npy', np.array(c_preds_list, dtype=object), allow_pickle=True)
np.save(f'{results_path}/n_bilstm.npy', np.array(n_preds_list, dtype=object), allow_pickle=True)
