import torch
import json
import sys
import os
from full_seq_dataloader import load_sequence_dataset
from tqdm import tqdm
import numpy as np

sys.path.append('./code/')
from models import BiLSTM, BiLSTMAttention, ESM2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# dataset_path = './data/benchmark/multiple_windows.csv'
dataset_path = './data/benchmark/lt2_windows__cvs_gt2.csv'

# model_c_json_path = './code/run_jsons/c_bilstm_att.json'
# model_n_json_path = './code/run_jsons/n_bilstm_att.json'

# model_c_params = './params/models/c_BiLSTM_att.pt'
# model_n_params = './params/models/n_BiLSTM_att.pt'

model_c_json_path = './code/run_jsons/c_esm2.json'
model_n_json_path = './code/run_jsons/n_esm2.json'

model_c_params = './params/models/c_ESM2.pt'
model_n_params = './params/models/n_ESM2.pt'

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

# vocab = torch.load("./params/vocab.pt").to(DEVICE)
esm2, vocab = torch.hub.load("facebookresearch/esm:main", "esm2_t30_150M_UR50D")
tokenizer = vocab.get_batch_converter()

# model_c_conf = {
#     "vocab_size": len(vocab),
#     "embedding_dim": model_c_json['embedding_dim'],
#     "rnn_size": model_c_json['rnn_size1'],
#     # "rnn_size1": model_c_json['rnn_size1'],
#     # "rnn_size2": model_c_json['rnn_size2'],
#     "hidden_size": model_c_json['linear_size1'],
#     "dropout": model_c_json['dropout'],
#     "out_neurons": model_c_json['out_neurons'],
#     "num_heads": model_c_json['num_heads1'],
#     # "seq_len": model_c_json['seq_len'],
#     # "batch_norm": model_c_json['batch_norm'],
# }

# model_n_conf = {
#     "vocab_size": len(vocab),
#     "embedding_dim": model_n_json['embedding_dim'],
#     "rnn_size": model_n_json['rnn_size1'],
#     # "rnn_size1": model_n_json['rnn_size1'],
#     # "rnn_size2": model_n_json['rnn_size2'],
#     "hidden_size": model_n_json['linear_size1'],
#     "dropout": model_n_json['dropout'],
#     "out_neurons": model_n_json['out_neurons'],
#     "num_heads": model_n_json['num_heads1'],
#     # "seq_len": model_n_json['seq_len'],
#     # "batch_norm": model_n_json['batch_norm'],
# }

model_c_conf = {
    "pretrained_model": esm2,
    "dropout": model_c_json['dropout'],
    "out_neurons": model_c_json['out_neurons'],
}

# model_n_conf = {
#     "pretrained_model": esm2,
#     "dropout": model_n_json['dropout'],
#     "out_neurons": model_n_json['out_neurons'],
# }

print("Loading models")

# model_c = BiLSTM(**model_c_conf).to(DEVICE)
# model_n = BiLSTM(**model_n_conf).to(DEVICE)

# model_c = BiLSTMAttention(**model_c_conf).to(DEVICE)
# model_n = BiLSTMAttention(**model_n_conf).to(DEVICE)

model_c = ESM2(**model_c_conf).to(DEVICE)
# model_n = ESM2(**model_n_conf).to(DEVICE)

model_c.load_state_dict(torch.load(model_c_params))
# model_n.load_state_dict(torch.load(model_n_params))

model_c.eval()
# model_n.eval()

print("Predicting")

# n_preds_list = []
c_preds_list = []

for data in tqdm(dataset):
    windows, n_lbl, c_lbl, clvs = data
    print(f"windows: {len(windows)}, {len(windows[0])}")
    batch = [(c_l, w) for c_l, w in zip(c_lbl, windows)]
    lbl, _, seq = tokenizer(batch)
    seq = seq.to(DEVICE)
    seq = seq.long()
    print(f"seq shape: {seq.shape}")

    # windows = torch.tensor([tokenizer(w) for w in windows]).to(DEVICE)
    # windows = torch.tensor([vocab(list(w)) for w in windows]).to(DEVICE)
    # n_preds = model_n(windows)

    c_preds_full = []
    for i in range(0, len(seq), 32):
        c_preds_full.append(model_c(seq[i:i + 32]).cpu())

    # c_preds = model_c(seq)
    c_preds = torch.cat(c_preds_full, dim=0)
    # n_preds_list.append(n_preds.detach().cpu().numpy())
    c_preds_list.append(c_preds.detach().cpu().numpy())

print("Saving predictions")
np.save(f'{results_path}/c_esm2.npy', np.array(c_preds_list, dtype=object), allow_pickle=True)
# np.save(f'{results_path}/n_esm2.npy', np.array(n_preds_list, dtype=object), allow_pickle=True)
