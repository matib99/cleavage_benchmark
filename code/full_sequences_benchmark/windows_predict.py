import torch
import json
import sys
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc

import matplotlib.pyplot as plt

sys.path.append('./code/')
from utils import read_data
from models import BiLSTM, BiLSTMAttention, ESM2
from loaders import CleavageLoader
from processors import train_or_eval_base

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# model_name = "bilstm"
# model_name = "bilstm_att"
model_name = "esm2"

train_data_type = "their"
# train_data_type = "my"


c_test_dataset_path = './data/c_test.csv'
n_test_dataset_path = './data/n_test.csv'

model_c_json_path = f'./code/run_jsons/c_{model_name}.json'
model_n_json_path = f'./code/run_jsons/n_{model_name}.json'

# model_c_params = f'./params/models/{train_data_type}_data/c_{model_name}.pt'
# model_n_params = f'./params/models/{train_data_type}_data/n_{model_name}.pt'

# model_c_json_path = './code/run_jsons/c_bilstm_att.json'
# model_n_json_path = './code/run_jsons/n_bilstm_att.json'

# model_c_params = './params/models/their_data/c_bilstm_att.pt'
# model_n_params = './params/models/their_data/n_bilstm_att.pt'

# model_c_json_path = './code/run_jsons/c_esm2.json'
# model_n_json_path = './code/run_jsons/n_esm2.json'

model_c_params = './params/models/c_ESM2.pt'
model_n_params = './params/models/n_ESM2.pt'

results_path = f'/home/matib99/cleavage_benchmark/data/benchmark/windows_{train_data_type}'
os.system(f"mkdir -p {results_path}")

print("Loading dataset and model json configs")

c_test_data = read_data(c_test_dataset_path)
n_test_data = read_data(n_test_dataset_path)


model_c_json = json.load(open(model_c_json_path))
model_n_json = json.load(open(model_n_json_path))

if 'batch_norm' not in model_c_json:
    model_c_json['batch_norm'] = False
if 'batch_norm' not in model_n_json:
    model_n_json['batch_norm'] = False

# vocab = torch.load("./params/vocab.pt").to(DEVICE)
# tokenizer = lambda x: vocab(list(x))

esm2, vocab = torch.hub.load("facebookresearch/esm:main", "esm2_t30_150M_UR50D")
tokenizer = vocab.get_batch_converter()



c_loader = CleavageLoader(
    c_test_data,
    c_test_data,
    c_test_data,
    tokenizer=tokenizer,
    batch_size=32,
    num_workers=4,
)

n_loader = CleavageLoader(
    n_test_data,
    n_test_data,
    n_test_data,
    tokenizer=tokenizer,
    batch_size=32,
    num_workers=4,
)


# _, _, c_test_loader = c_loader.load(
#     "BiLSTM", nad=False, unk_idx=0
# )  # unk_idx should be 0
# 
# 
# _, _, n_test_loader = n_loader.load(
#     "BiLSTM", nad=False, unk_idx=0
# )  # unk_idx should be 0

_, _, c_test_loader = c_loader.load(
    "ESM2", nad=False, unk_idx=0
)  # unk_idx should be 0


_, _, n_test_loader = n_loader.load(
    "ESM2", nad=False, unk_idx=0
)  # unk_idx should be 0

# BiLSTM

# model_c_conf = {
#     "vocab_size": len(vocab),
#     "embedding_dim": model_c_json['embedding_dim'],
#     "rnn_size1": model_c_json['rnn_size1'],
#     "rnn_size2": model_c_json['rnn_size2'],
#     "hidden_size": model_c_json['linear_size1'],
#     "dropout": model_c_json['dropout'],
#     "out_neurons": model_c_json['out_neurons'],
#     "seq_len": model_c_json['seq_len'], # bilstm
#     "batch_norm": model_c_json['batch_norm'], # bilstm
# }

# model_n_conf = {
#     "vocab_size": len(vocab),
#     "embedding_dim": model_n_json['embedding_dim'],
#     "rnn_size1": model_n_json['rnn_size1'], # bilstm
#     "rnn_size2": model_n_json['rnn_size2'], # bilstm
#     "hidden_size": model_n_json['linear_size1'],
#     "dropout": model_n_json['dropout'],
#     "out_neurons": model_n_json['out_neurons'],
#     "seq_len": model_n_json['seq_len'], # bilstm
#     "batch_norm": model_n_json['batch_norm'], # bilstm
# }

# BiLSTMAttention

# model_c_conf = {
#     "vocab_size": len(vocab),
#     "embedding_dim": model_c_json['embedding_dim'],
#     "rnn_size": model_c_json['rnn_size1'],
#     "hidden_size": model_c_json['linear_size1'],
#     "dropout": model_c_json['dropout'],
#     "out_neurons": model_c_json['out_neurons'],
#     "num_heads": model_c_json['num_heads1'],
# }
# 
# model_n_conf = {
#     "vocab_size": len(vocab),
#     "embedding_dim": model_n_json['embedding_dim'],
#     "rnn_size": model_n_json['rnn_size1'], # bilstm_att
#     "hidden_size": model_n_json['linear_size1'],
#     "dropout": model_n_json['dropout'],
#     "out_neurons": model_n_json['out_neurons'],
#     "num_heads": model_n_json['num_heads1'], # bilstm_att
# }

model_c_conf = {
    "pretrained_model": esm2,
    "dropout": model_c_json['dropout'],
    "out_neurons": model_c_json['out_neurons'],
}

model_n_conf = {
   "pretrained_model": esm2,
   "dropout": model_n_json['dropout'],
   "out_neurons": model_n_json['out_neurons'],
}

print("Loading models")
###
# model_c = BiLSTM(**model_c_conf).to(DEVICE)
# model_n = BiLSTM(**model_n_conf).to(DEVICE)

# model_c = BiLSTMAttention(**model_c_conf).to(DEVICE)
# model_n = BiLSTMAttention(**model_n_conf).to(DEVICE)

model_c = ESM2(**model_c_conf).to(DEVICE)
model_n = ESM2(**model_n_conf).to(DEVICE)

model_c.load_state_dict(torch.load(model_c_params))
model_n.load_state_dict(torch.load(model_n_params))

model_c.eval()
model_n.eval()

print("Predicting")

def predict(
    model, loader, device
):
    preds, lbls = [], []

    for seq, lbl in tqdm(
        loader,
        desc="Eval: ",
        file=sys.stdout,
        unit="batches",
    ):
        seq, lbl = seq.to(device), lbl.to(device)

        logits = model(seq)
        preds.extend(logits.detach().tolist())
       
        lbls.extend(lbl.detach().tolist())
    return lbls, preds

with torch.no_grad():
    c_lbls, c_preds = predict(model_c, c_test_loader, DEVICE)
    n_lbls, n_preds = predict(model_n, n_test_loader, DEVICE)

c_lbls = np.array(c_lbls)
c_preds = np.array(c_preds)

n_lbls = np.array(n_lbls)
n_preds = np.array(n_preds)

np.save(f"{results_path}/c_{model_name}.npy", c_preds)
np.save(f"{results_path}/n_{model_name}", n_preds)

# Calculating metrics and generating raport

raport = f"{model_name} {train_data_type} data\n"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_metrics(lbls, preds):
    preds = sigmoid(preds)
    preds_bin = (preds > 0.5).astype(int)
    TP = np.sum((preds_bin == 1) & (lbls == 1))
    TN = np.sum((preds_bin == 0) & (lbls == 0))
    FP = np.sum((preds_bin == 1) & (lbls == 0))
    FN = np.sum((preds_bin == 0) & (lbls == 1))

    acc = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    f1 = 2 * prec * rec / (prec + rec)

    roc_auc = roc_auc_score(lbls, preds)

    precisions, recalls, thresholds = precision_recall_curve(lbls, preds)
    pr_auc = auc(recalls, precisions)

    return acc, prec, rec, f1, roc_auc, pr_auc

c_acc, c_prec, c_rec, c_f1, c_roc_auc, c_pr_auc = get_metrics(c_lbls, c_preds)
n_acc, n_prec, n_rec, n_f1, n_roc_auc, n_pr_auc = get_metrics(n_lbls, n_preds)

raport += "\n"
raport += "------------------------------"
raport += f"C terminus\n"
raport += "\n"
raport += f"Accuracy: {c_acc}\n"
raport += f"Precision: {c_prec}\n"
raport += f"Recall: {c_rec}\n"
raport += f"F1: {c_f1}\n"
raport += f"ROC AUC: {c_roc_auc}\n"
raport += f"PR AUC: {c_pr_auc}\n"

raport += "\n"
raport += "------------------------------"
raport += f"N terminus\n"
raport += "\n"
raport += f"Accuracy: {n_acc}\n"
raport += f"Precision: {n_prec}\n"
raport += f"Recall: {n_rec}\n"
raport += f"F1: {n_f1}\n"
raport += f"ROC AUC: {n_roc_auc}\n"
raport += f"PR AUC: {n_pr_auc}\n"

print(raport)

with open(f"{results_path}/raport_{model_name}.txt", "w") as f:
    f.write(raport)

print("Generating Plots")


# generate ROC AUC plots side by side for both models with their auc scores
c_fpr, c_tpr, _ = roc_curve(c_lbls, c_preds)
n_fpr, n_tpr, _ = roc_curve(n_lbls, n_preds)

plt.figure(figsize=(10, 5))
plt.suptitle(f"{model_name} trained on {train_data_type} data")
plt.subplot(1, 2, 1)
plt.plot(c_fpr, c_tpr, label=f"AUC: {c_roc_auc}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("C terminus ROC AUC")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_fpr, n_tpr, label=f"AUC: {n_roc_auc}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("N terminus ROC AUC")
plt.legend()

plt.tight_layout()
plt.savefig(f"{results_path}/roc_auc_{model_name}.png")
plt.close()

# generate PR AUC plots side by side for both models with their auc scores
c_precisions, c_recalls, _ = precision_recall_curve(c_lbls, c_preds)
n_precisions, n_recalls, _ = precision_recall_curve(n_lbls, n_preds)

plt.figure(figsize=(10, 5))
plt.suptitle(f"{model_name} trained on {train_data_type} data")
plt.subplot(1, 2, 1)
plt.plot(c_recalls, c_precisions, label=f"AUC: {c_pr_auc}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("C terminus PR AUC")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_recalls, n_precisions, label=f"AUC: {n_pr_auc}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("N terminus PR AUC")
plt.legend()

plt.tight_layout()
plt.savefig(f"{results_path}/pr_auc_{model_name}.png")
plt.close()

print("Done")



