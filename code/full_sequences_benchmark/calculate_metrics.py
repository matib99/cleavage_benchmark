# %%
import numpy as np
import pandas as pd
from ast import literal_eval
import os
import sys
import matplotlib.pyplot as plt
from sklearn import metrics


model_name = "bilstm"
# model_name = "bilstm_att"
# model_name = "esm2"

training_set="their"
# training_set="my"

run_name=f"{model_name}_{training_set}_data"

dir_path=f"{sys.argv[1]}/{run_name}"
os.makedirs(dir_path, exist_ok=True)

raport=""
raport += f"{run_name}\n"

dataset_path = './data/benchmark/full_test.csv'
# dataset_path = './data/benchmark/lt2_windows__cvs_gt2.csv'
# dataset_path = '../../data/benchmark/multiple_windows.csv'


n_preds_path = f'./data/benchmark/preds_{training_set}/n_{model_name}.npy'
c_preds_path = f'./data/benchmark/preds_{training_set}/c_{model_name}.npy'

data_df = pd.read_csv(dataset_path)
data_df.head()

c_preds = np.load(c_preds_path, allow_pickle=True)
n_preds = np.load(n_preds_path, allow_pickle=True)

cleavages = data_df['cleavages'].apply(literal_eval).values
seq_lens = data_df['protein'].apply(len).values

n_targets = [np.zeros(seq_len + 1) for seq_len in seq_lens]
c_targets = [np.zeros(seq_len + 1) for seq_len in seq_lens]

for i, cleavage in enumerate(cleavages):
    for n, c in cleavage:
        n_targets[i][n - 1] = 1
        c_targets[i][c] = 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

n_preds = [sigmoid(pred) for pred in n_preds]
c_preds = [sigmoid(pred) for pred in c_preds]

c_preds_concat = np.concatenate(c_preds)
n_preds_concat = np.concatenate(n_preds)

c_targets_concat = np.concatenate(c_targets)
n_targets_concat = np.concatenate(n_targets)

print("calculate roc auc")

fpr_c, tpr_c, _ = metrics.roc_curve(c_targets_concat, c_preds_concat)
fpr_n, tpr_n, _ = metrics.roc_curve(n_targets_concat, n_preds_concat)

roc_auc_c = metrics.auc(fpr_c, tpr_c)
roc_auc_n = metrics.auc(fpr_n, tpr_n)

print(f"C - Terminus ROC AUC: {roc_auc_c}")
print(f"N - Terminus ROC AUC: {roc_auc_n}")
raport+="ROC AUC:\n"
raport+=f"N terminus: {roc_auc_n}\n"
raport+=f"C terminus: {roc_auc_c}\n"
raport+="\n"

# plot ROC curves for both models (on separate plots):

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(fpr_c, tpr_c, label=f"C - Terminus ROC AUC: {roc_auc_c}")
ax[0].plot([0, 1], [0, 1], 'k--')
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].set_title('C - Terminus ROC Curve')
ax[0].legend()

ax[1].plot(fpr_n, tpr_n, label=f"N - Terminus ROC AUC: {roc_auc_n}")
ax[1].plot([0, 1], [0, 1], 'k--')
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('N - Terminus ROC Curve')
ax[1].legend()

plt.savefig(f"{dir_path}/roc_auc.png")

# PR Plot and AUC

precisions_c, recalls_c, _ = metrics.precision_recall_curve(c_targets_concat, c_preds_concat)
precisions_n, recalls_n, _ = metrics.precision_recall_curve(n_targets_concat, n_preds_concat)
pr_auc_c = metrics.auc(recalls_c, precisions_c)
pr_auc_n = metrics.auc(recalls_n, precisions_n)

print(f"C - Terminus PR AUC: {pr_auc_c}")
print(f"N - Terminus PR AUC: {pr_auc_n}")
raport+="PR AUC:\n"
raport+=f"N terminus: {pr_auc_n}\n"
raport+=f"C terminus: {pr_auc_c}\n"
raport+="\n"

# Plot PR curves for both models (on separate plots):

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(recalls_c, precisions_c, label=f"C - Terminus PR AUC: {pr_auc_c}")
ax[0].set_xlabel('Recall')
ax[0].set_ylabel('Precision')
ax[0].set_title('C - Terminus PR Curve')
ax[0].legend()

ax[1].plot(recalls_n, precisions_n, label=f"N - Terminus PR AUC: {pr_auc_n}")
ax[1].set_xlabel('Recall')
ax[1].set_ylabel('Precision')
ax[1].set_title('N - Terminus PR Curve')
ax[1].legend()

plt.savefig(f"{dir_path}/pr_auc.png")

trh = 0.5

print("calculate precission/recall")
raport+="\n"
raport+="PRECISSION:\n"
raport+=f"C Terminus: {metrics.precision_score(c_targets_concat, c_preds_concat > trh)}\n"
raport+=f"N Terminus: {metrics.precision_score(n_targets_concat, n_preds_concat > trh)}\n"

raport+="\n"
raport+="RECALL:\n"
raport+=f"C Terminus: {metrics.recall_score(c_targets_concat, c_preds_concat > trh)}\n"
raport+=f"N Terminus: {metrics.recall_score(n_targets_concat, n_preds_concat > trh)}\n"

raport+="\n"
raport+="F1 SCORE:\n"
raport+=f"C Terminus: {metrics.f1_score(c_targets_concat, c_preds_concat > trh)}\n"
raport+=f"N Terminus: {metrics.f1_score(n_targets_concat, n_preds_concat > trh)}\n"

print("positive rank")
# N terminus positives rank
n_pos_quantiles = []

for n_pred, n_target in zip(n_preds, n_targets):
    positive_indices = np.where(n_target == 1)[0]
    normalized_ranks = np.argsort(-n_pred).argsort() / len(n_pred)
    n_pos_quantiles.extend(normalized_ranks[positive_indices])    

c_pos_quantiles = []

for c_pred, c_target in zip(c_preds, c_targets):
    positive_indices = np.where(c_target == 1)[0]
    normalized_ranks = np.argsort(-c_pred).argsort() / len(c_pred)
    c_pos_quantiles.extend(normalized_ranks[positive_indices])


def sanity_check_cn(id):
    print(f"SANITY CHECK {id}")
    true_n = [n-1 for n, c in cleavages[id]]
    true_c = [c for n, c in cleavages[id]]
    print("N terminus")
    print(f"True: {true_n}")
    n_rank = np.argsort(-n_preds[id]).argsort()
    n_sorted_ids = np.argsort(-n_preds[id])
    print(f"True ranks: {n_rank[true_n]}")
    print(f"True probs: {n_preds[id][true_n]}")
    print(f"Top preds: ")
    for i in range(20):
        ii = n_sorted_ids[i]
        print(f"{ii} - {n_preds[id][ii]} {'*' if ii in true_n else ''}")
    print()
    print("C terminus")
    print(f"True: {true_c}")
    c_rank = np.argsort(-c_preds[id]).argsort()
    c_sorted_ids = np.argsort(-c_preds[id])
    print(f"True ranks: {c_rank[true_c]}")
    print(f"True probs: {c_preds[id][true_c]}")
    print(f"Top preds: ")
    for i in range(20):
        ii = c_sorted_ids[i]
        print(f"{ii} - {c_preds[id][ii]} {'*' if ii in true_c else ''}")
    print()

for i in range(4, 10):
    sanity_check_cn(i)
    

    # ranks of the true ns



# plot histograms
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].hist(n_pos_quantiles, bins=50)
ax[0].set_title('N Terminus Positive Rank Distribution')
ax[0].set_xlabel('Rank')
ax[0].set_ylabel('Frequency')

ax[1].hist(c_pos_quantiles, bins=50)
ax[1].set_title('C Terminus Positive Rank Distribution')
ax[1].set_xlabel('Rank')
ax[1].set_ylabel('Frequency')

plt.savefig(f"{dir_path}/positive_ranks.png")

# Heuristic epitope retrieval

def get_epitopes(n_pred, c_pred, min_epitope_len=4, max_epitope_len=16, threshold=0.1):
    epitopes = []
    probs = []
    
    for n in range(len(n_pred)):
        if n_pred[n] < threshold:
            continue
        for c in range(n + min_epitope_len, min(n + max_epitope_len + 1, len(c_pred))):
            if c_pred[c] < threshold:
                continue
            epitopes.append((n+1, c))
            probs.append(n_pred[n] * c_pred[c])
        sorted_indices = np.argsort(-np.array(probs))
        epitopes = [epitopes[i] for i in sorted_indices]
        probs = [probs[i] for i in sorted_indices]

    return epitopes, probs

## SANITY CHECK
def sanity_check(id, max_eps=30):
    s_ep, s_pr = get_epitopes(n_preds[id], c_preds[id])
    t_eps = cleavages[id]
    t_starts = [n for n, _ in t_eps]
    t_ends = [c for _, c in t_eps]
    print("SANITY CHECK")
    print("Predicted epitopes:")
    for ep, pr in zip(s_ep[:max_eps], s_pr[:max_eps]):
        s = ep[0] in t_starts
        e = ep[1] in t_ends
        print(f"{(100*pr):.2f}% -- {ep} {'*' if s else '-'}{'*' if e else '-'}")
    print("True epitopes:")
    for ep in t_eps:
        rank = s_ep.index(ep) if ep in s_ep else None
        prob = 0 if rank is None else s_pr[rank]
        print(f"{ep} Rank: {'-' if rank is None else rank} ({'-' if prob is None else (100*prob):.2f}%)")
    print("")

for i in range(4, 10):
    sanity_check(i)
# %%
def calculate_at_K_metrics(n_preds, c_preds, cleavages, Ks=[1, 2, 3, 5, 10]):
    metrics = {
        k: {
            'precision': [],
            'recall': [],
            'rprecision': []
        } for k in Ks
    }
    for n_pred, c_pred, cleavs in zip(n_preds, c_preds, cleavages):
        pred_epitopes, _ = get_epitopes(n_pred, c_pred)
        for k in Ks:
            r = int(len(cleavs))
            s = min(k, r)
            pred_epitopes_k = pred_epitopes[:k]
            intersection = set(tuple(x) for x in pred_epitopes_k) & set(tuple(x) for x in cleavs)
            metrics[k]['precision'].append(len(intersection) / k)
            metrics[k]['recall'].append(len(intersection) / r)
            pred_epitopes_s = pred_epitopes[:s]
            intersection = set(tuple(x) for x in pred_epitopes_s) & set(tuple(x) for x in cleavs)
            metrics[k]['rprecision'].append(len(intersection) / s)
    for k in Ks:
        metrics[k]['precision']=np.mean(metrics[k]['precision'])
        metrics[k]['recall']=np.mean(metrics[k]['recall'])
        metrics[k]['rprecision']=np.mean(metrics[k]['rprecision'])
    return metrics


print("AT K metrics")
at_k_metrics=calculate_at_K_metrics(n_preds, c_preds, cleavages,)
raport+="\n"
raport+="AT K METRICS\n"
for k in at_k_metrics:
    raport+="\n"
    raport+=f"K = {k}\n"
    raport+=f"PRECISION: {at_k_metrics[k]['precision']}\n"
    raport+=f"RECALL: {at_k_metrics[k]['recall']}\n"
    raport+=f"RPRECISION: {at_k_metrics[k]['rprecision']}\n"

with open(f"{dir_path}/raport.txt", "w") as rf:
    rf.write(raport)
    
print(raport)
