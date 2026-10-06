import argparse, json, copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from partition import dirichlet_partition

p = argparse.ArgumentParser()
p.add_argument("--alpha", type=float, default=0.5)
p.add_argument("--rounds", type=int, default=20)
p.add_argument("--clients", type=int, default=10)
p.add_argument("--local_epochs", type=int, default=1)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)

d = np.load("data/processed.npz", allow_pickle=True)
X_train, y_train = torch.tensor(d["X_train"]), torch.tensor(d["y_train"], dtype=torch.long)
X_test, y_test = torch.tensor(d["X_test"]), d["y_test"]

parts = dirichlet_partition(d["y_train"], a.clients, a.alpha, a.seed)

def make_model():
    return nn.Sequential(nn.Linear(46, 128), nn.ReLU(),
                         nn.Linear(128, 64), nn.ReLU(),
                         nn.Linear(64, 15))

def local_train(global_state, idx):
    model = make_model()
    model.load_state_dict(global_state)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()
    X, y = X_train[idx], y_train[idx]
    for _ in range(a.local_epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss_fn(model(X[b]), y[b]).backward()
            opt.step()
    return model.state_dict()

def aggregate(states, sizes):
    total = sum(sizes)
    return {k: sum(s[k] * (n / total) for s, n in zip(states, sizes))
            for k in states[0]}

global_model = make_model()
n_params = sum(p.numel() for p in global_model.parameters())
# each client downloads and uploads the whole model, 4 bytes per parameter
comm_mb = n_params * 4 * 2 * a.clients / 1e6
print(f"Parameters: {n_params}, communication per round: {comm_mb:.3f} MB")

history = []
for r in range(1, a.rounds + 1):
    g = copy.deepcopy(global_model.state_dict())
    states = [local_train(g, idx) for idx in parts]
    sizes = [len(idx) for idx in parts]
    global_model.load_state_dict(aggregate(states, sizes))

    with torch.no_grad():
        pred = global_model(X_test).argmax(1).numpy()
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    history.append({"round": r, "accuracy": acc, "macro_f1": f1, "comm_mb": comm_mb})
    print(f"Round {r}: acc={acc:.4f}, macro-F1={f1:.4f}")

with open(f"results_alpha{a.alpha}_seed{a.seed}.json", "w") as f:
    json.dump(history, f, indent=2)
