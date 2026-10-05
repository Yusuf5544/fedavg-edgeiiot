import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

d = np.load("data/processed.npz", allow_pickle=True)
X_train, y_train = d["X_train"], d["y_train"]
X_test, y_test = d["X_test"], d["y_test"]

model = nn.Sequential(
    nn.Linear(46, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 15),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

Xt = torch.tensor(X_train); yt = torch.tensor(y_train, dtype=torch.long)
Xe = torch.tensor(X_test)

for epoch in range(5):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 512):
        idx = perm[i:i+512]
        opt.zero_grad()
        loss = loss_fn(model(Xt[idx]), yt[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(Xe).argmax(1).numpy()
    print(f"Epoch {epoch+1}: acc={accuracy_score(y_test, pred):.4f}, "
          f"macro-F1={f1_score(y_test, pred, average='macro'):.4f}")