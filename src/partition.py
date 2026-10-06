import numpy as np

def dirichlet_partition(y, n_clients=10, alpha=0.5, seed=42, min_size=10):
       """Divide the training data between clients. A lower alpha gives each client fewer classes."""
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    while True:
        client_idx = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            props = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                client_idx[k].extend(part.tolist())
        if min(len(ci) for ci in client_idx) >= min_size:
            break
    return [np.array(ci) for ci in client_idx]

if __name__ == "__main__":
    d = np.load("data/processed.npz", allow_pickle=True)
    y = d["y_train"]
    for alpha in [0.1, 0.5, 1.0]:
        parts = dirichlet_partition(y, alpha=alpha)
        print(f"\nalpha={alpha}")
        for k, p in enumerate(parts):
            counts = np.bincount(y[p], minlength=15)
            print(f"client {k}: size={len(p):7d}  top class={counts.argmax():2d} "
                  f"({counts.max() / len(p):.0%})")
