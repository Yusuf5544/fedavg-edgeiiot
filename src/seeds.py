import json
import numpy as np

accs, f1s = [], []
for s in [42, 1, 2]:
    h = json.load(open(f"results_alpha0.5_seed{s}.json"))[-1]
    accs.append(h["accuracy"]); f1s.append(h["macro_f1"])
    print(f"seed {s}: acc={h['accuracy']:.4f}, macro-F1={h['macro_f1']:.4f}")

print(f"\nalpha=0.5 mean ± std over 3 seeds")
print(f"accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")