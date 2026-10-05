import json
import matplotlib.pyplot as plt

alphas = [0.1, 0.5, 1.0]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
print("| alpha | accuracy | macro-F1 | MB/round |")
print("|---|---|---|---|")
for a in alphas:
    h = json.load(open(f"results_alpha{a}_seed42.json"))
    r = [x["round"] for x in h]
    ax[0].plot(r, [x["accuracy"] for x in h], label=f"alpha={a}")
    ax[1].plot(r, [x["macro_f1"] for x in h], label=f"alpha={a}")
    last = h[-1]
    print(f"| {a} | {last['accuracy']:.4f} | {last['macro_f1']:.4f} | {last['comm_mb']:.2f} |")
for axis, t in zip(ax, ["Accuracy", "Macro-F1"]):
    axis.set_xlabel("Round"); axis.set_title(t); axis.legend()
plt.tight_layout()
plt.savefig("results.png", dpi=150)