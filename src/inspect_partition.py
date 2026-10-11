import os
import pandas as pd

CSV = "data/DNN-EdgeIIoT-dataset.csv"
OUT = "results/partition_inspection.txt"

cols = ["ip.src_host", "ip.dst_host", "mqtt.msg", "mqtt.topic",
        "http.request.method", "dns.qry.name", "Attack_type"]
df = pd.read_csv(CSV, usecols=cols, dtype=str, low_memory=False)

lines = []
def out(text=""):
    print(text)
    lines.append(text)

def mix(attack_series, min_pct=0.1):
    p = attack_series.value_counts(normalize=True) * 100
    return ", ".join(f"{k} {v:.1f}%" for k, v in p.items() if v >= min_pct)

def nonempty(s):
    s = s.fillna("").str.strip()
    return ~s.isin(["", "0", "0.0"])

out(f"Total rows: {len(df)}")
out("ALL ROWS attack mix: " + mix(df["Attack_type"]))

# Part 1: host fields
for col in ["ip.src_host", "ip.dst_host"]:
    s = df[col].fillna("missing")
    out(f"\n=== {col} ===")
    out(f"Distinct values: {s.nunique()}")
    share = s.map(s.value_counts(normalize=True))
    label = s.where(share >= 0.01, "other")
    sizes = label.value_counts()
    out(f"Groups with at least 1% of rows (not counting 'other'): {(sizes.index != 'other').sum()}")
    for v in sizes.index:
        rows = df["Attack_type"][label == v]
        out(f"\n{v}: {len(rows)} rows ({len(rows) / len(df) * 100:.1f}% of data)")
        out("  " + mix(rows))

# Part 2: protocol groups
out("\n=== Protocol fields: top values (to check what 'empty' looks like) ===")
for col in ["mqtt.msg", "mqtt.topic", "http.request.method", "dns.qry.name"]:
    out(f"{col}: {df[col].value_counts(dropna=False).head(5).to_dict()}")

mqtt = nonempty(df["mqtt.msg"]) | nonempty(df["mqtt.topic"])
http = nonempty(df["http.request.method"])
dns = nonempty(df["dns.qry.name"])
groups = {"MQTT": mqtt, "HTTP": http, "DNS": dns,
          "None of the three": ~(mqtt | http | dns)}

out("\n=== Protocol groups ===")
for name, mask in groups.items():
    rows = df["Attack_type"][mask]
    out(f"\n{name}: {len(rows)} rows ({len(rows) / len(df) * 100:.1f}% of data)")
    out("  " + mix(rows))
out(f"\nRows in more than one protocol group: {((mqtt.astype(int) + http.astype(int) + dns.astype(int)) > 1).sum()}")

os.makedirs("results", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nSaved to {OUT}")