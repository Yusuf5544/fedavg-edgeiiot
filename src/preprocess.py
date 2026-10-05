import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

SEED = 42
np.random.seed(SEED)

df = pd.read_csv("data/DNN-EdgeIIoT-dataset.csv", low_memory=False)

# 1. Remove duplicate rows, separate labels from features
df = df.drop_duplicates()
y_raw = df["Attack_type"]
X = df.drop(columns=["Attack_type", "Attack_label"])

# 2. Drop identifier and payload columns
drop_cols = ["frame.time", "ip.src_host", "ip.dst_host",
             "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
             "http.file_data", "http.request.full_uri",
             "icmp.transmit_timestamp", "http.request.uri.query",
             "tcp.options", "tcp.payload", "tcp.srcport",
             "tcp.dstport", "udp.port", "mqtt.msg"]
X = X.drop(columns=[c for c in drop_cols if c in X.columns])

# 3. Turn remaining text columns into integer codes
for col in X.select_dtypes(exclude="number").columns:
    X[col] = X[col].astype("category").cat.codes

X = X.astype(np.float32)

# 4. Encode the 15 class names as numbers
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 5. Split BEFORE scaling (avoids data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 6. Scale using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# 7. Save
np.savez("data/processed.npz",
         X_train=X_train, y_train=y_train,
         X_test=X_test, y_test=y_test,
         classes=le.classes_)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Classes:", list(le.classes_))