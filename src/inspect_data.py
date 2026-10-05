import pandas as pd

df = pd.read_csv("data/DNN-EdgeIIoT-dataset.csv", low_memory=False)
print("Shape:", df.shape)
print(df.dtypes.value_counts())
print(df["Attack_type"].value_counts())