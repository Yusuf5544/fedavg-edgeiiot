import pandas as pd

df = pd.read_csv("data/DNN-EdgeIIoT-dataset.csv", low_memory=False)

print("ALL COLUMNS:")
print(list(df.columns))

print("\nTEXT COLUMNS (name, number of unique values):")
for col in df.select_dtypes(exclude="number").columns:
    print(col, df[col].nunique())