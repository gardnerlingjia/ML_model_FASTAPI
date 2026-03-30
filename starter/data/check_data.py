import pandas as pd
df = pd.read_csv("starter/data/census.csv")
df.columns = df.columns.str.strip()
df.to_csv("starter/data/census.csv", index=False)
print(df.head())
print(df.columns)
