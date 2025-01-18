import pandas as pd
import numpy as np

data = pd.read_csv('dataset/harga_rumah.csv') # membaca file csv

print(data.head())

df = pd.DataFrame(data)

# print(np.array(df).shape)

X = df.iloc[:, 0:3]
print(np.array(X).shape)