# Tutorial 2 Problem 4
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = [ [1.2234, 0.3302, 123.50, 0.0081, 30033.81, 1],
[1.3456, 0.3208, 113.24, 0.0067, 29283.18, -1],
[0.9988, 0.2326, 133.45, 0.0093, 36034.33, 1],
[1.1858, 0.4301, 128.55, 0.0077, 34037.35, 1],
[1.1533, 0.3853, 116.70, 0.0066, 22033.58, -1],
[1.2755, 0.3102, 118.30, 0.0098, 30183.65, 1],
[1.0045, 0.2901, 123.52, 0.0065, 31093.98, -1],
[1.1131, 0.3912, 113.15, 0.0088, 29033.23, -1] ]

df = pd.DataFrame(data)
df.head(7)

from sklearn import preprocessing

# Z-score scaling
df_scaled = preprocessing.scale(df)

print(df_scaled.mean(axis=0))
print(df_scaled.std(axis=0))

# min-max scaling
mix_max_scale = preprocessing.MinMaxScaler()
df_minax = mix_max_scale.fit_transform(df)