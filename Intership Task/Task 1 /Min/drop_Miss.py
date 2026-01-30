import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("SAR Rental.csv")
dataset.head(4)

print(dataset.shape)

# print(dataset.isnull().sum())

dataset.drop(columns=["package_id"], inplace=True)
dataset.drop(columns=["from_city_id"], inplace=True)
dataset.drop(columns=["to_city_id"], inplace=True)
dataset.drop(columns=["to_date"], inplace=True)

# print(dataset.isnull().sum())

print(dataset.shape)

dataset.dropna(inplace=True)

print(dataset.isnull().sum())

print(dataset.shape)

# sns.heatmap(dataset.isnull())
# plt.show()

