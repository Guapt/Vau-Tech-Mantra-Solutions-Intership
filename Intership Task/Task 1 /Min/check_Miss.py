import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv("SAR Rental.csv")

print(dataset.head())

print(dataset.shape)

print(dataset.isnull().sum())

print((dataset.isnull().sum()/dataset.shape[0])*100)

print((dataset.isnull().sum().sum())/(dataset.shape[0]*dataset.shape[1])*100)

print(dataset.notnull().sum().sum())

sns.heatmap(dataset.isnull())
plt.show()
