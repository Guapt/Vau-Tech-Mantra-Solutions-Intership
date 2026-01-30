import pandas as pd
import seaborn as sns

dataset = pd.read_csv('SAR Rental.csv')

# print(dataset.head())

# print(dataset.isnull().sum())

dataset.drop(columns=["package_id"], inplace=True)
dataset.drop(columns=["from_city_id"], inplace=True)
dataset.drop(columns=["to_city_id"], inplace=True)
dataset.drop(columns=["to_date"], inplace=True)

# print(dataset.isnull().sum())f

# print(dataset.fillna(10).head(10))


# print(dataset.info())

# print(dataset.fillna(method="bfill"))
# print(dataset.fillna(method="ffill"))
# print(dataset.fillna(method="ffill", axis = 1))
# print(dataset.fillna(method="bfill", axis = 1))


# print(dataset.isnull().sum())


# dataset["to_lat"].fillna(dataset["to_lat"].mode()[0], inplace=True)
# print(dataset["to_lat"].mode()[0])
# print(dataset["to_lat"].fillna(dataset["to_lat"].mode()[0]))

# print(dataset.select_dtypes(include="object"))

# print(dataset.select_dtypes(include="object").isnull().sum())

for i in dataset.select_dtypes(include="object").columns:
    dataset.fillna(dataset[i].mode()[0], inplace=True)
