import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

dataset = pd.read_csv("SAR Rental.csv")

dataset.drop(columns=["package_id"], inplace=True)
dataset.drop(columns=["from_city_id"], inplace=True)
dataset.drop(columns=["to_city_id"], inplace=True)
dataset.drop(columns=["to_date"], inplace=True)

# print(dataset.info())

print(dataset.select_dtypes(include="float64").columns)


# print(dataset.isnull().sum())

si = SimpleImputer(strategy="mean")
ar = si.fit_transform(dataset[['user_id', 'from_area_id', 'to_area_id', 'from_lat', 'from_long',
       'to_lat', 'to_long']])

new_dataset = pd.DataFrame(ar, columns=dataset.select_dtypes(include="float64").columns)

print(new_dataset.isnull().sum())
# print(new_dataset.info())

