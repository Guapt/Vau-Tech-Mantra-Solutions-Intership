import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv(r"C:\Users\reena\Programming Languange\ML\ML_Project\magic04.data", names=cols)
# print(df.head())

# print(df["class"].unique())

df["class"]= (df["class"] == "g").astype(int)
# print(df.head())

# for label in cols[:-1]:
#     plt.hist(df[df["class"]==1][label], color= 'blue', label='gamma', alpha=0.7, density=True)
#     plt.hist(df[df["class"]==0][label], color= 'red', label='hydron', alpha=0.7, density=True)
#     plt.ylabel("Probability")
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()


train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_detaset(dataframe, oversample= False):
    X = dataframe[dataframe.columns[:-1]].values
    Y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X= scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, Y = ros.fit_resample(X,Y)

    data = np.hstack((X, np.reshape(Y, (-1, 1))))

    return data, X, Y

# print(len(train[train["class"]==1]))
# print(len(train[train["class"]==0]))

scale_detaset(train, oversample=True)

train, X_train, Y_train = scale_detaset(train, oversample=True)
valid, X_valid, Y_valid = scale_detaset(valid, oversample=False)
test, X_test, Y_test = scale_detaset(test, oversample=False)

# print(len(Y_train))
# print(sum(Y_train == 1))
# print(sum(Y_train == 0))


knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, Y_train)

Y_pred = knn_model.predict(X_test)

print(classification_report(Y_test, Y_pred))
