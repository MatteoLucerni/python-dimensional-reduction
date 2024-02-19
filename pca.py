import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

cols = [
    "label",
    "alcol",
    "acido malico",
    "cenere",
    "alcalinità della cenere",
    "magnesio",
    "fenoli totali",
    "flavonoidi",
    "fenoli non-flavonoidi",
    "proantocianidine",
    "intensità del colore",
    "tonalità",
    "OD280/OD315 dei vini diluiti",
    "prolina",
]


wines = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    names=cols,
)

wines.head()

X = wines.drop("label", axis=1).values
Y = wines["label"].values

ss = StandardScaler()

X = ss.fit_transform(X)

# fixed components
pca = PCA(n_components=2)

X_pc = pca.fit_transform(X)

plt.xlabel("First component")
plt.ylabel("Second component")
plt.scatter(X_pc[:, 0], X_pc[:, 1], c=Y)
plt.show()

# analisys n components
pca = PCA(n_components=None)
pca.fit(X)

pevr = pca.explained_variance_ratio_

print(pevr)

plt.step(range(1, 14), np.cumsum(pevr), where="mid")
plt.bar(range(1, 14), pevr)
plt.ylabel("Var")
plt.xlabel("Components")
plt.show()
