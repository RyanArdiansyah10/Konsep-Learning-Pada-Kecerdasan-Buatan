import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, davies_bouldin_score

dataset = pd.read_csv('gender.csv')

print('Sample Data')
print(dataset.head())

dataset.isna().sum()

dataset.dtypes

x = dataset.iloc[:, :4]
y = dataset['gender']

xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=0.3, random_state=0)

dt = DecisionTreeClassifier()
dt.fit(xTrain, yTrain)

print('Decisoin Tree Accuracy: {:.3f}'.format(
    accuracy_score(yTest, dt.predict(xTest))))

plt.figure(figsize=(5, 5))
tree.plot_tree(dt, filled=True)

KMeans = KMeans(n_clusters=3)
labels = KMeans.fit_predict(x)

db_score = davies_bouldin_score(x, labels)
print(db_score)

cols = dataset.columns
plt.scatter(x.loc[labels == 0, cols[0]],
            x.loc[labels == 0, cols[1]],
            s=100, c='purple', label='setosa')
plt.scatter(x.loc[labels == 1, cols[0]],
            x.loc[labels == 1, cols[1]],
            s=100, c='orange', label='versicolour')
plt.scatter(x.loc[labels == 2, cols[0]],
            x.loc[labels == 2, cols[1]],
            s=100, c='green', label='virginica')
plt.scatter(KMeans.cluster_centers_[:, 0],
            KMeans.cluster_centers_[:, 1],
            s=100, c='red', label='centroids')
