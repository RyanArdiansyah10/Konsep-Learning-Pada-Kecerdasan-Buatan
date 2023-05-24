import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, davies_bouldin_score

dataset = pd.read_csv('InternetUser.csv')

print('Sample Data')
print(dataset.head())

print(dataset.isna().sum())

print(dataset.dtypes)

x = dataset.iloc[:, :4]
y = dataset['Other_Social_Network']

xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=0.3, random_state=0)

dt = DecisionTreeClassifier()
dt.fit(xTrain, yTrain)

print('Decision Tree Accuracy: {:.3f}'.format(
    accuracy_score(yTest, dt.predict(xTest))))

plt.figure(figsize=(10, 10))
tree.plot_tree(dt, filled=True)

kMeans = KMeans(n_clusters=3)
labels = kMeans.fit_predict(x)

db_score = davies_bouldin_score(x, labels)
print('Davies Bouldin Score:', db_score)

cols = dataset.columns
plt.scatter(x.loc[labels == 0, cols[0]],
            x.loc[labels == 0, cols[1]],
            s=100, c='purple',
            label='Cluster 0')
plt.scatter(x.loc[labels == 1, cols[0]],
            x.loc[labels == 1, cols[1]],
            s=100, c='orange',
            label='Cluster 1')
plt.scatter(x.loc[labels == 2, cols[0]],
            x.loc[labels == 2, cols[1]],
            s=100, c='green',
            label='Cluster 2')

plt.scatter(kMeans.cluster_centers_[:, 0],
            kMeans.cluster_centers_[:, 1],
            s=100, c='red',
            label='Centroids')

plt.legend()
plt.show()
