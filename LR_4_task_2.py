import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris['data']
y = iris['target']

kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(X)

y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.title("Кластеризація K-середніх (Iris)")
plt.xlabel("Довжина чашолистка")
plt.ylabel("Ширина чашолистка")
plt.show()
