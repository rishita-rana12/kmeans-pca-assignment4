import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# ------------------------
# KMeans without PCA
# ------------------------
kmeans1 = KMeans(n_clusters=3, random_state=42)
labels1 = kmeans1.fit_predict(X)

score1 = adjusted_rand_score(y, labels1)
print("ARI without PCA:", score1)

plt.scatter(X[:,0], X[:,1], c=labels1)
plt.title("KMeans without PCA")
plt.show()

# ------------------------
# PCA + KMeans
# ------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

kmeans2 = KMeans(n_clusters=3, random_state=42)
labels2 = kmeans2.fit_predict(X_pca)

score2 = adjusted_rand_score(y, labels2)
print("ARI with PCA:", score2)

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels2)
plt.title("PCA + KMeans")
plt.show()