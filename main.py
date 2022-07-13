from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, Y = make_classification(n_features=4, n_classes=3, n_redundant=0, n_informative=3, n_clusters_per_class=2)

plt.title("Multi-class data, 4 informative features, 3 classes", fontsize="large")
plt.scatter(X[:,0], X[:,1], marker="o", c=Y, s=25, edgecolor="k")

plt.show()