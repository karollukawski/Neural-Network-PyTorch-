from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

X, Y = make_classification(n_features=4, n_classes=3, n_redundant=0, n_informative=3, n_clusters_per_class=2)

plt.title("Multi-class data, 4 informative features, 3 classes", fontsize="large")
plt.scatter(X[:,0], X[:,1], marker="o", c=Y, s=25, edgecolor="k")

plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=123)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

class Data(Dataset):
    def __init__(self):
        self.X = torch.from_numpy(X_train)
        self.Y = torch.from_numpy(Y_train)
        self.len = self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    def __len__(self):
        return self.len

data=Data()
loader = DataLoader(dataset=data,batch_size=64)

print(data.X[0:5])
print(data.X.shape)
print(data.Y[0:5])
print(data.Y.shape)

input_dim = 4
hidden_dim = 25
output_dim = 3


class Net(nn.Module):
    def __init__(self, input, H, output):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input, H)
        self.linear2 = nn.Linear(H, output)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x

clf=Net(input_dim,hidden_dim,output_dim)

print(clf.parameters)