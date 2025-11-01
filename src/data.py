import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST

def loadData(root='data'):
    trainData = FashionMNIST(root=root, train=True, download=True)
    testData = FashionMNIST(root=root, train=False, download=True)
    return trainData, testData

def splitTrainVal(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_val, y_train, y_val

def convertDatasetToArray(dataset):
    X = []
    y = []
    for img, label in dataset:
        X.append(np.array(img).flatten()) #flatten image data
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y