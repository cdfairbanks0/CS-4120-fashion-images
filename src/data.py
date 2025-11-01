import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST

def loadData(root='data'):
    trainData = FashionMNIST(root=root, train=True, download=True)
    testData = FashionMNIST(root=root, train=False, download=True)
    return trainData, testData

def splitTrainVal(dataset, val_size=0.2, seed=42):
    labels = np.array(dataset.targets)
    labelArrange = np.arrange(len(labels))
    train_x, val_x = train_test_split(
        labelArrange, test_size=val_size, stratify=labels, random_state=seed
    )
    return train_x, val_x