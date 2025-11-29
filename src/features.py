from data import loadData, splitTrainVal, convertDatasetToArray, convertDatasetToArrayCNN
from sklearn.model_selection import train_test_split
import numpy as np

train_data, test_data = loadData()

X_train_initial, y_train_initial = convertDatasetToArray(train_data)
X_test, y_test = convertDatasetToArray(test_data)

X_train, X_val, y_train, y_val = splitTrainVal(X_train_initial, y_train_initial)

def getScaledFeaturesLogReg(X_train_in, X_eval_in):
    X_train_scaled = X_train_in / 255.0 # normalize pixel brightness from 0-255 to 0-1 scale
    X_eval_scaled = X_eval_in / 255.0
    return X_train_scaled, X_eval_scaled

def getTrainValidateSplits():
    return X_train, X_val, y_train, y_val

def getRegressionData():
    means = []
    images = []
    for image, label in train_data:
        arr = np.array(image)
        means.append(arr.mean())
        images.append(np.array(image).flatten())

    X = np.array(images)
    y = np.array(means)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


""" def getCNNData():
    X, y = convertDatasetToArrayCNN(train_data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)
    return X_train, X_val, y_train, y_val """

def getClassificationCNNData():
    X, y = convertDatasetToArrayCNN(train_data)
    X_test, y_test = convertDatasetToArrayCNN(test_data)

    #y = np.array([img.mean() for img in X])
    #y_test = np.array([img.mean() for img in X_test])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)


    #X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)

    return X_train, X_val, X_test, y_train, y_val, y_test


def getRegressionCNNData():
    X, y = convertDatasetToArrayCNN(train_data)
    X_test, y_test = convertDatasetToArrayCNN(test_data)

    y = np.array([img.mean() for img in X])
    y_test = np.array([img.mean() for img in X_test])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)

    return X_train, X_val, X_test, y_train, y_val, y_test