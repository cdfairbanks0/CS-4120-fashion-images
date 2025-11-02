from data import loadData, splitTrainVal, convertDatasetToArray
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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
    #labels = []
    means = []
    images = []
    for image, label in train_data:
        arr = np.array(image)
        means.append(arr.mean())
        #labels.append(label)
        images.append(np.array(image).flatten())

    X = np.array(images)
    y = np.array(means)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


getRegressionData()