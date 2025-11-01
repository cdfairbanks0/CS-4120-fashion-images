from data import loadData, splitTrainVal, convertDatasetToArray

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

