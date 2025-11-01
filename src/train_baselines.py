import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from features import getTrainValidateSplits, getScaledFeaturesLogReg


def trainLogisticRegression():
    X_train, X_validate, y_train, y_validate = getTrainValidateSplits()
    X_train_scaled, X_validate_scaled = getScaledFeaturesLogReg(X_train, X_validate)

    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state = 42)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_validate_scaled)
    accuracy = accuracy_score(y_validate, y_pred)
    print("Validation set accuracy for Log Reg:", accuracy)
    return

def trainDecisionTree():
    X_train, X_validate, y_train, y_validate = getTrainValidateSplits()
    model = DecisionTreeClassifier(max_depth = 15, random_state = 42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_validate)
    accuracy = accuracy_score(y_validate, y_pred)
    print("Validation set accuracy for DT:", accuracy)
    return

trainLogisticRegression()
trainDecisionTree()