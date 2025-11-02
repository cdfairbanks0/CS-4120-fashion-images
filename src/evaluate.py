from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from data import loadData
from utils import logConfusionMatrix
from train_baselines import trainLogisticRegression, trainDecisionTree

# Test and prediction sets from training logistic regression model
y_test, y_pred = trainLogisticRegression()

# Class labels for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Classification report (precision, recall, f1-score per class) for logistic regression
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

# Test and prediction sets from training decision tree model
y_DTtest, y_DTpred = trainDecisionTree()

# Classification report for decision tree
report = classification_report(y_DTtest, y_DTpred, target_names=class_names)
print(report)

def runPlot3ConfusionLogreg():
    logConfusionMatrix(
        y_test,
        y_pred,
        class_names,
        title="Plot 3 – Confusion Matrix (Logistic Regression, Test Set)"
    )
    mlflow.end_run()

runPlot3ConfusionLogreg()