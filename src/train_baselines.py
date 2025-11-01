import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
from mlflow.models import infer_signature


from features import getTrainValidateSplits, getScaledFeaturesLogReg


def trainLogisticRegression():
    mlflow.set_experiment("MLflow Classification Tracking")
    
    X_train, X_validate, y_train, y_validate = getTrainValidateSplits()
    X_train_scaled, X_validate_scaled = getScaledFeaturesLogReg(X_train, X_validate)

    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state = 42)

    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", "1000")
        mlflow.log_param("solver", "lbfgs")
        mlflow.log_param("random_state", "42")

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_validate_scaled)

        signature = infer_signature(X_validate_scaled, y_pred)

        accuracy = accuracy_score(y_validate, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        print("Validation set accuracy for Log Reg:", accuracy)

        #mlflow.sklearn.log_model(model, "Logistic Regression Model")
        mlflow.sklearn.log_model(
            sk_model=model,
            signature=signature,
            name="fashion_mnist_lr_model",
            input_example=X_train_scaled
        )
    return y_validate, y_pred

def trainDecisionTree():
    X_train, X_validate, y_train, y_validate = getTrainValidateSplits()
    model = DecisionTreeClassifier(max_depth = 15, random_state = 42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_validate)
    accuracy = accuracy_score(y_validate, y_pred)
    print("Validation set accuracy for DT:", accuracy)
    return y_validate, y_pred

#trainLogisticRegression()
#trainDecisionTree()