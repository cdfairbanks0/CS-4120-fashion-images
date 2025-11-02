from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from data import loadData
from utils import logConfusionMatrix
from train_baselines import trainLogisticRegression, trainDecisionTreeClassifier, trainDecisionTreeRegressor, trainLinearRegression

# Test and prediction sets from training logistic regression model
y_test, y_pred = trainLogisticRegression()

# Class labels for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Classification report (precision, recall, f1-score per class) for logistic regression
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

# Test and prediction sets from training decision tree model
y_DTtest, y_DTpred = trainDecisionTreeClassifier()

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

# MAE and RMSE for decision tree regressor
y_validate, y_pred = trainDecisionTreeRegressor()

mae = mean_absolute_error(y_validate, y_pred)
rmse = np.sqrt(mean_squared_error(y_validate, y_pred))

print(f"Mean Absolute Error for decision tree: {mae:.4f}")
print(f"Root Mean Squared Error for decision tree: {rmse:.4f}")

residuals = y_validate - y_pred

plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel("Predicted Brightness")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()




# MAE and RMSE for linear regression model
y_validate, y_pred = trainLinearRegression()

mae = mean_absolute_error(y_validate, y_pred)
rmse = np.sqrt(mean_squared_error(y_validate, y_pred))

print(f"Mean Absolute Error for linear regression: {mae:.4f}")
print(f"Root Mean Squared Error for linear regression: {rmse:.4f}")

residuals = y_validate - y_pred

plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel("Predicted Brightness")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()