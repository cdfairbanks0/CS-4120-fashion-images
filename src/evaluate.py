from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from data import loadData
from utils import logConfusionMatrix
from train_baselines import trainLogisticRegression, trainDecisionTreeClassifier, trainDecisionTreeRegressor, trainLinearRegression
"""
# Test and prediction sets from training logistic regression model
y_test, y_pred = trainLogisticRegression()

# Class labels for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Classification report (precision, recall, f1-score per class) for logistic regression
report = classification_report(y_test, y_pred, target_names=class_names)
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

# Test and prediction sets from training decision tree model
y_DTtest, y_DTpred = trainDecisionTreeClassifier()

# Classification report for decision tree
report = classification_report(y_DTtest, y_DTpred, target_names=class_names)
print(report)

mlflow.end_run()

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

mlflow.end_run()



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

mlflow.end_run()
"""
def plot_cnn_learning_curve_for_latest_run_classification():
    client = MlflowClient()
    exp = client.get_experiment_by_name("MLflow Classification Tracking")
    if exp is None:
        print("Experiment 'MLflow Classification Tracking' not found")
        return
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if len(runs) == 0:
        print("No runs found in experiment 'MLflow Classification Tracking'.")
        return
    run_id = runs[0].info.run_id
    print("Using CNN Run Id:", run_id)
    train_hist = client.get_metric_history(run_id, "train_accuracy")
    val_hist = client.get_metric_history(run_id, "val_accuracy")
    train_values = [m.value for m in train_hist]
    val_values = [m.value for m in val_hist]
    epochs = range(1, len(train_values) + 1)
    plt.figure()
    plt.plot(epochs, train_values, label="Train accuracy")
    plt.plot(epochs, val_values, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Plot 1 – Classification NN Learning Curve")
    plt.legend()
    plt.show()


def plot_cnn_learning_curve_for_latest_run_regression():
    client = MlflowClient()
    exp = client.get_experiment_by_name("MLflow Regression Tracking")
    if exp is None:
        print("Experiment 'MLflow Regression Tracking' not found")
        return
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if len(runs) == 0:
        print("No runs found in experiment 'MLflow Regression Tracking'.")
        return
    run_id = runs[0].info.run_id
    print("Using CNN Run Id:", run_id)
    train_hist = client.get_metric_history(run_id, "train_mae")
    val_hist = client.get_metric_history(run_id, "val_mae")
    train_values = [m.value for m in train_hist]
    val_values = [m.value for m in val_hist]
    epochs = range(1, len(train_values) + 1)
    plt.figure()
    plt.plot(epochs, train_values, label="Train mae")
    plt.plot(epochs, val_values, label="Validation mae")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Plot 1 – Regression NN Learning Curve")
    plt.legend()
    plt.show()

plot_cnn_learning_curve_for_latest_run_classification()
plot_cnn_learning_curve_for_latest_run_regression()