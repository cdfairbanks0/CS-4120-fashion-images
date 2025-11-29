from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import tensorflow as tf
from data import loadData
from utils import logConfusionMatrix
from train_baselines import trainLogisticRegression, trainDecisionTreeClassifier, trainDecisionTreeRegressor, trainLinearRegression
from train_nn import trainClassificationCNN, trainRegressionCNN
from features import getClassificationCNNData, getRegressionCNNData


# Test and prediction sets from training logistic regression model
X_train, X_validate, y_validate, y_pred, logreg_model = trainLogisticRegression()

# Class labels for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Classification report (precision, recall, f1-score per class) for logistic regression
report = classification_report(y_validate, y_pred, target_names=class_names)
print(report)


def runPlot3ConfusionLogreg():
    logConfusionMatrix(
        y_validate,
        y_pred,
        class_names,
        title="Plot 3 – Confusion Matrix (Logistic Regression, Test Set)"
    )
    mlflow.end_run()

runPlot3ConfusionLogreg()


result = permutation_importance(
    logreg_model,
    X_validate,
    y_validate,
    n_repeats=3,
    random_state=42
)

# Plot results
importances = result.importances_mean
importance_map = importances.reshape(28, 28)
#features = range(len(importances))  # logistic regression uses flattened pixels as features

plt.figure(figsize=(6,5))
#plt.bar(features, importances)
plt.imshow(importance_map, cmap='hot')
plt.colorbar(label="Mean decrease in accuracy")
plt.title("Plot 5 – Permutation Importance Heatmap (Logistic Regression)")
plt.xlabel("Image width")
plt.ylabel("Image height")
plt.show()



model = tf.keras.models.load_model("models/cnn_classification_model.keras")
X_train, X_val, x_test, y_train, CNN_y_val, CNN_y_test = getClassificationCNNData()

CNN_y_pred = model.predict(X_val).argmax(axis=1)
CNN_y_test_pred = model.predict(x_test).argmax(axis=1)
#CNN_y_pred, CNN_y_val, CNN_y_test_pred, CNN_y_test = trainClassificationCNN()

report = classification_report(CNN_y_val, CNN_y_pred, target_names=class_names)
print("classification report CNN classification validation set")
print(report)

report = classification_report(CNN_y_test, CNN_y_test_pred, target_names=class_names)
print("classification report CNN classification test set")
print(report)






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



def plot_cnn_learning_curve_for_latest_run_classification():
    client = MlflowClient()
    exp = client.get_experiment_by_name("MLflow CNN Classification Tracking")
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

    if not train_hist or not val_hist:
        print("Missing accuracy metrics for this run.")
        print("Make sure you're logging train_accuracy and val_accuracy in your CNN training function.")
        return

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
    exp = client.get_experiment_by_name("MLflow CNN Regression Tracking")
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