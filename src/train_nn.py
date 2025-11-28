import tensorflow as tf
import numpy as np
import mlflow
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, f1_score
from features import getCNNData


X_train, X_val, x_test, y_train, y_val, y_test = getCNNData()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", x_test.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

model = models.Sequential()

model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10)) #output layer

model.summary()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

mlflow.set_experiment("MLflow Classification Tracking")
with mlflow.start_run(run_name="cnn_classification"):
    mlflow.log_param("model_type", "CNN")
    mlflow.log_param("conv1_filters", 28)
    mlflow.log_param("conv2_filters", 64)
    mlflow.log_param("conv3_filters", 64)
    mlflow.log_param("dense_units", 64)
    mlflow.log_param("epochs", 15)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("optimizer", "adam")

    history = model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=1
    )

    for epoch, (acc, val_acc) in enumerate(zip(history.history["accuracy"],
                                            history.history["val_accuracy"])):
        mlflow.log_metric("train_accuracy", acc, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

    y_pred = model.predict(X_val).argmax(axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    print("Validation set accuracy for CNN:", accuracy)
    print("Validation set f1 for CNN:", f1)
    mlflow.log_metric("val_accuracy_final", float(accuracy))
    mlflow.log_metric("val_f1", float(f1))

    y_test_pred = model.predict(x_test).argmax(axis=1)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    print("Test set accuracy for CNN:", test_accuracy)
    print("Test set f1 for CNN:", test_f1)
    mlflow.log_metric("test_accuracy", float(test_accuracy))
    mlflow.log_metric("test_f1", float(test_f1))