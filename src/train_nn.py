import tensorflow as tf
import keras_tuner
import numpy as np
import mlflow
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, f1_score
from features import getClassificationCNNData, getRegressionCNNData

def build_model(hyperparam):
    model = models.Sequential()

    # hyperparams
    hp_filter1 = hyperparam.Choice("filters_1", values=[32, 48])
    hp_dense = hyperparam.Choice("dense_units", values=[64, 128])
    hp_learning_rate = hyperparam.Choice("learning_rate", [0.001, 0.0005, 0.0001])

    regulizer = tf.keras.regularizers.l2(0.0003)

    model.add(layers.Conv2D(hp_filter1, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=regulizer))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regulizer))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regulizer))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regulizer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(hp_dense, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax')) #output

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def trainClassificationCNNHyperparameters():

    X_train, X_val, x_test, y_train, y_val, y_test = getClassificationCNNData()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    tuner = keras_tuner.GridSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=20,
        overwrite=True
    )
    

    tuner.search(
        X_train,
        y_train,
        epochs=35,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    best_hyperparams = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    y_pred = best_model.predict(X_val).argmax(axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")

    print("Validation set accuracy for CNN:", accuracy)
    print("Validation set f1 for CNN:", f1)
    


def trainClassificationCNN():
    X_train, X_val, x_test, y_train, y_val, y_test = getClassificationCNNData()

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", x_test.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("y_test shape:", y_test.shape)

    regulizer = tf.keras.regularizers.l2(0.0003)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True
    )

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=regulizer))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regulizer))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regulizer))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regulizer))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax')) #output layer

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    mlflow.set_experiment("MLflow Classification Tracking")
    with mlflow.start_run(run_name="cnn_classification"):

        mlflow.log_param("model_type", "CNN")
        mlflow.log_param("conv1_filters", 32)
        mlflow.log_param("conv1_filters", 32)
        mlflow.log_param("conv2_filters", 64)
        mlflow.log_param("conv3_filters", 64)
        mlflow.log_param("dense_units", 128)
        mlflow.log_param("regularization", 0.0003)
        mlflow.log_param("epochs", 15)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("learning rate", "0.001")
        mlflow.log_param("dropout_rates", "0.25, 0.25, 0.5")


        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=64,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
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


        # only used after we tuned our hyperparameters and network on the validation set
        
        y_test_pred = model.predict(x_test).argmax(axis=1)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average="weighted")
        print("Test set accuracy for CNN:", test_accuracy)
        print("Test set f1 for CNN:", test_f1)
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_metric("test_f1", float(test_f1))
    mlflow.end_run()

    model.save("models/cnn_classification_model.keras")

    return y_pred, y_val, y_test_pred, y_test
        


def trainRegressionCNN():
    X_train, X_val, x_test, y_train, y_val, y_test = getRegressionCNNData()

    model = models.Sequential()

    model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) #output layer

    model.summary()

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    mlflow.set_experiment("MLflow Regression Tracking")
    with mlflow.start_run(run_name="cnn_regression"):

        mlflow.log_param("model_type", "CNN")
        mlflow.log_param("conv1_filters", 28)
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

        for epoch in range(len(history.history["loss"])):
            mlflow.log_metric("train_mse", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("train_mae", history.history["mae"][epoch], step=epoch)
            mlflow.log_metric("val_mse", history.history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("val_mae", history.history["val_mae"][epoch], step=epoch)

        y_pred = model.predict(X_val).flatten()
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)

        print("Validation MAE:", mae)
        print("Validation MSE:", mse)

        mlflow.log_metric("val_mae_final", float(mae))
        mlflow.log_metric("val_mse_final", float(mse))
    mlflow.end_run()

    model.save("models/cnn_regression_model.keras")


#trainRegressionCNN()
#trainClassificationCNN()
#trainClassificationCNNHyperparameters()