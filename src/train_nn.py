import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from features import getCNNData


X_train, X_val, y_train, y_val = getCNNData()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

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

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)

y_pred = model.predict(X_val).argmax(axis=1)

accuracy = accuracy_score(y_val, y_pred)

print("Validation set accuracy for CNN:", accuracy)