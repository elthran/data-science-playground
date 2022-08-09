import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from data_cleaning import customers, articles, transactions, generate_correlation

transactions = transactions.groupby('date')['price'].sum().reset_index()

print(transactions.head())

transactions.set_index('date', inplace=True)

print(transactions.head())

results = seasonal_decompose(transactions['price'])
results.observed.plot(figsize=(12, 6))
plt.show()

results.trend.plot(figsize=(12, 6))
plt.show()

results.seasonal.plot(figsize=(12, 6))
plt.show()

results.resid.plot(figsize=(12, 6))
plt.show()

data = transactions[["price"]]
split_time = 600
x_train = data[:split_time]
x_valid = data[split_time:]

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_valid)

window_size = 60
n_features = 1
batch_size = 100
generator = TimeseriesGenerator(scaled_train, scaled_train, length=window_size, batch_size=batch_size)

X, y = generator[0]

callbacks = [TensorBoard(log_dir="logs"), EarlyStopping(patience=10, monitor="mae", mode="min")]

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                           strides=1, padding="causal",
                           activation="relu",
                           input_shape=(window_size, n_features)),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
])

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["mse"])

epochs = 500
history = model.fit(generator, epochs=epochs, callbacks=callbacks)

eval_batch = scaled_train[-window_size:]

eval_batch = eval_batch.reshape((1, window_size, n_features))

predict_batch = model.predict(eval_batch)

test_predictions = []

first_eval_batch = scaled_train[-window_size:]
current_batch = first_eval_batch.reshape((1, window_size, n_features))

current_pred = model.predict(current_batch)[0]

test_predictions.append(current_pred)

current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

loss_per_epoch = history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.show()

test_predictions = []

first_eval_batch = scaled_train[-window_size:]
current_batch = first_eval_batch.reshape((1, window_size, n_features))

for i in range(len(x_valid)):
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    # store prediction
    test_predictions.append(current_pred)
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

true_predictions = scaler.inverse_transform(test_predictions)

x_valid['Predictions'] = true_predictions

x_valid.plot(figsize=(10, 6))
plt.show()

print(tf.keras.metrics.mean_absolute_error(x_valid["price"], x_valid["Predictions"]).numpy())

print(x_valid)
print(x_valid.head(25))

x_valid.to_csv("submissions/keras-submission.csv", index=False, columns=["customer_id", "prediction"])
