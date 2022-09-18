import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_data = pd.read_csv("data/train_data.csv", nrows=350000, index_col="customer_ID")
train_data["is_training"] = True
test_data = pd.read_csv("data/test_data.csv", nrows=350000, index_col="customer_ID")
train_data["is_training"] = False
data = pd.concat([train_data, test_data])
# Extract the date column
data["S_2"] = pd.to_datetime(data["S_2"], format='%Y-%m-%d')
# Current model doesn't support timestamps so convert to integers
data["S_2"] = (data["S_2"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1d')
# Organize columns
numerical_columns = [column_name for column_name, column_type in dict(data.dtypes).items() if
                     column_type in ("float64", "uint8")]
categorical_columns = [column_name for column_name, column_type in dict(data.dtypes).items() if
                       column_type not in ("float64", "uint8") and column_name != "customer_ID"]

# Use an Imputer to fill in missing values
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imputer = mean_imputer.fit(data[numerical_columns])
data[numerical_columns] = mean_imputer.transform(data[numerical_columns].values)
# Aggregate to 1 row per customer
df_train_num_agg = data.groupby("customer_ID")[numerical_columns].agg(['mean', 'std', 'min', 'max', 'last'])
df_train_num_agg.columns = ['_'.join(x) for x in df_train_num_agg.columns]
df_train_cat_agg = data.groupby("customer_ID")[categorical_columns].agg(['count', 'nunique'])  # 'first', 'last',
df_train_cat_agg.columns = ['_'.join(x) for x in df_train_cat_agg.columns]
data = pd.concat([df_train_num_agg, df_train_cat_agg], axis=1)
# Remove null values that couldn't be imputed from training
data = data.fillna(data.mean())
# Normalize the data
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(data)
data = pd.DataFrame(min_max_scaler.transform(data), columns=data.columns)
# Load the target values. Merge with training data and set Customer_ID as Index
targets = pd.read_csv("data/train_labels.csv")
data = data.merge(targets, left_index=True, right_index=True, how='left')
del targets
data = data.set_index('customer_ID')

print(data.sample(100))
# raise
print("data:", data.shape)
# Remove null values that couldn't be joined correctly
data = data.dropna()

train_labels = data["target"]
train_labels = np.stack(train_labels, axis=0)
train_features = data.drop("target", axis=1)
train_features = train_features.to_numpy()

test_labels = train_labels[-10:]
test_features = train_features[-10:]

train_labels = train_labels[:-10]
train_features = train_features[:-10]

print("Training data:", train_features.shape)
print("Testing data:", test_features.shape)


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(10, activation=tf.nn.relu, input_shape=[train_features.shape[1]]),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
    return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


model = build_model()

train_features = np.asarray(train_features).astype('float32')
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
history = model.fit(train_features, train_labels, epochs=10000, verbose=0, validation_split=0.1,
                    callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

print("\n", hist)
# show RMSE measure to compare to Kaggle leaderboard on https://www.kaggle.com/c/boston-housing/leaderboard
rmse_final = np.sqrt(float(hist['val_mse'].tail(1)))
print()
print('Final Root Mean Square Error on validation set: {}'.format(round(rmse_final, 3)))


def plot_history():
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [Thousand Dollars$^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.legend()
    plt.show()


plot_history()

# test_features_norm = (test_features - train_mean) / train_std
mse, _, _ = model.evaluate(test_features, test_labels)
rmse = np.sqrt(mse)
print('Root Mean Square Error on test set: {}'.format(round(rmse, 3)))

for i in range(0, 10):
    # random_index = randint(0, 10)
    prediction = model.predict(test_features[[i]])
    print(f"expected: {test_labels[[i]][0]} vs {int(round(prediction[0][0], 0))}, prediction: {prediction[0][0]}")
