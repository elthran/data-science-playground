from random import randint

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Build the dataframe for train data
# train = pd.read_csv('data/train.csv', encoding='utf-8')
# test = pd.read_csv('data/test.csv', encoding='utf-8')

data = pd.read_csv('data/train.csv', index_col="Id")
data = data[["MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
             "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
             "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
             "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt",
             "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
             "PoolArea", "YrSold", "SalePrice"]]
data = data.dropna()
train_labels = data["SalePrice"]
train_labels = np.stack(train_labels, axis=0)
train_features = data.drop("SalePrice", axis=1)
train_features = train_features.to_numpy()

test_labels = train_labels[-10:]
test_features = train_features[-10:]

train_labels = train_labels[:-10]
train_features = train_features[:-10]

print("Training data:", train_features.shape)
print("Testing data:", test_features.shape)

# (train_features, train_labels), (test_features, test_labels) = keras.datasets.boston_housing.load_data()

# train_features = data_features
# train_labels = data_labels

# get per-feature statistics (mean, standard deviation) from the training set to normalize by
train_mean = np.mean(train_features, axis=0)
train_std = np.std(train_features, axis=0)
train_features = (train_features - train_mean) / train_std


def build_model():
    model = keras.Sequential(
        [keras.layers.Dense(20, activation=tf.nn.relu, input_shape=[len(train_features[0])]), keras.layers.Dense(1)])

    model.compile(optimizer=tf.optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
    return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
history = model.fit(train_features, train_labels, epochs=1000, verbose=0, validation_split=0.1,
                    callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

print(hist)
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
    plt.ylim([0, 5000000])
    plt.show()


plot_history()

test_features_norm = (test_features - train_mean) / train_std
mse, _, _ = model.evaluate(test_features_norm, test_labels)
rmse = np.sqrt(mse)
print('Root Mean Square Error on test set: {}'.format(round(rmse, 3)))

for i in range(0, 10):
    # random_index = randint(0, 10)
    prediction = model.predict(test_features_norm[[i]])
    print("expected:", test_labels[[i]], "prediction:", prediction)
