from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_df = pd.read_csv(r"data/train.csv", index_col="PassengerId")
test_df = pd.read_csv(r"data/test.csv", index_col="PassengerId")
df = pd.concat([train_df, test_df], sort=False)


def nan_padding(df, columns):
    """Uses knn to fill in N/A columns"""
    for column in columns:
        imputer = KNNImputer(n_neighbors=2)
        df[column] = imputer.fit_transform(df[column].values.reshape(-1, 1))
        df[columns] = df[columns].apply(pd.to_numeric)
    return df


nan_columns = ["Age", "SibSp", "Parch"]
df = nan_padding(df, nan_columns)

df.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1, inplace=True)
df['Sex'] = np.where(df['Sex'] == 'female', 0, 1)


def dummy_data(df, columns):
    for column in columns:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df = df.drop(column, axis=1)
    return df


dummy_columns = ["Pclass"]
df = dummy_data(df, dummy_columns)


def normalize_age(df):
    scaler = MinMaxScaler()
    df["Age"] = scaler.fit_transform(df["Age"].values.reshape(-1, 1))
    df["Fare"] = scaler.fit_transform(df["Fare"].values.reshape(-1, 1))
    return df


df = normalize_age(df)

train_df = df[~df["Survived"].isnull()]
test_df = df[df["Survived"].isnull()]
test_df = test_df.drop(["Survived"], axis=1)


print(train_df.head(5))
print(test_df.head(5))


def split_valid_test_data(df, fraction=(1 - 0.8)):

    y = df["Survived"]
    X = df.drop(["Survived"], axis=1)

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=fraction)

    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = split_valid_test_data(train_df)
print("train_x:{}".format(train_x.shape))
print("train_y:{}".format(train_y.shape))
print("test_x:{}".format(test_x.shape))
print("test_y:{}".format(test_y.shape))


model = tf.keras.Sequential()

def tf_it(test_df):
    # compile the model
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')

    # fit the model
    model.fit(train_x, train_y, epochs=100, batch_size=32)

    mse, mae = model.evaluate(test_x, test_y, verbose=0)
    print(f'MSE: {mse:.3f}, RMSE: {sqrt(mse):.3f}, MAE: {mae:.3f}')
    predict_test = model.predict(test_x)

    print("tst_y", test_y)
    print("predict_test", predict_test[0])

    def plot_horsepower():
        plt.scatter(train_x["Sex"], train_y, label='Data')
        plt.plot(test_y, predict_test, color='k', label='Predictions')
        plt.xlabel('Sex')
        plt.ylabel('Survivors')
        plt.legend()
        plt.show()

    plot_horsepower()

    prediction = model.predict(test_df)

    return prediction

prediction = tf_it(test_df)
