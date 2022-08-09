import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# First load the training and test data
df_train = pd.read_csv("data/train_data.csv", nrows=5000, index_col="customer_ID")
df_test = pd.read_csv("data/test_data.csv", nrows=5000, index_col="customer_ID")

# Extract the date column
df_train["S_2"] = pd.to_datetime(df_train["S_2"], format='%Y-%m-%d')
df_test["S_2"] = pd.to_datetime(df_test["S_2"], format='%Y-%m-%d')

# Current model doesn't support timestamps so convert to integers
df_train["S_2"] = (df_train["S_2"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1d')
df_test["S_2"] = (df_test["S_2"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1d')

# Organize columns
numerical_columns = [column_name for column_name, column_type in dict(df_train.dtypes).items() if
                     column_type in ("float64", "uint8")]
categorical_columns = [column_name for column_name, column_type in dict(df_train.dtypes).items() if
                       column_type not in ("float64", "uint8") and column_name != "customer_ID"]

# Use an Imputer to fill in missing values
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imputer = mean_imputer.fit(df_train[numerical_columns])
df_train[numerical_columns] = mean_imputer.transform(df_train[numerical_columns].values)
df_test[numerical_columns] = mean_imputer.transform(df_test[numerical_columns].values)

# Aggregate to 1 row per customer
df_train_num_agg = df_train.groupby("customer_ID")[numerical_columns].agg(['mean', 'std', 'min', 'max', 'last'])
df_test_num_agg = df_test.groupby("customer_ID")[numerical_columns].agg(['mean', 'std', 'min', 'max', 'last'])
df_train_num_agg.columns = ['_'.join(x) for x in df_train_num_agg.columns]
df_test_num_agg.columns = ['_'.join(x) for x in df_test_num_agg.columns]
df_train_cat_agg = df_train.groupby("customer_ID")[categorical_columns].agg(['count', 'nunique'])  # 'first', 'last',
df_test_cat_agg = df_test.groupby("customer_ID")[categorical_columns].agg(['count', 'nunique'])
df_train_cat_agg.columns = ['_'.join(x) for x in df_train_cat_agg.columns]
df_test_cat_agg.columns = ['_'.join(x) for x in df_test_cat_agg.columns]
df_train = pd.concat([df_train_num_agg, df_train_cat_agg], axis=1)
df_test = pd.concat([df_train_num_agg, df_train_cat_agg], axis=1)

# Remove null values that couldn't be imputed from training
df_train = df_train.dropna()
# Since we cannot remove null values from test data, we will fill with the mean
df_test = df_train.fillna(df_test.mean())

# Normalize the data
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(df_train)
df_train = pd.DataFrame(min_max_scaler.transform(df_train), columns=df_train.columns)
df_test = pd.DataFrame(min_max_scaler.transform(df_test), columns=df_test.columns)

# Load the target values. Merge with training data and set Customer_ID as Index
targets = pd.read_csv("data/train_labels.csv", nrows=5000)
df_train = df_train.merge(targets, left_index=True, right_index=True, how='left')
del targets
df_train = df_train.set_index('customer_ID')

# Split the training data into X and y
x_train = df_train.drop(['target'], axis=1)
y_train = df_train['target']

x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x_train, y_train, test_size=0.25,
                                                                            random_state=0)

model = LogisticRegression(penalty='l2')
model.fit(x_train_split, y_train_split)

y_predict = model.predict(x_test_split)

print('\nLogistics Regression Accuracy: {:.3f}'.format(accuracy_score(y_test_split, y_predict)))
print('\nLogistics Regression Precision: {:.3f}'.format(precision_score(y_test_split, y_predict)))
print('\nLogistics Regression Recall: {:.3f}'.format(recall_score(y_test_split, y_predict)))

prediction = model.predict(df_test)
# Merge the prediction and customer_ID into submission dataframe
submission = pd.DataFrame({"customer_ID": df_test.index, "prediction": prediction})
submission.to_csv('submission.csv', index=False)
