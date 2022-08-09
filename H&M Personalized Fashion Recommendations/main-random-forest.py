import gc

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from data_cleaning import load_data, clean_transaction_data, clean_customer_data, clean_article_data

prediction_date = "2020-09-23"
show_all = True
fresh_start = True


def create_clean_customer_data():
    """
    Merges customers with transactions and then groups by customer_id.
    Create two clean datasets:
        customers.csv: 1 row per user with all data known on first transaction
        all_customers.csv: simply 1 column with 1 row per username
    """
    customers = load_data("customers")
    customers = clean_customer_data(customers)
    initial_row_count = len(customers.index)
    transactions = load_data("transactions_train")
    transactions = clean_transaction_data(transactions)
    transactions = pd.merge(customers, transactions, on='customer_id', how='left')
    # Reduce transactions to first row per user
    transactions = transactions.groupby(by=['customer_id']).agg(
        first_transaction=pd.NamedAgg(column='date', aggfunc='min')
    ).reset_index()
    # Merge the customer data with grouped data
    customers = pd.merge(customers, transactions, on='customer_id', how='left')
    customers.to_csv("clean_data/customers.csv", index=False)
    all_users = pd.DataFrame({'customer_id': customers["customer_id"].unique(),
                              'date': pd.to_datetime(prediction_date)})
    all_users.to_csv("clean_data/customer_list.csv", index=False)
    assert initial_row_count == len(customers.index), f"Only {len(customers.index)} of {initial_row_count} users"
    assert initial_row_count == len(all_users.index), f"Only {len(customers.index)} of {initial_row_count} users"
    print(f"\nOne row per customer from create_clean_customer_data:\n {customers.head(5)}")


def create_clean_transaction_data():
    """
    Merges articles with transactions to generate one row per transaction with article data.
    Create a dataset that is one row per transaction including the future transactions.
    """
    articles = load_data("articles")
    articles = clean_article_data(articles)
    # Batch read the transactions, append it to the article data and write out it.
    batch_reader = pd.read_csv("data/transactions_train.csv", chunksize=1000000)
    transactions = None
    for index, batch in enumerate(batch_reader):
        batch = clean_transaction_data(batch)
        merged_batch = pd.merge(articles, batch, on='article_id', how='inner')
        if index == 0:
            transactions = merged_batch
        else:
            transactions = pd.concat([transactions, merged_batch])
        print(f"batch {index}")
    print("df fully loaded and ready")
    # Add a dummy transaction for each user with NaN for the prediction rows so we can merge
    all_users = pd.read_csv("clean_data/customer_list.csv")
    all_columns = transactions.columns
    for column in all_columns:
        if column not in ['customer_id', "date"]:
            all_users[column] = None
    # Merge prediction data
    print("about to concat")
    transactions_with_dummy_rows = pd.concat([transactions, all_users])
    # Write this out
    print("about to write out")
    transactions_with_dummy_rows.to_csv("clean_data/transactions.csv", index=False)
    print(f"\nTransactions from create_clean_transaction_data:\n {transactions_with_dummy_rows.head(5)}")


def create_historical_data_per_day():
    transactions = pd.read_csv("clean_data/transactions.csv")
    # Group by Customer and Date
    user_day_df = transactions.groupby(by=['customer_id', 'date']) \
        .agg(
        daily_revenue=pd.NamedAgg(column='price', aggfunc='sum'),
        daily_transactions=pd.NamedAgg(column='article_id', aggfunc='count')
    ) \
        .reset_index()
    # Create some historical columns and remove current transaction info
    user_day_df['lt_transactions'] = user_day_df.groupby(['customer_id'])['daily_transactions'].cumsum().sub(
        user_day_df.daily_transactions)
    user_day_df['lt_revenue'] = user_day_df.groupby(['customer_id'])['daily_revenue'].cumsum().sub(
        user_day_df.daily_revenue)
    user_day_df.drop(["daily_transactions", "daily_revenue"], axis=1, inplace=True)
    # Get the date difference between this transaction and the previous
    user_day_df["last_transaction"] = user_day_df.sort_values(['customer_id', 'date']).groupby(['customer_id'])[
        'date'].shift(1)
    del transactions
    gc.collect()
    # First join customer data onto our historical data
    customers = pd.read_csv("clean_data/customers.csv")
    user_day_df = pd.merge(user_day_df, customers, on=['customer_id'], how="inner")
    del customers
    gc.collect()
    # Drop the transactions to clear up memory. Then batch load them and merge.
    batched_transactions = pd.read_csv("clean_data/transactions.csv", chunksize=1000000)
    transactions = None
    for index, batch in enumerate(batched_transactions):
        batch = batch[["customer_id", "date", "article_id"]]
        merged_batch = pd.merge(user_day_df, batch, on=['customer_id', 'date'], how="inner")
        if index == 0:
            transactions = merged_batch
        else:
            transactions = pd.concat([transactions, merged_batch])
    print("df fully loaded and ready to write out")
    transactions.to_csv("clean_data/transactions_with_historical.csv", index=False)
    print(f"\nTransactions from create_historical_data_per_day:\n {transactions.head(5)}")


def create_final_df():
    # Remove rows with seldom bought articles
    # article_frequency = article_trans_data.groupby(by=["article_id"]) \
    #     .agg(bought_count=pd.NamedAgg(column='customer_id', aggfunc='count')) \
    #     .reset_index()
    # del article_trans_data
    # article_frequency = article_frequency[article_frequency["bought_count"] < 100]
    # df = df[df["article_id"].isin(article_frequency["article_id"])].reset_index(drop=True)
    # del article_frequency
    return df


if fresh_start:
    # create_clean_customer_data()
    create_clean_transaction_data()
    create_historical_data_per_day()

df = pd.read_csv("clean_data/transactions_with_historical.csv")
print(f"\nLoaded our df:\n {df.sample(50)}")

# Add RF time series data
df['weeks_since_first_trans'] = (df["date"] - df["first_transaction"]).dt.days // 7
df['weeks_since_first_trans'] = df['weeks_since_first_trans'].fillna(0)
df['weeks_since_first_trans'] = df['weeks_since_first_trans'].astype('uint8')
df['weeks_since_last_trans'] = (df["date"] - df["last_transaction"]).dt.days // 7
df['weeks_since_last_trans'] = df['weeks_since_last_trans'].fillna(0)
df['weeks_since_last_trans'] = df['weeks_since_last_trans'].astype('uint8')
df["year"] = df["date"].dt.year
# Create a month one-hot encoding
df["month"] = df["date"].dt.month
df = pd.concat([df, pd.get_dummies(df["month"], prefix='month')], axis=1)
df.drop(["first_transaction", "last_transaction", "month"], axis=1, inplace=True)

# For Random Forest, Encode all categorical columns
ordinal_encoder = OrdinalEncoder()
object_cols = df.select_dtypes(include=['category']).columns
df[object_cols] = ordinal_encoder.fit_transform(df[object_cols])

# Separate the final prediction rows from our training data
holdout_data = df[df["date"] == prediction_date]
holdout_data = holdout_data.drop(["date", "article_id"], axis=1)
df = df[df["date"] != prediction_date]

# LabelEncode the article_ids since that is what we are predicting
label_encoded_article_ids = LabelEncoder()
label_encoded_article_ids.fit(df["article_id"])
df["article_id"] = label_encoded_article_ids.transform(df["article_id"])

# Separate Features and targets
y = df['article_id']
X = df.drop(["date", "article_id", "customer_id"], axis=1)

# Split the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

# Train the model
rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)

if show_all:
    # The permutation based importance
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.tight_layout()
    plt.show()
    # The permutation based importance
    perm_importance = permutation_importance(rf, X_test, y_test)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.tight_layout()
    plt.show()
    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X.columns[i] for i in indices]
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation=20, fontsize=8)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

# Generate our predictions
holdout_indices = holdout_data['customer_id'].tolist()
holdout_data.drop(["customer_id"], axis=1, inplace=True)
# Get the probabilities for each article
prediction_probabilities = rf.predict_proba(holdout_data)
# Extract the top 12 prediction
top_12_predictions = np.argsort(prediction_probabilities)[:, :-12 - 1:-1]
# Decode each array of predictions and turn it into a string
top_n_predictions_decoded = []
for array in top_12_predictions:
    decoded_predictions = label_encoded_article_ids.inverse_transform(array)
    top_n_predictions_decoded.append(' '.join(decoded_predictions))
# Create a dataframe with the customer_ids and predictions
top_n_predictions_df = pd.DataFrame({'customer_id': holdout_indices,
                                     'prediction': top_n_predictions_decoded})
print("\n\nFinal Predictions:\n", top_n_predictions_df.head(15))
# Write it out to a csv
top_n_predictions_df.to_csv("submissions/random-forest-submission.csv", index=False)
