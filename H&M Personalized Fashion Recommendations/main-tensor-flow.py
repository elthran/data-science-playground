import shap
from matplotlib import pyplot as plt

# plt.rcParams["figure.figsize"] = (25, 5)
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import pandas as pd
import numpy as np

from data_cleaning import load_data, clean_transaction_data, clean_customer_data, clean_article_data

# prepare the data
customers = load_data("customers")
customers = clean_customer_data(customers)
# articles = load_data("articles")
# articles = clean_article_data(articles)
transactions = load_data("transactions_train")
transactions = clean_transaction_data(transactions)
transactions_sampling = transactions.sample(n=100)
df = pd.merge(transactions_sampling, customers, on='customer_id')
# df = pd.merge(df, articles, on='article_id')

# For Random Forest, Encode all categorical columns
ordinal_encoder = OrdinalEncoder()
object_cols = df.select_dtypes(include=['category']).columns
df[object_cols] = ordinal_encoder.fit_transform(df[object_cols])

# LabelEncode the article_ids since that is what we are predicting
label_encoded_article_ids = LabelEncoder()
label_encoded_article_ids.fit(df["article_id"])
df["article_id"] = label_encoded_article_ids.transform(df["article_id"])

# label_encoded_customer_ids = LabelEncoder()
# label_encoded_customer_ids.fit(df["customer_id"])
# df["customer_id"] = label_encoded_customer_ids.transform(df["customer_id"])

y = df['article_id']
X = df.drop(["date", 'customer_id', "article_id"], axis=1)

print(X.head(5))
print(y.head(5))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

# train the model
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# the permutation based importance
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", plot_size=[12, 12])
plt.show()

perm_importance = permutation_importance(rf, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X.columns[i] for i in indices]
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=20, fontsize=8)
plt.title("Feature Importance")
plt.show()

