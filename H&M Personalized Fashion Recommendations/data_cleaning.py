import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_data(filename):
    data = pd.read_csv(f"data/{filename}.csv")
    return data


def clean_customer_data(data):
    # Deal with NaN
    data['FN'] = data['FN'].fillna(0)
    data['FN'] = data['FN'].astype('uint8')

    data['Active'] = data['Active'].fillna(0)
    data['Active'] = data['Active'].astype('uint8')

    data['age'] = data['age'].fillna(-1)
    data['age'] = data['age'].astype('uint8')

    data['fashion_news_frequency'] = data['fashion_news_frequency'].fillna("None")

    data['club_member_status'] = data['club_member_status'].fillna(0)
    # One Hot Encode
    data['fashion_news_frequency'].mask(data['fashion_news_frequency'] == 'NONE', 'None', inplace=True)
    data = pd.concat([data, pd.get_dummies(data["fashion_news_frequency"], prefix='news_frequency')], axis=1)
    data['club_member_status'].mask(data['club_member_status'] == 'LEFT CLUB', 'left', inplace=True)
    data['club_member_status'].mask(data['club_member_status'] == 'PRE-CREATE', 'pre_create', inplace=True)
    data = pd.concat([data, pd.get_dummies(data["club_member_status"], prefix='status')], axis=1)
    # Drop some columns
    data.drop(["postal_code", "fashion_news_frequency", "club_member_status"], axis=1, inplace=True)
    data = change_data_types_to_save_memory(data)
    return data


def clean_article_data(data):
    data.drop(["detail_desc"], axis=1, inplace=True)

    data['article_id'] = data['article_id'].astype('string')

    data = change_data_types_to_save_memory(data)
    return data


def clean_transaction_data(data):
    data = data.rename(columns={'t_dat': 'date'})
    data["date"] = pd.to_datetime(data["date"])

    data['customer_id'] = data['customer_id'].astype('string')
    data['article_id'] = data['article_id'].astype('string')

    data['sales_channel_id'] = data['sales_channel_id'].astype('uint8')

    data = change_data_types_to_save_memory(data)
    return data


def change_data_types_to_save_memory(data):
    for col in data.select_dtypes(include=['int64']).columns:
        data[col] = data[col].astype('int32')
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category')
    return data


def add_num_sold_to_articles(transactions_df, articles_df):
    num_sold = transactions_df.groupby(['article_id']).size().reset_index(name='num_sold')
    articles_df = articles_df.merge(num_sold, how='left', on='article_id')
    articles_df['num_sold'] = articles_df['num_sold'].fillna(0)
    articles_df['num_sold'] = articles_df['num_sold'].astype('int32')
    return articles_df


def remove_old_articles(transactions_df, articles_df):
    # Get a list of only article ids that have sold recently
    last_transaction_cutoff = "2019-09-01"
    active_articles = transactions_df.groupby("article_id")["date"].max().reset_index()
    active_articles = active_articles[active_articles["date"] >= last_transaction_cutoff].reset_index()
    # Prune inactive articles from transactions and articles
    transactions_df = transactions_df[transactions_df["article_id"].isin(active_articles["article_id"])].reset_index(
        drop=True)
    articles_df = articles_df[articles_df["article_id"].isin(active_articles["article_id"])].reset_index(drop=True)
    return transactions_df, articles_df


def print_data_info():
    print(customers.head())
    print(customers.info())
    print(customers.describe())
    print(articles.head())
    print(articles.info())
    print(articles.describe())
    print(transactions.head())
    print(transactions.info())
    print(transactions.describe())
    for i in transactions.columns:
        print(i, len(transactions[i].unique()))


def view_articles_by_number_sold():
    f, ax = plt.subplots(figsize=(10, 5))
    ax = sns.histplot(data=articles, y='index_name', color='orange')
    ax.set_xlabel('count by index name')
    ax.set_ylabel('index name')
    plt.show()


def view_articles_by_number_sold_detailed():
    f, ax = plt.subplots(figsize=(10, 5))
    ax = sns.histplot(data=articles, y='garment_group_name', color='orange', hue='index_group_name', multiple="stack")
    ax.set_xlabel('count by garment group')
    ax.set_ylabel('garment group')
    plt.show()


def generate_correlation(df):
    plt.figure(figsize=[7, 5])
    sns.heatmap(df.corr())
    plt.show()


if __name__ == "__main__":
    customers = load_data("customers")
    customers = clean_customer_data(customers)

    articles = load_data("articles")
    articles = clean_article_data(articles)

    transactions = load_data("transactions_train")
    transactions = clean_transaction_data(transactions)

    # articles = add_num_sold_to_articles(transactions, articles)

    # Trim the transactions and articles down to only active items
    transactions, articles = remove_old_articles(transactions_df=transactions, articles_df=articles)

    # view_articles_by_number_sold()
    # view_articles_by_number_sold_detailed()
    print_data_info()
    pass
