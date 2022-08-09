import gc

import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from wordcloud import WordCloud, STOPWORDS

pd.options.mode.chained_assignment = None
pio.renderers.default = 'notebook_connected'

from data_cleaning import load_data, clean_transaction_data, clean_article_data

# PREPARE THE DATA
# customers = load_data("customers")
# customers = clean_customer_data(customers)
articles = load_data("articles")
articles = clean_article_data(articles)
transactions = load_data("transactions_train")
transactions = clean_transaction_data(transactions)
# REDUCE TRANSACTIONS TO A SMALL SAMPLING
# transactions = transactions.sample(n=1000)
df = pd.merge(transactions, articles, on='article_id')
del transactions
del articles
gc.collect()
articles = df.groupby(by=["garment_group_no"]) \
    .agg(item_count=pd.NamedAgg(column='index_code', aggfunc='count')) \
    .reset_index()
print("garment_group_no:\n", articles["garment_group_no"].value_counts(dropna=False))
articles = pd.concat([articles,
                      pd.get_dummies(articles["garment_group_no"],
                                     prefix='garment_no')],
                     axis=1).drop(['garment_group_no'],
                                  axis=1)

# Start with 104547 distinct article_ids
# articles = articles[articles["item_count"] > 1]
print("head:\n", articles.head(25))
print("info:\n", articles.info())
# print("article_ids:\n", articles["article_id"].value_counts(dropna=False))
# print("product_codes:\n", articles["product_code"].value_counts(dropna=False))
# print("department_no:\n", articles["department_no"].value_counts(dropna=False))
# print("section_no:\n", articles["section_no"].value_counts(dropna=False))
# print("garment_group_no:\n", articles["garment_group_no"].value_counts(dropna=False))
# Add time
df['year'] = df['date'].dt.isocalendar().year
df['week'] = df['date'].dt.isocalendar().week
df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str)


def get_seasonality(df):
    df = df.groupby('date')['article_id'].count().reset_index()
    print(df.head())
    df = df[['date', 'article_id']]
    df.set_index('date', inplace=True)
    results = seasonal_decompose(df['article_id'])
    results.observed.plot(figsize=(12, 6))
    plt.show()
    results.trend.plot(figsize=(12, 6))
    plt.show()


def weekly_sales_by_section(df):
    counted = df.groupby(['year_week', 'section_name'])['article_id'].count()
    counted_df = counted.reset_index()
    counted_df = counted_df.rename(columns={'article_id': 'count'})
    order = counted_df.section_name.unique().tolist()
    fig = px.bar(counted_df, x='section_name', y='count', animation_frame='year_week', animation_group='section_name',
                 range_y=[0, 110000], template='simple_white', title='Weekly Sales (Units) by Section')
    fig.update_xaxes(categoryorder='array', categoryarray=order)
    fig['layout']['updatemenus'][0]['pad'] = dict(r=10, t=240)
    fig['layout']['sliders'][0]['pad'] = dict(r=10, t=220)
    fig.show(renderer='notebook_connected')


def num_products_per_group(df):
    temp = df.groupby(["product_group_name"])["product_type_name"].nunique()
    df = pd.DataFrame({'Product Group': temp.index, 'Product Types': temp.values})

    df = df.sort_values(['Product Types'], ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.tick_params(axis='y', which='major', labelsize=15)
    ax.set_ylabel('Product group', weight='semibold', fontsize=20)
    ax.set_xlabel('Prodiuct type', weight='semibold', fontsize=20)

    plt.title('Number of Product Types per each Product Group', weight='semibold', fontsize=20)
    sns.set_color_codes("pastel")
    s = sns.barplot(x='Product Group', y="Product Types", data=df)

    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.6)
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    locs, labels = plt.xticks()
    plt.show()


def show_wordcloud(data, title=None):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=200, max_font_size=40, scale=5,
                          random_state=1).generate(str(data))

    fig = plt.figure(1, figsize=(10, 10))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


def num_products_per_group_2(df):
    sns.set_theme(style="darkgrid")
    temp = df.groupby(["product_group_name"])["product_type_name"].nunique()
    df = pd.DataFrame({'Product Group': temp.index, 'Product Types': temp.values})

    df = df.sort_values(['Product Types'], ascending=False)

    f, ax = plt.subplots(figsize=(10, 16))

    sns.set_color_codes("pastel")
    sns.barplot(x=df['Product Types'], y=df['Product Group'], linewidth=0.5, saturation=0.5)

    plt.title('Number of Product Types per each Product Group', weight='bold', fontsize=15)
    plt.xlabel("Product type", weight='semibold', fontsize=15)
    plt.ylabel("Product Group", weight='semibold', fontsize=15)
    plt.show()


def num_articles_per_product_type(df):
    temp = df.groupby(["product_type_name"])["article_id"].nunique()
    df = pd.DataFrame({'Product Type': temp.index, 'Articles': temp.values})
    total_types = len(df['Product Type'].unique())
    df = df.sort_values(['Articles'], ascending=False)[0:50]

    f, ax = plt.subplots(figsize=(10, 16))

    sns.set_color_codes("pastel")
    sns.barplot(x=df['Articles'], y=df['Product Type'], linewidth=0.5, saturation=0.5)

    plt.title(f'Number of Articles per each Product Type (top 50 from total: {total_types})', weight='bold',
              fontsize=15)
    plt.xlabel("Articles", weight='semibold', fontsize=15)
    plt.ylabel("Product Type", weight='semibold', fontsize=15)
    plt.show()


print("Running get_seasonality")
get_seasonality(df)
# print("Running weekly_sales_by_section")
# weekly_sales_by_section(df)
print("Running num_products_per_group")
num_products_per_group(df)
print("Running num_products_per_group_2")
num_products_per_group_2(df)
print("Running num_articles_per_product_type")
num_articles_per_product_type(df)
# print("Running show_wordcloud")
# show_wordcloud(df["prod_name"], "Wordcloud from product name")
