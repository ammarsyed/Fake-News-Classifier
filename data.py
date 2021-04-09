import os
import newspaper
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing import one_hot

urls = {
    'reliable': [
        'https://www.washingtonpost.com/',
        'https://www.nytimes.com/',
        'https://www.nbcnews.com/'
    ],
    'unreliable': [
        'https://www.breitbart.com/',
        'https://www.infowars.com/',
        'https://ussanews.com/News1/'
    ]
}


def scrape_data(article_type, url_list):
    data = []
    file_name = '{}_scraped_articles.csv'.format(article_type)
    article_label = (0, 1)[article_type == 'unreliable']
    for url in url_list:
        print(url)
        paper = newspaper.build(url, memoize_articles=False)
        articles = paper.articles
        for i, article in enumerate(articles):
            try:
                article.download()
                article.parse()
            except:
                continue
            if(article.publish_date is None):
                continue
            print(i, article.title)
            article_features = {
                'title': article.title,
                'text': article.text,
                'label': article_label
            }
            data.append(article_features)
    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    df.to_csv('./Data/{}'.format(file_name))
    return df


def get_scraped_date(urls):
    if(os.path.exists('reliable_scraped_articles.csv') and os.path.exists('unreliable_scraped_articles.csv')):
        reliable_articles = pd.read_csv('reliable_scraped_articles.csv')
        unreliable_articles = pd.read_csv('unreliable_scraped_articles.csv')
    else:
        reliable_articles = scrape_data('reliable', urls['reliable'])
        unreliable_articles = scrape_data('unreliable', urls['unreliable'])
    return pd.concat([reliable_articles, unreliable_articles])


def tfidf_vectorize(corpus):
    vectorizer = TfidfVectorizer()
    vectorized_text = vectorizer.fit_transform(text)
    return vectorized_text


def one_hot_vectorization(corpus, vocab_size):
    return [one_hot(text, vocab_size) for text in corpus]


if __name__ == '__main__':
    get_scraped_date(urls)
