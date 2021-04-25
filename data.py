import os
import newspaper
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


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


def preprocess_words(dataFrame):
    dataFrame = dataFrame.dropna().reset_index()
    dataFrame['text'] = dataFrame['title'] + " " + dataFrame['text']
    # Lowercase all letters
    dataFrame['text'] = dataFrame['text'].apply(lambda x: x.lower())
    # for i in range(1, len(dataFrame)):
    #     try:
    #         # dataFrame['text'] = dataFrame['text'].apply(lambda x: x.lower())
    #         dataFrame['text'][i] = dataFrame['text'][i].lower()
    #     except:
    #         print(i, dataFrame['text'][i])
    # Remove stopwords, punctuation, and numbers
    stop = stopwords.words('english')
    dataFrame['text'] = dataFrame['text'].apply(lambda x: ' '.join([word for word in word_tokenize(
        x) if word not in (stop) and not word in string.punctuation and not word.isdigit()]))
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    dataFrame['text'] = dataFrame['text'].apply(lambda x: ' '.join(
        [lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
    return dataFrame


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


def get_scraped_data(urls):
    # if(os.path.exists('./Data/reliable_scraped_articles.csv') and os.path.exists('./Data/unreliable_scraped_articles.csv')):
    #    reliable_articles = pd.read_csv('./Data/reliable_scraped_articles.csv')
    #    unreliable_articles = pd.read_csv(
    #        './Data/unreliable_scraped_articles.csv')
    # else:
    reliable_articles = scrape_data('reliable', urls['reliable'])
    unreliable_articles = scrape_data('unreliable', urls['unreliable'])
    return pd.concat([reliable_articles, unreliable_articles], ignore_index=True)


def get_data(urls):
    scraped_data = get_scraped_data(urls)
    reliable_data = pd.read_csv('./Data/reliable_articles.csv')
    unreliable_data = pd.read_csv('./Data/unreliable_articles.csv')
    data = pd.concat([scraped_data, reliable_data,
                      unreliable_data], ignore_index=True)
    return data


def tfidf_vectorize(corpus):
    vectorizer = TfidfVectorizer()
    vectorized_corpus = vectorizer.fit_transform(corpus)
    return vectorized_corpus


if __name__ == '__main__':
    # get_scraped_date(urls)
    data = get_data(urls)[:250]
    data = preprocess_words(data)

    data.to_csv('processed.csv')
