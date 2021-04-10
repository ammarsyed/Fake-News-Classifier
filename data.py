import os
import newspaper
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
    print('hello')
    # Lowercase all letters
    dataFrame['text'] = dataFrame['text'].apply(lambda x: x.lower())

    # Remove stopwords 
    stop = stopwords.words('english')
    dataFrame['text'] = dataFrame['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in (stop)]))

    # Stem
    ps = PorterStemmer()
    dataFrame['text'] = dataFrame['text'].apply(lambda x: ' '.join([ps.stem(word) for word in word_tokenize(x)]))
    
    # Lemmatize
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    dataFrame['text'] = dataFrame['text'].apply(lambda x: ' '.join([lemmatizer(word) for word in word_tokenize(x)]))



def combineAndShuffle(dataFrame1, dataFrame2):
    print('hello')


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
    if(os.path.exists('./Data/reliable_scraped_articles.csv') and os.path.exists('./Data/unreliable_scraped_articles.csv')):
        reliable_articles = pd.read_csv('./Data/reliable_scraped_articles.csv')
        unreliable_articles = pd.read_csv(
            './Data/unreliable_scraped_articles.csv')
    else:
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
    data = get_data(urls)

    print(data.head())