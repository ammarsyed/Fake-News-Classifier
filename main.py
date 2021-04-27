import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from data import get_preprocessed_data
from models import FakeNewsNN, FakeNewsSVM, FakeNewsLogisticRegression, FakeNewsNaiveBayes, FakeNewsDecisionTree

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import sklearn.metrics as metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    args = parser.parse_args()
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
    data = get_preprocessed_data(urls)
    data = data.dropna().reset_index()
    X = data['text']
    y = np.array(data['label'])

    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(X)
    freq_term_matrix = count_vectorizer.transform(X)

    tfidf = TfidfTransformer()
    tfidf.fit(freq_term_matrix)
    tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)

    if(args.model == 'svm'):

        X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y)
        model = FakeNewsSVM().get_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Accuracy is:", metrics.accuracy_score(y_test, y_pred))
        print(metrics.classification_report(y_test, y_pred))




    elif(args.model == 'nn'):
        vocab_size = 10000
        features = 50
        max_length = 5000
        X = [one_hot(x, vocab_size) for x in X]
        X = pad_sequences(X, maxlen=max_length)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        if(True):
            pass
        else:
            model = FakeNewsNN(vocab_size, features, max_length).get_model()
            model.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=1, batch_size=64)
        model.evaluate(X_test, y_test)

    elif args.model == 'logreg':

        X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y)
        model = FakeNewsLogisticRegression().get_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Accuracy is:", metrics.accuracy_score(y_test, y_pred))
        print(metrics.classification_report(y_test, y_pred))



    elif args.model =='nb':

        X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y)
        model = FakeNewsNaiveBayes().get_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Accuracy is:", metrics.accuracy_score(y_test, y_pred))
        print(metrics.classification_report(y_test, y_pred))

    elif args.model == 'dt':

        X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y)
        model = FakeNewsDecisionTree().get_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Accuracy is:", metrics.accuracy_score(y_test, y_pred))
        print(metrics.classification_report(y_test, y_pred))

    else:
        print('Invalid Model Type')
