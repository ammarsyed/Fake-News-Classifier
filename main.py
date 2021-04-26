import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from data import get_preprocessed_data
from models import FakeNewsNN, FakeNewsSVM

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
    if(args.model == 'svm'):
        model = FakeNewsSVM().get_model()
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
    else:
        print('Invalid Model Type')
