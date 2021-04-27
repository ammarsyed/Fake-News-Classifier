import os
import argparse
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from data import get_preprocessed_data
from models import FakeNewsNN, FakeNewsSVM
from visualizations import *

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
    if(not os.path.exists('./Figures/class_distribution.png')):
        plot_class_distribution(data)
    if(not os.path.exists('./Figures/reliable_word_cloud.png') or not os.path.exists('./Figures/unreliable_word_cloud.png')):
        plot_word_clouds(data)
    X = data['text']
    y = np.array(data['label'])
    if(args.model == 'svm'):
        count_vectorizer = CountVectorizer()
        count_vectorizer.fit_transform(X)
        freq_term_matrix = count_vectorizer.transform(X)
        tfidf = TfidfTransformer()
        tfidf.fit(freq_term_matrix)
        tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
        X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y)
        print(tf_idf_matrix[0])
        print('split data')
        model = FakeNewsSVM().get_model()
        print('got model')
        model.fit(X_train, y_train)
        print('fit model')
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc_val = roc_auc_score(y_test, y_pred)
        plot_confusion_matrix(cm, 'svm', 'SVM Confusion Matrix')
        plot_roc_curve(fpr, tpr, 'svm', 'SVM ROC Curve')
        print('done predicting model')
        print('Accuracy: ', accuracy_score(y_test, y_pred))
        print('AUC: ', auc_val)
        print(classification_report(y_test, y_pred))
    elif(args.model == 'nn'):
        vocab_size = 10000
        num_features = 50
        max_text_length = 5000
        X = [one_hot(x, vocab_size) for x in X]
        X = pad_sequences(X, maxlen=max_text_length)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        model = FakeNewsNN(vocab_size, num_features,
                           max_text_length).get_model()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=1, batch_size=64)
        #model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        y_pred = np.round(y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc_val = roc_auc_score(y_test, y_pred)
        plot_confusion_matrix(cm, 'nn', 'CNN-RNN Confusion Matrix')
        plot_roc_curve(fpr, tpr, 'nn', 'CNN-RNN ROC Curve')
        print('Accuracy: ', accuracy_score(y_test, y_pred))
        print('AUC: ', auc_val)
        print(classification_report(y_test, y_pred))
    else:
        print('Invalid Model Type')
