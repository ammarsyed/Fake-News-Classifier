import os
import pickle
import argparse
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from data import get_preprocessed_data
from models import FakeNewsNN, FakeNewsSVM, FakeNewsLogisticRegression, FakeNewsNaiveBayes, FakeNewsDecisionTree

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import sklearn.metrics as metrics
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
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(X)
    freq_term_matrix = count_vectorizer.transform(X)
    tfidf = TfidfTransformer()
    tfidf.fit(freq_term_matrix)
    tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
    if(args.model == 'svm'):
        X_train, X_test, y_train, y_test = train_test_split(
            tf_idf_matrix, y, test_size=0.25, random_state=42)
        if(os.path.exists('./Models/fake_news_svm.pickle')):
            model = pickle.load(open('./Models/fake_news_svm.pickle', 'rb'))
        else:
            model = FakeNewsSVM('linear').get_model()
            print('Training SVM Model')
            model.fit(X_train, y_train)
            pickle.dump(model, open('./Models/fake_news_svm.pickle', 'wb'))
        print('Making Predictions on Test Data')
        y_pred = model.predict(X_test)
        print('Generating Figures and Reporting Results')
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc_val = roc_auc_score(y_test, y_pred)
        plot_confusion_matrix(cm, 'svm', 'SVM Confusion Matrix')
        plot_roc_curve(fpr, tpr, 'svm', 'SVM ROC Curve')
        print('Accuracy: ', accuracy_score(y_test, y_pred))
        print('AUC: ', auc_val)
        print(classification_report(y_test, y_pred, digits=4))
    elif(args.model == 'nn'):
        vocab_size = 5000
        num_features = 40
        max_text_length = 3000
        X = [one_hot(x, vocab_size) for x in X]
        X = pad_sequences(X, maxlen=max_text_length)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)
        model = FakeNewsNN(vocab_size, num_features,
                           max_text_length).get_model()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=2, batch_size=32)
        #model.evaluate(X_test, y_test)
        print('Making Predictions on Test Data')
        y_pred = model.predict(X_test)
        y_pred = np.round(y_pred)
        print('Generating Figures and Reporting Results')
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc_val = roc_auc_score(y_test, y_pred)
        plot_confusion_matrix(cm, 'nn', 'CNN-RNN Confusion Matrix')
        plot_roc_curve(fpr, tpr, 'nn', 'CNN-RNN ROC Curve')
        print('Accuracy: ', accuracy_score(y_test, y_pred))
        print('AUC: ', auc_val)
        print(classification_report(y_test, y_pred, digits=4))

    elif args.model == 'logreg':

        X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y, test_size=0.25, random_state=42)
        model = FakeNewsLogisticRegression().get_model()
        model.fit(X_train, y_train)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Accuracy is:", metrics.accuracy_score(y_test, y_pred))
        print(metrics.classification_report(y_test, y_pred))



    elif args.model =='nb':

        X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y, test_size=0.25, random_state=42)
        model = FakeNewsNaiveBayes().get_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Accuracy is:", metrics.accuracy_score(y_test, y_pred))
        print(metrics.classification_report(y_test, y_pred))

    elif args.model == 'dt':

        X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y, test_size=0.25, random_state=42)
        model = FakeNewsDecisionTree().get_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Accuracy is:", metrics.accuracy_score(y_test, y_pred))
        print(metrics.classification_report(y_test, y_pred))


    else:
        print('Invalid Model Type')
