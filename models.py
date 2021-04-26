from sklearn.svm import SVC
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


class FakeNewsNN:
    def __init__(self, vocab_size, embedding_vector_features, max_sequence_length):
        self.model = Sequential()
        self.model.add(Embedding(
            vocab_size, embedding_vector_features, input_length=max_sequence_length))
        self.model.add(Dropout(0.4))
        self.model.add(Conv1D(filters=32, kernel_size=5,
                              padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(100))
        self.model.add(BatchNormalization())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

    def get_model(self):
        return self.model


class FakeNewsSVM:
    def __init__(self):
        self.model = SVC(kernel='linear', verbose=True) #can try with default (rbf) kernel too


    def get_model(self):
        return self.model
