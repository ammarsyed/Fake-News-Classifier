from sklearn.svm import SVC
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


class FakeNewsNN:
    def __init__(self, vocab_size, embedding_vector_features, max_sequence_length):
        self.model = Sequential()
        self.model.add(Embedding(
            vocab_size, embedding_vector_features, input_length=max_sequence_length))
        self.model.add(Conv1D(filters=128, kernel_size=5))
        self.model.add(MaxPooling1D(pool_size=1))
        self.model.add(Dense(1, activation='relu'))

    def get_model(self):
        return self.model


class FakeNewsSVM:
    def __init__(self):
        self.model = SVC()

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = FakeNewsNN(100, 100, 100).get_model()
    print(model.summary())
