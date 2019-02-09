from keras import Sequential
from keras.layers import Bidirectional, LSTM, Permute, Reshape, Dense
from keras.optimizers import Adam

from utils.images import create_dataset

batch_size = 8
rows = 32
cols = 32
channels = 27
classes = 21
hidden_size = 30

def build_model():
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=(rows, cols, channels)))
    model.add(Reshape((rows * cols, channels)))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Reshape((rows, cols, hidden_size)))
    model.add(Permute((2, 1, 3)))
    model.add(Reshape((rows * cols, hidden_size)))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dense(classes, activation='tanh'))
    model.add(Reshape((rows, cols, classes)))
    model.summary()
    return model

def build_another_model():
    model = Sequential()
    model.add(build_model())
    # model.add(Dense(10))
    model.summary()

if __name__ == '__main__':

    model = build_another_model()
    # optimizer = Adam(lr=10e-4)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # X, y = create_dataset()
    # print('dataset created')
    # model.fit(x=X, y=y, batch_size=batch_size, epochs=5, verbose=1)