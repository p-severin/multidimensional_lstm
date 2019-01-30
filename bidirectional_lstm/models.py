from keras import Sequential
from keras.layers import Bidirectional, LSTM, Permute, Reshape, Dense
from keras.optimizers import Adam

from utils.images import create_dataset

batch_size = 16
rows = 128
cols = 128
channels = 3
classes = 21

def build_model():
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=(rows, cols, channels)))
    model.add(Reshape((rows * cols, channels)))
    model.add(Bidirectional(LSTM(30, return_sequences=True)))
    model.add(Reshape((rows, cols, 2 * 30)))
    model.add(Permute((2, 1, 3)))
    model.add(Reshape((rows * cols, 2 * 30)))
    model.add(Bidirectional(LSTM(30, return_sequences=True)))
    model.add(Dense(classes, activation='softmax'))
    model.add(Reshape((rows, cols, classes)))
    model.summary()
    return model

if __name__ == '__main__':

    model = build_model()
    optimizer = Adam(lr=10e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    X, y = create_dataset()
    print('dataset created')
    model.fit(x=X, y=y, batch_size=batch_size, epochs=5, verbose=1)