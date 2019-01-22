from keras import Sequential, Input
from keras.layers import Bidirectional, LSTM

batch_size = 1
rows = 270
cols = 270
channels = 3

def build_model():
    input_rows = Input(shape=(rows, channels))
    input_cols = Input(shape=(cols, channels))
    model = Sequential()
    model.add(Bidirectional(LSTM(30, return_sequences=True), input_shape=(rows, channels)))
    model.summary()

if __name__ == '__main__':
    build_model()