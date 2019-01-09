import tensorflow as tf
from tensorflow.contrib.rnn import GridLSTMCell

class MyModel:
    input_shape = (-1, 3, 3, 3)

    model = GridLSTMCell(num_units=10, num_frequency_blocks=10, input_shape=input_shape)

