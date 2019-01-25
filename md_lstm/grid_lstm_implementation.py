import tensorflow as tf

# Network Parameters
from tensorflow.contrib import rnn
from tensorflow.contrib.grid_rnn.python.ops import grid_rnn_cell

n_depth = 5
n_input_x = 200 # MNIST data input (img shape: 28*28)
n_input_y = 200
n_hidden = 128 # hidden layer num of features
n_classes = 2

# tf Graph input
x = tf.placeholder("float", [None, n_depth, n_input_x, n_input_y])
y = tf.placeholder("float", [None, n_depth, n_input_x, n_input_y, n_classes])

# Define weights
weights = {}
biases = {}

# Initialize weights
for i in range(n_depth * n_input_x * n_input_y):
    weights[i] = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    biases[i] = tf.Variable(tf.random_normal([n_classes]))

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_input_y, n_input_x)
    # Permuting batch_size and n_input_y
    x = tf.reshape(x, [-1, n_input_y, n_depth * n_input_x])
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_input_y*batch_size, n_input_x)

    x =  tf.reshape(x, [-1, n_input_x * n_depth])

    # Split to get a list of 'n_input_y' tensors of shape (batch_size, n_hidden)
    # This input shape is required by `rnn` function
    x = tf.split(0, n_depth * n_input_x * n_input_y, x)

    # Define a lstm cell with tensorflow
    lstm_cell = grid_rnn_cell.GridRNNCell(n_hidden, input_dims=[n_depth, n_input_x, n_input_y])
    # lstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * 12, state_is_tuple=True)
    # lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.8)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    # pdb.set_trace()

    output = []
    for i in range(n_depth * n_input_x * n_input_y):
        #I'll need to do some sort of reshape here on outputs[i]
        output.append(tf.matmul(outputs[i], weights[i]) + biases[i])

    return output


pred = RNN(x, weights, biases)
pred = tf.transpose(tf.pack(pred),[1,0,2])
pred = tf.reshape(pred, [-1, n_depth, n_input_x, n_input_y, n_classes])
# pdb.set_trace()
temp_pred = tf.reshape(pred, [-1, n_classes])
n_input_y = tf.reshape(y, [-1, n_classes])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(temp_pred, n_input_y))