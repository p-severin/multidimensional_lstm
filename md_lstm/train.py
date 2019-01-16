import argparse
import logging
from datetime import time
from enum import Enum
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from md_lstm.md_lstm import multi_dimensional_rnn_while_loop

logger = logging.getLogger(__name__)


def get_script_arguments():
    parser = argparse.ArgumentParser(description='MD LSTM trainer.')
    parser.add_argument('--model_type',
                        required=True,
                        type=ModelType.from_string,
                        choices=list(ModelType), help='Model type.')
    parser.add_argument('--enable_plotting', action='store_true')

    args = get_arguments(parser)
    logger.info('Script inputs: {}.'.format(args))
    return args

class ModelType(Enum):
    MD_LSTM = 'MD_LSTM'

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()


def get_arguments(parser: argparse.ArgumentParser):
    args = None
    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        exit(1)
    return args


def train(enable_plotting):
    learning_rate = 10e-6
    batch_size = 16
    h = 3
    w = 3
    channels = 3
    hidden_size = 16

    x = tf.placeholder(tf.float32, [batch_size, h, w, channels])
    y = tf.placeholder(tf.float32, [batch_size, h, w, channels])

    logger.info('Using Multi Dimensional LSTM.')
    rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                      input_data=x, sh=[1, 1])

    model_out = slim.fully_connected(inputs=rnn_out,
                                     num_outputs=1,
                                     activation_fn=tf.nn.sigmoid)

    loss = tf.reduce_mean(tf.square(y - model_out))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    # fp = FileLogger('out_{}.tsv'.format(model_type),
    #                 ['steps_{}'.format(model_type),
    #                  'overall_loss_{}'.format(model_type),
    #                  'time_{}'.format(model_type),
    #                  'relevant_loss_{}'.format(model_type)])
    steps = 1000
    for i in range(steps):
        batch = next_batch(batch_size, h, w)
        grad_step_start_time = time()
        batch_x = np.expand_dims(batch[0], axis=3)
        batch_y = np.expand_dims(batch[1], axis=3)

        model_preds, tot_loss_value, _ = sess.run(
            [model_out, loss, grad_update], feed_dict={x: batch_x, y: batch_y})

        """
        ____________
        |          |
        |          |
        |     x    |
        |      x <----- extract this prediction. Relevant loss is only computed for this value.
        |__________|    we don't care about the rest (even though the model is trained on all values
                        for simplicity). A standard LSTM should have a very high value for relevant loss
                        whereas a MD LSTM (which can see all the TOP LEFT corner) should perform well. 
        """

        # extract the predictions for the second x
        relevant_pred_index = get_relevant_prediction_index(batch_y)
        true_rel = np.array([batch_y[i, x, y, 0] for (i, (y, x)) in
                             enumerate(relevant_pred_index)])
        pred_rel = np.array([model_preds[i, x, y, 0] for (i, (y, x)) in
                             enumerate(relevant_pred_index)])
        relevant_loss = np.mean(np.square(true_rel - pred_rel))

        values = [str(i).zfill(4), tot_loss_value,
                  time() - grad_step_start_time, relevant_loss]
        format_str = 'steps = {0} | overall loss = {1:.3f} | time {2:.3f} | relevant loss = {3:.3f}'
        logger.info(format_str.format(*values))
        fp.write(values)

        display_matplotlib_every = 500
        if enable_plotting and i % display_matplotlib_every == 0 and i != 0:
            visualise_mat(
                sess.run(model_out, feed_dict={x: batch_x})[0].squeeze())
            visualise_mat(batch_y[0].squeeze())


def main():
    args = get_script_arguments()
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)
    train(args.model_type, args.enable_plotting)

if __name__ == '__main__':
    main()