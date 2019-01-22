import argparse
import logging
from datetime import time
from enum import Enum
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from md_lstm.images import create_dataset
from md_lstm.md_lstm_implementation import multi_dimensional_rnn_while_loop

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


def train():
    learning_rate = 10e-4
    batch_size = 2
    h = 270
    w = 270
    channels = 3
    hidden_size = 16
    how_many_classes = 21
    eps = 10e-4

    data_X, data_y = create_dataset()
    print(data_X.shape)
    print(data_y.shape)

    x = tf.placeholder(tf.float32, [batch_size, h, w, channels])
    x_v = tf.placeholder(tf.float32, [batch_size, h, w, channels])
    x_h = tf.placeholder(tf.float32, [batch_size, h, w, channels])
    x_vh = tf.placeholder(tf.float32, [batch_size, h, w, channels])

    y = tf.placeholder(tf.int32, [batch_size, h // 3, w // 3, how_many_classes])

    logger.info('Using Multi Dimensional LSTM.')

    rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                  input_data=x, sh=[3, 3],
                                                  scope_n='layer_1')
    rnn_out_v, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                    input_data=x_v, sh=[3, 3],
                                                    scope_n='layer_2')
    rnn_out_h, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                    input_data=x_h, sh=[3, 3],
                                                    scope_n='layer_3')
    rnn_out_vh, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                     input_data=x_vh, sh=[3, 3],
                                                     scope_n='layer_4')

    model_out = slim.fully_connected(
        inputs=tf.concat([rnn_out, rnn_out_v, rnn_out_h, rnn_out_vh], axis=3),
        num_outputs=how_many_classes,
        activation_fn=tf.nn.tanh)

    model_out_v = tf.image.flip_left_right(model_out)
    model_out_h = tf.image.flip_up_down(model_out)
    model_out_vh = tf.image.flip_up_down(model_out_v)

    rnn_out_2, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                    input_data=model_out,
                                                    sh=[1, 1],
                                                    scope_n='layer_2_1')

    rnn_out_2_v, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                      input_data=model_out_v,
                                                      sh=[1, 1],
                                                      scope_n='layer_2_2')

    rnn_out_2_h, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                      input_data=model_out_h,
                                                      sh=[1, 1],
                                                      scope_n='layer_2_3')

    rnn_out_2_vh, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size,
                                                       input_data=model_out_vh,
                                                       sh=[1, 1],
                                                       scope_n='layer_2_4')

    model_output = slim.fully_connected(
        inputs=tf.concat([rnn_out_2, rnn_out_2_v, rnn_out_2_h, rnn_out_2_vh],
                         axis=3),
        num_outputs=how_many_classes,
        activation_fn=tf.nn.softmax)

    loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(onehot_labels=y, logits=model_output))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    # fp = FileLogger('out_{}.tsv'.format(model_type),
    #                 ['steps_{}'.format(model_type),
    #                  'overall_loss_{}'.format(model_type),
    #                  'time_{}'.format(model_type),
    #                  'relevant_loss_{}'.format(model_type)])
    for i in range(data_X.shape[0] // batch_size):
        grad_step_start_time = time()
        batch_x = data_X[i: i + batch_size]
        batch_x += eps
        print(batch_x.shape)
        batch_y = data_y[i: i + batch_size]

        model_preds, tot_loss_value, _ = sess.run(
            [model_out, loss, grad_update], feed_dict={x: batch_x[:, 0],
                                                       x_v: batch_x[:, 1],
                                                       x_h: batch_x[:, 2],
                                                       x_vh: batch_x[:, 3],
                                                       y: batch_y})

        # print('model preds: {}'.format(model_preds.shape))
        print('total_loss_value: {}'.format(tot_loss_value))

        # import matplotlib.pyplot as plt
        # output_image = model_preds[0, :, :, 0]
        # print(np.min(output_image), np.max(output_image))
        # plt.imshow(output_image, vmin=0, vmax=1)
        # plt.show()

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
        # relevant_pred_index = get_relevant_prediction_index(batch_y)
        # true_rel = np.array([batch_y[i, x, y, 0] for (i, (y, x)) in
        #                      enumerate(relevant_pred_index)])
        # pred_rel = np.array([model_preds[i, x, y, 0] for (i, (y, x)) in
        #                      enumerate(relevant_pred_index)])
        # relevant_loss = np.mean(np.square(true_rel - pred_rel))
        #
        # values = [str(i).zfill(4), tot_loss_value,
        #           time() - grad_step_start_time, relevant_loss]
        # format_str = 'steps = {0} | overall loss = {1:.3f} | time {2:.3f} | relevant loss = {3:.3f}'
        # logger.info(format_str.format(*values))
        # fp.write(values)

        # display_matplotlib_every = 500
        # if enable_plotting and i % display_matplotlib_every == 0 and i != 0:
        #     visualise_mat(
        #         sess.run(model_out, feed_dict={x: batch_x})[0].squeeze())
        #     visualise_mat(batch_y[0].squeeze())


def main():
    # args = get_script_arguments()
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    train()


if __name__ == '__main__':
    main()
