"""Picker network picks an image out of group to attend to."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .utils import weight_variable, bias_variable


class PickerNet(object):
    """Picker network.

    Takes the output from LSTM cell and produces number of the next image for
    glimpse network.
    """

    def __init__(self, config):
        """Initialise the Location Network.

        Args:
            config(Config) - configuration object that should posses
                of following properties: `loc_dim`, `cell_output_size`,
                `loc_std`, `n_img_group`.
                You will find the explanation behind the properites
                in main.py.
        """
        self.input_dim = config.cell_output_size
        self.n_images = config.n_img_group

        self.__init_weights()

    def __init_weights(self):
        with tf.variable_scope('image_number'):
            self.w_pick = weight_variable((self.input_dim, self.n_images))
            self.b_pick = bias_variable((self.n_images,))

    def __call__(self, input):
        """Produce number of the next image for the glimpse network.

        Given the output of LSTM cell.
        """
        picker_softmax = tf.nn.softmax(
            tf.nn.xw_plus_b(input, self.w_pick, self.b_pick)
        )
        # image_number = [batch_size, 1]
        image_number = tf.cast(
            tf.argmax(tf.stop_gradient(picker_softmax), 1), dtype=tf.int32
        )
        return image_number
