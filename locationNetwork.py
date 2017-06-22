"""Location network is responsible for manifesting of the locations.

The retina encoding ρ(x, l) extracts k square patches centered at location l,
with the first patch being gw ×gw pixels in size, and each successive patch
having twice the width of the previous. The k patches are then all resized
to gw×gw and concatenated. Glimpse locations l were encoded as
real-valued (x, y) coordinates2 with (0, 0) being the center of the image
x and (−1,−1) being the top left corner of x.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from utils import weight_variable, bias_variable


class LocNet(object):
    """Location network.

    Takes the output from LSTM cell and produces the next location.

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
        self.loc_dim = config.loc_dim
        self.input_dim = config.cell_output_size
        self.loc_std = config.loc_std
        self.n_images = config.n_img_group
        self._sampling = True

        self.__init_weights()

    def __init_weights(self):
        with tf.variable_scope('coordinates'):
            self.w_loc = weight_variable((self.input_dim, self.loc_dim))
            self.b_loc = bias_variable((self.loc_dim,))
        with tf.variable_scope('image_number'):
            self.w_pick = weight_variable((self.input_dim, self.n_images))
            self.b_pick = bias_variable((self.n_images,))

    def __call__(self, input):
        """Produces next location for the glimpse network."""
        # EXTENSTION: don't clip all values, only location vals
        mean = tf.clip_by_value(
            tf.nn.xw_plus_b(input, self.w_loc, self.b_loc), -1., 1.
        )
        mean = tf.stop_gradient(mean)
        picker_softmax = tf.nn.softmax(
            tf.nn.xw_plus_b(input, self.w_pick, self.b_pick)
        )
        # image_number = [batch_size, 1]
        image_number = np.argmax(tf.stop_gradient(picker_softmax), 1)
        if self._sampling:
            loc = mean + tf.random_normal(tf.shape(mean), stddev=self.loc_std)
            loc = tf.clip_by_value(loc, -1., 1.)
        # else:
        #     loc = mean
        loc = tf.stop_gradient(loc)
        return loc, mean, image_number

    # @property
    # def sampling(self):
    #     return self._sampling
    #
    # @sampling.setter
    # def sampling(self, sampling):
    #     self._sampling = sampling



# tf.stop_gradient
# If you insert this op in the graph it inputs are masked from the gradient
# generator. They are not taken into account for computing gradients.
