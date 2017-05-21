from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import weight_variable, bias_variable

"""
The retina encoding ρ(x, l) extracts k square patches centered at location l,
with the first patch being gw ×gw pixels in size, and each successive patch
having twice the width of the previous. The k patches are then all resized
to gw×gw and concatenated. Glimpse locations l were encoded as
real-valued (x, y) coordinates2 with (0, 0) being the center of the image
x and (−1,−1) being the top left corner of x.
"""


class LocNet(object):
    """Location network.

    Take output from other network and produce and sample the next location.

    """

    def __init__(self, config):
        self.loc_dim = config.loc_dim
        self.input_dim = config.cell_output_size
        self.loc_std = config.loc_std
        self._sampling = True

        self.init_weights()

    def init_weights(self):
        self.w = weight_variable((self.input_dim, self.loc_dim))
        self.b = bias_variable((self.loc_dim,))

    def __call__(self, input):
        # EXTENSTION: don't clip all values, only location vals
        mean = tf.clip_by_value(
            tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.
        )
        mean = tf.stop_gradient(mean)
        if self._sampling:
            loc = mean + tf.random_normal(tf.shape(mean), stddev=self.loc_std)
            loc = tf.clip_by_value(loc, -1., 1.)
        # else:
        #     loc = mean
        loc = tf.stop_gradient(loc)
        return loc, mean

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, sampling):
        self._sampling = sampling



# tf.stop_gradient
# If you insert this op in the graph it inputs are masked from the gradient
# generator. They are not taken into account for computing gradients.
