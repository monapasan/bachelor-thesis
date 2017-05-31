from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import weight_variable, bias_variable

"""
The glimpse network fg(x, l) had two fully connected layers.
Let Linear(x) de- note a linear transformation of the vector x,
i.e. Linear(x) = Wx+b for some weight matrixW and bias vector b,
and letRect(x) = max(x, 0) be the rectifier nonlinearity.
The output g of the glimpse network was defined as
g = Rect(Linear(hg)+Linear(hl)) where hg = Rect(Linear(ρ(x, l)))
and hl = Rect(Linear(l)). The dimensionality of hg and hl was 128
while the dimensionality of g was 256 for all attention models.


GlimpseNet(x, l) = Rect(Linear(hg)+Linear(hl))
where:
- hg = Rect(Linear(ρ(x, l)))
- hl = Rect(Linear(l))
or:
GlimpseNet(x, l) = Rect(Hg + Hl)
where:
Hg = Linear(Rect(Linear(ρ(x, l))))
Hl = Linear(Rect(Linear(l)))

----
where:
x - input image
l - location
"""


class GlimpseNet(object):
    """Glimpse network.

    Take glimpse location input and output features for RNN.

    """

    def __init__(self, glimpse_sensor, config, images_ph):
        self.glimpse_sensor = glimpse_sensor

        self.original_size = config.original_size
        self.num_channels = config.num_channels

        self.sensor_size = config.win_size**2 * config.glimpse_depth
        # self.win_size = config.win_size
        # self.minRadius = config.minRadius

        self.hg_size = config.hg_size
        self.hl_size = config.hl_size
        self.g_size = config.g_size

        self.loc_dim = config.loc_dim

        self.images_ph = images_ph

        self.init_weights()

    def init_weights(self):
        """ Initialize all the trainable weights."""
        """fg(.; {θ_0_g, θ_1_g, θ_2_g})"""
        self.w_g0 = weight_variable((self.sensor_size, self.hg_size))  # 64x128
        self.b_g0 = bias_variable((self.hg_size,))

        self.w_l0 = weight_variable((self.loc_dim, self.hl_size))  # 2x128
        self.b_l0 = bias_variable((self.hl_size,))

        self.w_g1 = weight_variable((self.hg_size, self.g_size))  # 128 x 256
        self.b_g1 = bias_variable((self.g_size,))

        self.w_l1 = weight_variable((self.hl_size, self.g_size))
        self.b_l1 = weight_variable((self.g_size,))

    def _get_glimpse(self, loc, n_image):
        # self.images_ph = [batch, n_images, pixel]
        # self.images_ph = [batch, pixel]
        # n_image.shape  = [batch_size, 1]
        # filter placholder based on the value from location network, n_images
        # EXTENSTION: choose image based on location value
        # TODO: ^^
        batch_size = tf.shape(self.images_ph)[0]
        indicies = tf.stack([tf.range(batch_size), tf.squeeze(n_image)])
        selected_images = tf.gather_nd(self.images_ph, indicies, axis=-1)

        return self.glimpse_sensor(selected_images, loc)

    def __call__(self, loc, n_image):
        """
        GlimpseNet(x, l) = Rect(Hg + Hl)
        where:
        Hg = Linear(Rect(Linear(ρ(x, l))))
        Hl = Linear(Rect(Linear(l)))
        """
        glimpse_input = self._get_glimpse(loc, n_image)
        # glimpse_input = tf.reshape(glimpse_input,
        #                            (tf.shape(loc)[0], self.sensor_size))
        # g_0 --> batch_size, 128
        g_0 = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, self.w_g0, self.b_g0))
        # g_1 --> batch_size x 256, g_size
        g_1 = tf.nn.xw_plus_b(g_0, self.w_g1, self.b_g1)

        # loc --> [batch_size, 2]
        # l_0 --> [batch_size, 128]
        l_0 = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
        # l_1 --> [batch_size, 256] g_size
        l_1 = tf.nn.xw_plus_b(l_0, self.w_l1, self.b_l1)

        g = tf.nn.relu(g_1 + l_1)
        # g --> [batch_size, 256],  g_size=256
        return g
