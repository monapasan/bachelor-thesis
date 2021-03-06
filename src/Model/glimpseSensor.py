"""Glimpse sensor is responsible for extracting glimpses from images"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class GlimpseSensor(object):
    """GlimpseSensor is configurable class that extract `depth` glimpses.

    This class does not hold any images, but the images should be passed
    as paramter to the __call__ of the class.
    """

    def __init__(self, config):
        """Initialise the GlimpseSensor.

        Args:
            config(Config) - configuration object that should posses
                of following properties: `glimpse_depth`, `win_size`,
                `num_channels`, `original_size`, `glimpse_scale`.
                You will find the explanation behind the properites
                in main.py.
    """
        # TODO: config should have property depth/scale
        # which represents how many patches should one glimpse have.
        self.depth = config.glimpse_depth
        self.win_size = config.win_size
        self.num_channels = config.num_channels
        self.original_size = config.original_size
        self.scale = config.glimpse_scale

    def __call__(self, images_ph, loc):
        """Return glimpse on the original images."""
        # loc.shape --> [batch_size, 2]
        # imgs.shape --> [batch_size, 28, 28, 1]
        batch_size = tf.shape(images_ph)[0]
        imgs = tf.reshape(images_ph, [
            batch_size, self.original_size,
            self.original_size, self.num_channels
        ])
        # win_size = 8
        # [batch_size, win_size, win_size, num_channels]
        glimpse_imgs = tf.image.extract_glimpse(
            imgs, [self.win_size, self.win_size], loc
        )
        # imgs = []
        # for i in range(self.depth):
        #     # win_size = self.win_size * self.scale
        #     if(i == 1):
        #         win_size = self.win_size
        #     else:
        #         win_size = win_size * self.scale
        #     imgs.append(tf.image.extract_glimpse(
        #         imgs, [self.win_size, self.win_size], loc
        #     ))

        glimpse_imgs = tf.reshape(glimpse_imgs, [
            batch_size, self.win_size * self.win_size * self.num_channels
        ])
        # [batch_size, 8 * 8 * 1] or
        # [batch_size, win_size * win_size * num_channels]

        # here we exract only one patch
        # which is good enough to reproduce results
        # accroding to the paper.
        # TODO: but, to generalise the model, it's needed to extract several
        # patches. Therefore this function should be replaced by:
        # self.forward(img, loc),
        # glimpse_sensor class will have property n_patches
        # to resize images take a look at
        # https://www.tensorflow.org/api_docs/python/tf/image/resize_bilinear
        # https://www.tensorflow.org/api_docs/python/tf/image/resize_images
        # tf.resize_bilinear().
        return glimpse_imgs
