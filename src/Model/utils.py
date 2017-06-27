"""Helper functions for the model component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

distributions = tf.contrib.distributions


class dotdict(dict):
    """Small class helper to access to dictionary attributes via '.'."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def weight_variable(shape):
    """Initilise weight variable."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    """Initilise bias variable."""
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def loglikelihood(mean_arr, sampled_arr, sigma):
    """Calculate log likelihood."""
    mu = tf.stack(mean_arr)  # mu = [timesteps, batch_sz, loc_dim]
    sampled = tf.stack(sampled_arr)  # same shape as mu
    gaussian = distributions.Normal(mu, sigma)
    logll = gaussian.log_pdf(sampled)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    logll = tf.transpose(logll)  # [batch_sz, timesteps]
    return logll

# helper
# scp -r src/  s0547796@deepgreen03.f4.htw-berlin.de:/home/05/47796/final_version
# to forward 127.0.0.1:16006 to the remote server :6006 use:
# ssh -L 16006:127.0.0.1:6006 s0547796@deepgreen03.f4.htw-berlin.de
