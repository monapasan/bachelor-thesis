import os

_dir = os.getcwd()


class Config(object):
    win_size = 8
    bandwidth = win_size**2
    batch_size = 32
    eval_batch_size = 50
    # standard deviation for
    # the gaussian distribution
    # when added to location
    loc_std = 0.22
    # zise of MNIST image, 28px x 28px
    original_size = 28
    # number of channels in the image, MNIST has 1 channel, i.e. white channel
    num_channels = 1
    #  number of patches which should be extracted in a glimpse
    n_patches = 1
    sensor_size = win_size**2 * n_patches
    minRadius = 8
    # Size of parameters for Glimpse network
    hg_size = hl_size = 128
    g_size = 256

    log_dir = _dir + '/log/'

    cell_output_size = 256
    # location shape --> (x, y)
    loc_dim = 2
    cell_size = 256

    num_glimpses = 6
    num_classes = 10
    max_grad_norm = 5.

    # amount of steps for training the model
    step = 100000
    lr_start = 1e-3
    lr_min = 1e-4

    # Monte Carlo sampling
    M = 10
