"""The main file of the prototype. It runs the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import argparse
import sys

import config

_dir = os.getcwd()
FLAGS = None


def main(_):
    """Run the Visual Attention model with the extension."""
    from Model.ram import run_training
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--win_size',
        type=int,
        default=8,
        help='Windows size for the glimpse sensor.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=50,
        help='Evaluation Batch size.'
    )
    parser.add_argument(
        '--loc_std',
        type=float,
        default=0.22,
        help='Standard deviation for normal distribution when samling location'
    )
    parser.add_argument(
        '--original_size',
        type=int,
        default=28,
        help='Size of original image. (Mnist image = 28x28)'
    )
    parser.add_argument(
        '--num_channels',
        type=int,
        default=1,
        help='number of channels in the image, f.i. MNIST has 1 channel.'
    )
    parser.add_argument(
        '--glimpse_depth',
        type=int,
        default=1,
        help='depth is number of patches to crop per glimpse \
            (one patch per depth).'
    )
    parser.add_argument(
        '--glimpse_scale',
        type=int,
        default=1,
        help='Scale determines the size(t) = scale * size(t-1) of \
            successive cropped patches.'
    )
    parser.add_argument(
        '--hg_size',
        type=int,
        default=128,
        help='Size of parameters for Glimpse network.'
    )
    parser.add_argument(
        '--hl_size',
        type=int,
        default=128,
        help='Size of parameters for Glimpse network.'
    )
    parser.add_argument(
        '--g_size',
        type=int,
        default=256,
        help='Size of parameters for Glimpse network.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=(_dir + '/log/'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--cell_output_size',
        type=int,
        default=256,
        help='Output size for RNN cell'
    )
    parser.add_argument(
        '--cell_size',
        type=str,
        default=256,
        help='Size of RNN cell'
    )

    parser.add_argument(
        '--loc_dim',
        type=int,
        default=2,
        help='Size of dimension for location representation'
    )

    parser.add_argument(
        '--num_glimpses',
        type=int,
        default=24,
        help='Amount of glimpses needs to be taken before making \
        a classfication decision.'
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.,
        help='The clipping ratio for gradient clipping.'
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=100000,
        help='Amount of steps for training the model.'
    )

    parser.add_argument(
        '--lr_start',
        type=float,
        default=1e-3,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--lr_min',
        type=float,
        default=1e-4,
        help='Minimal learning rate.'
    )
    parser.add_argument(
        '--M',
        type=int,
        default=10,
        help='Amount of episodes per image for Monte Carlo sampling.'
    )

    """Dataset configuration."""

    parser.add_argument(
        '--n_img_group',
        type=int,
        default=4,
        help='Amount of images per group.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=(_dir + '/MNIST_data/'),
        help='Directory to store the original MNIST data'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='Amount of classes for classification task.'
    )
    parser.add_argument(
        '--noises_per_class',
        type=list,
        default=[2, 1],
        help='Amount of noise images per class'
    )
    parser.add_argument(
        '--num_examples_per_class',
        type=list,
        default=[30000, 30000],
        help='Amount of examples to generate for the training dataset'
    )
    parser.add_argument(
        '--num_examples_per_class_val',
        type=list,
        default=[4000, 4000],
        help='Amount of examples to generate for the validation dataset'
    )
    parser.add_argument(
        '--num_examples_per_class_test',
        type=list,
        default=[7000, 7000],
        help='Amount of examples to generate for the test dataset'
    )
    parser.add_argument(
        '--noise_label_index',
        type=list,
        default=[0],
        help='Images at these labels are noise images'
    )
    parser.add_argument(
        '--data_label_index',
        type=list,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        help='Images at these labels are non-noise images'
    )
    FLAGS, unparsed = parser.parse_known_args()
    config.init_config(FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
