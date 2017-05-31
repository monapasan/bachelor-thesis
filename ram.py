"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

import tensorflow as tf
import numpy as np

from glimpseNetwork import GlimpseNet
from locationNetwork import LocNet
from glimpseSensor import GlimpseSensor

from utils import weight_variable, bias_variable, loglikelihood
from config import Config

from tensorflow.examples.tutorials.mnist import input_data

logging.getLogger().setLevel(logging.INFO)

rnn_cell = tf.contrib.rnn
seq2seq = tf.contrib.legacy_seq2seq

loc_mean_arr = []
sampled_loc_arr = []


def placeholder_inputs():
    """Generate placeholder variables to represent the input tensors.
    """
    img_size = (
        Config.original_size * Config.original_size * Config.num_channels
    )
    with tf.variable_scope('data'):
        images_ph = tf.placeholder(
            tf.float32, [None, Config.n_img_group, img_size], name='images'
        )
        labels_ph = tf.placeholder(
            tf.int64, [None, Config.n_img_group], name='labels'
        )
    return images_ph, labels_ph


# Build the aux nets.
def init_glimpse_network(images_ph):
    with tf.variable_scope('glimpse_net'):
        gl_sensor = GlimpseSensor(Config)
        gl = GlimpseNet(gl_sensor, Config, images_ph)
    return gl


def init_location_network(config):
    with tf.variable_scope('loc_net'):
        loc_net = LocNet(config)
    return loc_net


def init_baseline_net(outputs):
    # Time independent baselines
    baselines = []
    with tf.variable_scope('baseline'):
        w_baseline = weight_variable((Config.cell_output_size, 1))
        b_baseline = bias_variable((1,))
        for output in outputs[1:]:
            baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
            baseline_t = tf.squeeze(baseline_t)
            baselines.append(baseline_t)
        baselines = tf.stack(baselines)  # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)  # [batch_sz, timesteps]
    return baselines


def lstm_inputs(glimpse_net, N):
    # number of examples
    # N = tf.shape(images_ph)[0]
    # TODO: two-component Gaussian with a fixed variance
    init_loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
    init_glimpse = glimpse_net(init_loc)
    # amount of inputs data should be equal to amound of glimpses
    # once there is no input more, model will make a cl. decision
    inputs = [init_glimpse]
    inputs.extend([0] * Config.num_glimpses)
    return inputs


def init_lstm_cell(config, N):
    lstm_cell = rnn_cell.LSTMCell(Config.cell_size, state_is_tuple=True)
    init_state = lstm_cell.zero_state(N, tf.float32)
    return lstm_cell, init_state


def inference(output):
    # Build classification network.
    with tf.variable_scope('cls'):
        w_logit = weight_variable(
            (Config.cell_output_size, Config.num_classes)
        )
        b_logit = bias_variable((Config.num_classes,))
        logits = tf.nn.xw_plus_b(output, w_logit, b_logit, name='cls_net')
    return logits


def fill_feed_dict(data_set, images_pl, labels_pl):
    images, labels = data_set.next_batch(Config.batch_size)
    # duplicate M times, see Eqn (2)
    # running kinda m episodes
    images_feed = np.tile(images, [Config.M, 1])
    labels_feed = np.tile(labels, [Config.M])
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def init_seq_rnn(images_ph, labels_ph):
    # Core network.
    N = tf.shape(images_ph)[0]
    glimpse_net = init_glimpse_network(images_ph)
    loc_net = init_location_network(Config)
    inputs = lstm_inputs(glimpse_net, N)
    lstm_cell, init_state = init_lstm_cell(Config, N)

    def get_next_input(output, i):
        loc, loc_mean, n_image = loc_net(output)
        gl_next = glimpse_net(loc, n_image)
        loc_mean_arr.append(loc_mean)
        sampled_loc_arr.append(loc)
        return gl_next

    outputs, _ = seq2seq.rnn_decoder(
        inputs, init_state, lstm_cell, loop_function=get_next_input
    )
    return outputs


def accuracy(softmax, batch_size, labels):
    # real_labels = tf.placeholder(tf.int64, [None], name='evaluation_labels')
    softmax_acc = tf.reshape(
        softmax, [Config.M, -1, Config.num_classes]
    )
    softmax_mean = tf.reduce_mean(softmax_acc, 0)  # [32x10]
    pred_labels = tf.argmax(softmax_mean, 1)

    n_correct = tf.reduce_sum(
        tf.cast(tf.equal(pred_labels, labels[:batch_size]), tf.float32)
    )
    # accuracy = n_correct / batch_size
    return n_correct / batch_size


# def n_correct_pred(softmax, batch_size, labels):
#     softmax_acc = tf.reshape(
#         softmax, [Config.M, -1, Config.num_classes]
#     )
#     softmax_mean = tf.reduce_mean(softmax_acc, 0)  # [32x10]
#     pred_labels = tf.argmax(softmax_mean, 1)
#
#     n_correct = tf.reduce_sum(
#         tf.cast(tf.equal(pred_labels, labels[:batch_size]), tf.float32)
#     )
#
#     return n_correct

# def evaluation(sess, dataset, softmax, images_ph, labels_ph):
#     steps_per_epoch = dataset.num_examples // Config.eval_batch_size
#     correct_cnt = 0
#     num_samples = steps_per_epoch * Config.batch_size
#     # loc_net.sampling = True
#     for test_step in range(steps_per_epoch):
#         images, labels = dataset.next_batch(Config.batch_size)
#         labels_bak = labels
#         # Duplicate M times
#         images = np.tile(images, [Config.M, 1])
#         labels = np.tile(labels, [Config.M])
#         feed_dict = {images_ph: images, labels_ph: labels}
#         softmax_val = sess.run(softmax, feed_dict=feed_dict)
#         softmax_val = np.reshape(
#             softmax_val, [Config.M, -1, Config.num_classes]
#         )
#         softmax_val = np.mean(softmax_val, 0)  # [32x10]
#         pred_labels_val = np.argmax(softmax_val, 1)
#         pred_labels_val = pred_labels_val.flatten()
#         correct_cnt += np.sum(pred_labels_val == labels_bak)
#     acc = correct_cnt / num_samples
#     logging.info('accuracy = {}'.format(acc))


def n_correct_pred(softmax, labels):
    softmax_acc = np.reshape(
        softmax, [Config.M, -1, Config.num_classes]
    )
    softmax_mean = np.mean(softmax_acc, 0)  # [32x10]
    pred_labels = np.argmax(softmax_mean, 1).flatten()

    n_correct = np.sum(pred_labels == labels[:Config.batch_size])
    return n_correct
    # tf.cast(tf.equal(pred_labels, labels[:batch_size]), tf.float32)
    # )
    # accuracy = n_correct / batch_size


def evaluation(
    sess, dataset, softmax, images_ph, labels_ph, summary_writer, step, tag
):
    steps_per_epoch = dataset.num_examples // Config.eval_batch_size
    correct_cnt = 0
    num_samples = steps_per_epoch * Config.batch_size
    # loc_net.sampling = True
    for test_step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(dataset, images_ph, labels_ph)
        softmax_val = sess.run(softmax, feed_dict=feed_dict)
        correct_cnt += n_correct_pred(softmax_val, feed_dict[labels_ph])
        # correct_cnt += sess.run(correct_pred, feed_dict=feed_dict)

    acc = correct_cnt / num_samples
    logging.info('{} = {}% \n'.format(tag, acc))
    add_summary(summary_writer, acc, step, True, tag)


def cross_entropy(logits, labels_ph):
    # cross-entropy for classification decision
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels_ph, name="x_entr_for_cls"
    )
    xent = tf.reduce_mean(xent)
    return xent


def get_rewards(pred_labels, labels_ph):
    # 0/1 reward.
    with tf.variable_scope('rewards'):
        reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
        rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
        rewards = tf.tile(rewards, (1, Config.num_glimpses))  # [b_sz,tsteps]
        reward = tf.reduce_mean(reward)
    return reward, rewards


def init_learning_rate(global_step, training_steps_per_epoch):
    starter_learning_rate = Config.lr_start
    learning_rate = tf.train.exponential_decay(
        starter_learning_rate,
        global_step,
        training_steps_per_epoch,
        0.97,
        staircase=True)
    learning_rate = tf.maximum(learning_rate, Config.lr_min)
    return learning_rate


def run_training():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    with tf.Graph().as_default():
        images_ph, labels_ph = placeholder_inputs()
        outputs = init_seq_rnn(images_ph, labels_ph)
        baselines = init_baseline_net(outputs)

        # Take the last step only.
        logits = inference(outputs[-1])
        softmax = tf.nn.softmax(logits)

        xent = cross_entropy(logits, labels_ph)

        pred_labels = tf.argmax(logits, 1)

        reward, rewards = get_rewards(pred_labels, labels_ph)

        logll = loglikelihood(loc_mean_arr, sampled_loc_arr, Config.loc_std)

        advs = rewards - tf.stop_gradient(baselines)
        logllratio = tf.reduce_mean(logll * advs)

        baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
        var_list = tf.trainable_variables()
        # hybrid loss
        loss = -logllratio + xent + baselines_mse  # `-` for minimize

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', loss)

        grads = tf.gradients(loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, Config.max_grad_norm)
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False
        )

        training_steps_per_epoch = (
            mnist.train.num_examples // Config.batch_size
        )

        learning_rate = init_learning_rate(
            global_step, training_steps_per_epoch
        )

        opt = tf.train.AdamOptimizer(learning_rate)
        train_op = opt.apply_gradients(
            zip(grads, var_list), global_step=global_step
        )

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        # saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()  # config=tf.ConfigProto(log_device_placement=T))

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(Config.log_dir, sess.graph)

        accuracy_op = accuracy(
            softmax, Config.batch_size, labels_ph
        )

        accuracy_summary = tf.summary.scalar('Batch accuracy', accuracy_op)

        sess.run(init)
        for i in range(Config.n_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(mnist.train, images_ph, labels_ph)
            runnables = [
                advs, baselines_mse, xent, logllratio,
                reward, loss, learning_rate, accuracy_op, train_op
            ]

            adv_val, baselines_mse_val, xent_val, logllratio_val, reward_val, \
                loss_val, lr_val, acc_val, _ = sess.run(
                    runnables, feed_dict=feed_dict
                )

            duration = time.time() - start_time
            if i and i % 100 == 0:
                log_step(
                    i, duration, lr_val, reward_val, loss_val,
                    xent_val, logllratio_val, baselines_mse_val
                )
                logging.info('Batch accuracy: %f%%. \n' % (acc_val))
                summary_str, acc_sum = sess.run(
                    [summary, accuracy_summary],
                    feed_dict=feed_dict
                )
                add_summary(summary_writer, summary_str, i)
                add_summary(summary_writer, acc_sum, i)
            if i and i % training_steps_per_epoch == 0:
                logging.info('Validation dataset: ')
                evaluation(
                    sess, mnist.validation, softmax, images_ph, labels_ph,
                    summary_writer, i, 'accuracy on validation data'
                )
                logging.info('Test dataset: ')
                evaluation(
                    sess, mnist.test, softmax, images_ph, labels_ph,
                    summary_writer, i, 'accuracy on test data'
                )


def add_summary(writer, summary_val, step, is_custom=False, tag=None):
    if is_custom:
        summary_val = tf.Summary(value=[
            tf.Summary.Value(
                tag=tag, simple_value=summary_val
            )
        ])
    writer.add_summary(summary_val, step)
    writer.flush()


def log_step(
    i, duration, learning_rate,
    reward, loss, x_entr, logllratio, baseline_mse
):
    logging.info('step {}: lr = {:3.6f}, duration: {:.3f} sec'.format(
        i, learning_rate, duration)
    )
    info = 'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'
    logging.info(
        info.format(i, reward, loss, x_entr)
    )
    logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
        logllratio, baseline_mse)
    )
