"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

import tensorflow as tf
import numpy as np

from Model.glimpseNetwork import GlimpseNet
from Model.locationNetwork import LocNet
from Model.pickerNetwork import PickerNet
from Model.glimpseSensor import GlimpseSensor

from Model.utils import weight_variable, bias_variable, loglikelihood, dotdict
from Dataset.GroupDataset import GroupDataset
from Dataset.IndexGenerator import IndexGenerator

from config import Config

from tensorflow.examples.tutorials.mnist import input_data

logging.getLogger().setLevel(logging.INFO)

rnn_cell = tf.contrib.rnn
seq2seq = tf.contrib.legacy_seq2seq

loc_mean_arr = []
sampled_loc_arr = []


def init_group_dataset(dataset, n_examples):
    """Initilise the GroupDataset for testing purposes."""
    noise_quantity = Config.noises_per_class
    amount_of_classes = Config.num_classes

    noise_label_index = Config.noise_label_index
    data_label_index = Config.data_label_index

    images_per_sample = Config.n_img_group

    myIG = IndexGenerator(noise_quantity, amount_of_classes)
    return GroupDataset(
        myIG, dataset, noise_label_index,
        data_label_index, amount_of_classes, noise_quantity,
        n_examples, images_per_sample
    )


def init_dataset():
    """Initialise dataset for the training, validation and testing."""
    path = Config.data_dir
    mnist = input_data.read_data_sets(path, one_hot=True)
    n_train = Config.num_examples_per_class
    n_validation = Config.num_examples_per_class_test
    n_test = Config.num_examples_per_class_val

    dataset = {
        'train': init_group_dataset(mnist.train, n_train),
        'validation': init_group_dataset(mnist.validation, n_validation),
        'test': init_group_dataset(mnist.test, n_test),
    }
    return dotdict(dataset)


def placeholder_inputs():
    """Generate placeholder variables to represent the input tensors."""
    img_size = (
        Config.original_size * Config.original_size * Config.num_channels
    )
    images_shape = [None, Config.n_img_group, img_size]
    labels_shape = [None]
    with tf.variable_scope('data'):
        images_ph = tf.placeholder(tf.float32, images_shape, name='images')
        labels_ph = tf.placeholder(tf.int64, labels_shape, name='labels')
    return images_ph, labels_ph


# Build the auxiliar nets.
def init_glimpse_network(images_ph):
    """Initialise glimpse network using `glimpse_net` scope and return it."""
    with tf.variable_scope('glimpse_net'):
        gl_sensor = GlimpseSensor(Config)
        gl = GlimpseNet(gl_sensor, Config, images_ph)
    return gl


def init_location_network(config):
    """Initialise location network using `loc_net` scope and return it."""
    with tf.variable_scope('loc_net'):
        loc_net = LocNet(config)
    return loc_net


def init_picker_network(config):
    """Initialise picker netwotk using `pick_net` scope and return it."""
    with tf.variable_scope('pick_net'):
        pick_net = PickerNet(config)
    return pick_net


def init_baseline_net(outputs):
    """Initialise value function as baseline for the advantage function.

    Takes outputs from LSTM cells and feed it into hidden layer producing
    desired representation of the outputs, i.e. value function.
    """
    # Time independent baselines
    baselines = []
    with tf.variable_scope('baseline'):
        w_baseline = weight_variable((Config.cell_output_size, 1))
        b_baseline = bias_variable((1,))
        for each in outputs[1:]:
            baseline_t = tf.nn.xw_plus_b(each, w_baseline, b_baseline)
            baseline_t = tf.squeeze(baseline_t)
            baselines.append(baseline_t)
        baselines = tf.stack(baselines)  # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)  # [batch_sz, timesteps]
    return baselines


def lstm_inputs(glimpse_net, N):
    """Initialise inputs for LSTM cell."""
    # number of examples
    # N = tf.shape(images_ph)[0]
    # TODO: two-component Gaussian with a fixed variance
    init_loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
    init_img = tf.random_uniform(
        [N], minval=0, maxval=(Config.n_img_group), dtype=tf.int32
    )
    init_glimpse = glimpse_net(init_loc, init_img)
    # amount of inputs data should be equal to amound of glimpses
    # once there is no input more, model will make a cl. decision
    inputs = [init_glimpse]
    inputs.extend([0] * Config.num_glimpses)
    return inputs


def init_lstm_cell(config, N):
    """Initialise LSTM cell."""
    lstm_cell = rnn_cell.LSTMCell(Config.cell_size, state_is_tuple=True)
    init_state = lstm_cell.zero_state(N, tf.float32)
    return lstm_cell, init_state


def inference(output):
    """Initialise logits for classification task."""
    # Build classification network.
    with tf.variable_scope('cls'):
        w_logit = weight_variable(
            (Config.cell_output_size, Config.num_classes)
        )
        b_logit = bias_variable((Config.num_classes,))
        logits = tf.nn.xw_plus_b(output, w_logit, b_logit, name='cls_net')
    return logits


def fill_feed_dict(dataset, images_pl, labels_pl):
    """Fill `images_pl` and `labels_pl` with data from `dataset`."""
    images, labels = dataset.next_batch(Config.batch_size)
    # duplicate M times, see Eqn (2)
    # running kinda m episodes
    images_feed = np.tile(images, [Config.M, 1, 1])
    labels_feed = np.tile(labels, [Config.M])
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def init_seq_rnn(images_ph, labels_ph):
    """Feed locations to LSTM cells and return output of LSTM cells."""
    # Core network.
    N = tf.shape(images_ph)[0]
    glimpse_net = init_glimpse_network(images_ph)
    loc_net = init_location_network(Config)
    pick_net = init_picker_network(Config)
    inputs = lstm_inputs(glimpse_net, N)
    lstm_cell, init_state = init_lstm_cell(Config, N)

    def get_next_input(output, i):
        loc, loc_mean = loc_net(output)
        n_image = pick_net(output)
        gl_next = glimpse_net(loc, n_image)
        loc_mean_arr.append(loc_mean)
        sampled_loc_arr.append(loc)
        return gl_next

    outputs, _ = seq2seq.rnn_decoder(
        inputs, init_state, lstm_cell, loop_function=get_next_input
    )
    return outputs


def accuracy(softmax, batch_size, labels):
    """Calculate the accuracy of the batch based on softmax."""
    # real_labels = tf.placeholder(tf.int64, [None], name='evaluation_labels')
    softmax_acc = tf.reshape(
        softmax, [Config.M, -1, Config.num_classes]
    )
    softmax_mean = tf.reduce_mean(softmax_acc, 0)  # [32x10]
    pred_labels = tf.argmax(softmax_mean, 1)

    n_correct = tf.reduce_sum(
        tf.cast(tf.equal(pred_labels, labels[:batch_size]), tf.float32)
    )
    return n_correct / batch_size


def n_correct_pred(softmax, labels):
    """Calculate number of correct predictions."""
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
    """Evaluate result of the training.

    Including the logging of results, and adding summary for TensorBoard.
    """
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
    logging.info('{} = {:.4f}% \n'.format(tag, acc))
    add_summary(summary_writer, acc, step, True, tag)


def cross_entropy_with_logits(logits, labels_ph):
    """Initialise cross-entropy operation for classicatiton.

    I.e. cross-entropy between logins and ground truth.
    """
    # cross-entropy for classification decision
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels_ph, name="x_entr_for_cls"
    )
    xent = tf.reduce_mean(xent)
    return xent


def get_rewards(pred_labels, labels_ph):
    """Calucalate rewards based on ground truth and predicted labels."""
    # 0/1 reward.
    with tf.variable_scope('rewards'):
        reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
        rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
        rewards = tf.tile(rewards, (1, Config.num_glimpses))  # [b_sz,tsteps]
        reward = tf.reduce_mean(reward)
    return reward, rewards


def init_learning_rate(global_step, training_steps_per_epoch):
    """Initialise learning with exponential decay."""
    starter_learning_rate = Config.lr_start
    learning_rate = tf.train.exponential_decay(
        starter_learning_rate,
        global_step,
        training_steps_per_epoch,
        0.97,
        staircase=True)
    learning_rate = tf.maximum(learning_rate, Config.lr_min)
    return learning_rate


def add_summary(writer, summary_val, step, is_custom=False, tag=None):
    """Will add summary to the tensorboard.

    Summart of either custom value in case of evaluation, or summary value
    in case of the training process.
    """
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
    reward, loss, x_entr, logllratio, baseline_mse, summary_writer
):
    """Print the current state of the model.

    Including: step, duration of the step, reward value, loss, cross-entropy,
    loglikelihood ratio.
    """
    logging.info('Step {}. Step duration: {:.3f} sec.'.format(
        i, duration)
    )
    logging.info('learning_rate = {:3.6f}'.format(learning_rate))
    info = 'reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'
    logging.info(
        info.format(reward, loss, x_entr)
    )
    logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
        logllratio, baseline_mse)
    )
    add_summary(summary_writer, learning_rate, i, True, 'Learning rate value')
    add_summary(summary_writer, reward, i, True, 'Average reward')
    add_summary(summary_writer, loss, i, True, 'Hybrid loss value')
    add_summary(summary_writer, x_entr, i, True, 'Cross entropy')
    add_summary(summary_writer, logllratio, i, True, 'Log likelihood ratio')
    add_summary(
        summary_writer, baseline_mse, i, True, 'Baseline mean squared error'
    )



def run_training():
    """Run the model."""
    mnist = init_dataset()
    with tf.Graph().as_default():
        images_ph, labels_ph = placeholder_inputs()
        outputs = init_seq_rnn(images_ph, labels_ph)
        baselines = init_baseline_net(outputs)

        # Take the last step only.
        logits = inference(outputs[-1])
        softmax = tf.nn.softmax(logits)

        xent = cross_entropy_with_logits(logits, labels_ph)

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
        clipped_grads, _ = tf.clip_by_global_norm(grads, Config.max_grad_norm)
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

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(
            zip(clipped_grads, var_list), global_step=global_step
        )

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # TODO: Create a saver for writing training checkpoints.
        # saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()  # config=tf.ConfigProto(log_device_placement=T))

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(Config.log_dir, sess.graph)

        accuracy_op = accuracy(
            softmax, Config.batch_size, labels_ph
        )

        accuracy_summary = tf.summary.scalar('Batch_accuracy', accuracy_op)

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
                    xent_val, logllratio_val, baselines_mse_val, summary_writer
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
