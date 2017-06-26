"""This module provides provides a dataset for the extension of RAM."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .DummyDataset import DummyDataset
from .IndexGenerator import IndexGenerator
import itertools
import numpy as np


class GroupDataset(object):
    """The Dataset class where each sample holds MNIST samples in a group."""

    def __init__(
        self, index_generator, dataset, noise_label_index,
        data_label_index, amount_of_classes,
        noises_per_class, n_samples, sample_size
    ):
        """Construct a new Index Generator.

        Args:
            index_generator(IndexGenerator): index generator
            dataset(object): accept the dataset (MNIST)
                pythonic way: dataset should have properties:
                `images` and `labels`
            noise_label_index(list): noise label is expected at this index.
                All other indexes will be considered as places where actual
                information comes from.One either specifies `noise_label_index`
                or `data_label_index`. These properties are mutually exclusive.
            data_label_index(list): data label is expected at this index.
                Indexes of labels where relevant for the task
                information comes from.
            amount_of_classes(int): amount of classes the dataset should have
            noises_per_class(list):  amount of noise labels per sample that
                each of the class should have. Should be an array of size
                `amount_of_classes`. This equation should be fulfilled:
                all(
                    amount_of_classes > noise_n for noise_n in noises_per_class
                )
            n_samples(int): amount of sample in each of the class.
                Should be an array of size `amount_of_classes` or integer
                if the amount is the same acroll all classes.
            sample_size(int): amount of pictures in one
                sample(i.e. size of group).
        Raises:
            ValueError:
        """
        if(not(hasattr(dataset, "labels") and hasattr(dataset, "images"))):
            raise ValueError(
                'dataset object should have properties: `images` and `labels`'
            )

        if(not isinstance(index_generator, IndexGenerator)):
            raise ValueError(
                'index_generator should be an instance of IndexGenerator class'
            )

        if(
            len(set(noise_label_index)) != len(noise_label_index) or
            max(noise_label_index) >= amount_of_classes
        ):
            raise ValueError(
                'noise_label_index should not have duplicates. \
                Noise indexes is out of range'
            )

        if(max(noises_per_class) > sample_size):
            raise ValueError(
                'noises_per_class should be less than amount of \
                 classes(amount_of_classes)'
            )
        if(
            isinstance(n_samples, int) or
            len(n_samples) != amount_of_classes
        ):
            raise ValueError(
                'n_samples should either be an int or fullfil \
                 len(n_samples) == amount_of_classes'
            )
        self.__dataset = dataset
        self.index_generator = index_generator
        if(isinstance(n_samples, int)):
            self.n_samples_per_class = (
                [n_samples] * amount_of_classes
            )
        else:
            self.n_samples_per_class = n_samples

        self.noise_label_index = noise_label_index
        self.data_label_index = data_label_index

        self.amount_of_classes = amount_of_classes
        self.noises_per_class = noises_per_class
        self.sample_size = sample_size
        self._divide_dataset()
        self._build_groups()
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def epochs_completed(self):
        """Return amount of epoch completed of the dataset."""
        return self._epochs_completed

    @property
    def images(self):
        """Return the images of the dataset."""
        return self._images

    @property
    def labels(self):
        """Return the labels of the dataset."""
        return self._labels

    def _divide_dataset(self):
        # we need to extract two dataset from one which comes from
        # constructor. These properites should be something like
        # noise_data ; .labels, .images
        # information_dataset ; .labels, .images
        self.__noise_data = DummyDataset()
        self.__information_data = DummyDataset()
        for image, label in zip(self.__dataset.images, self.__dataset.labels):
            if(np.argmax(label) in self.noise_label_index):
                self.__noise_data.add_sample(image, label)
            if(np.argmax(label) in self.data_label_index):
                self.__information_data.add_sample(image, label)

    def _build_groups(self):
        self._images = []
        self._labels = []
        for i in range(self.amount_of_classes):
            self._build_group_for_class(i)
        self._images = np.array(self._images)
        self._labels = np.array(self._labels)
        self.length = self._labels.shape[0]
        self.permute()

    def _build_group_for_class(self, class_number):
        n_samples = self.n_samples_per_class[class_number]
        class_label = self._build_label_for_class(class_number)
        noise_comb, data_comb = self._build_combinations(class_number)
        for i in range(n_samples):
            group = []
            try:
                combination_noise_i = next(noise_comb)
                combinations_data_i = next(data_comb)
            except StopIteration:
                noise_comb, data_comb = self._build_combinations_and_permute(
                    class_number
                )
                combination_noise_i = next(noise_comb)
                combinations_data_i = next(data_comb)

            noise_images, noise_labels = self.__noise_data.get(
                combination_noise_i
            )

            info_images, info_labels = self.__information_data.get(
                combinations_data_i
            )
            noise_indexes = self.index_generator.get_indexes_for_class(
                class_number
            )[0]
            for j in range(self.sample_size):
                if(j in noise_indexes):
                    group.append(noise_images.pop())
                if(j not in noise_indexes):
                    group.append(info_images.pop())
            self._images.append(group)
            self._labels.append(class_label)

    def _permute_datasets(self):
        self.__noise_data.permute()
        self.__information_data.permute()

    def _build_label_for_class(self, cls_n):
        class_label = np.zeros((1, self.amount_of_classes))
        class_label[np.arange(1), [cls_n]] = 1
        return class_label

    def _build_combinations(self, cls_n):
            n_noise_per_class = self.noises_per_class[cls_n]
            n_data_per_class = self.sample_size - n_noise_per_class
            noise_combinations_indicices = itertools.combinations(
                range(self.__noise_data.length), n_noise_per_class
            )
            data_combinations_indicices = itertools.combinations(
                range(self.__information_data.length), n_data_per_class
            )
            return noise_combinations_indicices, data_combinations_indicices

    def _build_combinations_and_permute(self, cls_n):
        self._permute_datasets()
        return self._build_combinations(cls_n)

    def next_batch(self, batch_size):
        """Get next batch of size `batch_size`."""
        start = self._index_in_epoch
        if start + batch_size > self.length:
            self._epochs_completed += 1
            rest_num_examples = self.length - start
            images_rest_part = self._images[start:self.length]
            labels_rest_part = self._labels[start:self.length]
            self.permute()
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]

            result_images = np.concatenate(
                (images_rest_part, images_new_part), axis=0
            )
            result_labels = np.concatenate(
                (labels_rest_part, labels_new_part), axis=0
            )
            return result_images, result_labels
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

    def permute(self):
        """Permute the dataset."""
        perm0 = np.arange(self.length)
        np.random.shuffle(perm0)
        self._images = self._images[perm0]
        self._labels = self._labels[perm0]
        return self
