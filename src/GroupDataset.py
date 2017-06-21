from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IndexGenerator import IndexGenerator
from DummyDataset import DummyDataset
import itertools
import numpy as np


class GroupDataset(object):
    """The Dataset class where each sample holds MNIST samples in a group."""

    def __init__(
        self, index_generator, dataset, noise_label_index,
        data_label_index, amount_of_classes,
        noise_quantity, n_samples, sample_size
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
            noise_quantity(list):  amount of noise labels per sample that
                each of the class should have. Should be an array of size
                `amount_of_classes`. This equation should be fulfilled:
                all(amount_of_classes > noise_n for noise_n in noise_quantity)
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
                'dataset should have properites images and labels'
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
        if(max(noise_quantity) > sample_size):
            raise ValueError(
                'noise_quantity should be less than amount of \
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
        self.noises_per_class = noise_quantity
        self.sample_size = sample_size
        self._divide_dataset()
        self._build_groups()
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def length(self):
        """Return the length of the dataset."""
        return self.length


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
        # new_indexed_train_images = [
        #     (i, image) for i, image in enumerate(self.__dataset.images)
        # ]
        self._images = [None] * self.amount_of_classes
        self._labels = [None] * self.amount_of_classes
        for i in range(self.amount_of_classes):
            self._build_group_for_class(i)
        self._images = np.array(self._images)
        self._labels = np.array(self._labels)
        self.length = self._labels.shape[1]

    def _build_group_for_class(self, class_number):
        n_noise_per_class = self.noises_per_class[class_number]
        n_data_per_class = self.sample_size - n_noise_per_class
        n_samples = self.n_samples_per_class[class_number]
        self._images[class_number] = []
        self._labels[class_number] = []
        # self._permute_data()
        noise_combinations_indicices = itertools.combinations(
            range(self.__noise_data.length()), n_noise_per_class
        )
        data_combinations_indicices = itertools.combinations(
            range(self.__information_data.length()), n_data_per_class
        )
        # if(n_data_per_class == 1):
        #     self.__information_data.permute()
        for i in range(n_samples):
            group_label = []
            group_image = []
            try:
                combination_noise_i = next(noise_combinations_indicices)
                combinations_date_i = next(data_combinations_indicices)
            except StopIteration:
                self._permute_data()
                noise_combinations_indicices = itertools.combinations(
                    range(self.__noise_data.length()), n_noise_per_class
                )
                data_combinations_indicices = itertools.combinations(
                    range(self.__information_data.length()), n_data_per_class
                )
                combination_noise_i = next(noise_combinations_indicices)
                combinations_date_i = next(data_combinations_indicices)

            noise_images, noise_labels = self.__noise_data.get(
                combination_noise_i
            )

            info_images, info_labels = self.__information_data.get(
                combinations_date_i
            )
            # for combinations_i in noise_combinations_indicices:
            # noises = self.__noise_data.get(combinations_i)
            # self._images[class_number].append()
            # for combinations_i in data_combinations_indicices:
            # information = self.__information_data.get(combinations_i)
            # raise ValueError('SHIT')
            noise_indexes = self.index_generator.get_indexes_for_class(
                class_number
            )[0]
            # information_indexes = [
            #     i for i in range(self.sample_size) if i not in noise_indexes
            # ]
            for j in range(self.sample_size):
                if(j in noise_indexes):
                    group_image.append(noise_images.pop())
                    group_label.append(noise_labels.pop())
                if(j not in noise_indexes):
                    group_image.append(info_images.pop())
                    group_label.append(info_labels.pop())
            self._images[class_number].append(group_image)
            self._labels[class_number].append(group_label)
        # TODO: permute images and labels for classes

    def _permute_data(self):
        self.__noise_data.permute()
        self.__information_data.permute()

    def next_batch_for_class(self, cls_number, batch_size):
        """Return the next `batch_size` examples of class `cls_number`."""
        images = self._images[cls_number]
        labels = self._labels[cls_number]
        start = self._index_in_epoch
        if start + batch_size > self.length:
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.length - start
            images_rest_part = images[start:self.length]
            labels_rest_part = labels[start:self.length]
            # TODO: permute data
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = images[start:end]
            labels_new_part = labels[start:end]
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
            return images[start:end], labels[start:end]

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        # TODO: permute data
        # if self._epochs_completed == 0 and start == 0 and shuffle:
        #     self.permute()
        # Go to the next epoch
        if start + batch_size > self.length:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.length - start
            images_rest_part = self._images[start:self.length]
            labels_rest_part = self._labels[start:self.length]
            # # Shuffle the data
            # TODO: permute data
            # if shuffle:
            #     self.permute()
            # Start next epoch
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
