"""Implemenation of the class that holds images and labels of that images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
# import numpy as np


class DummyDataset(object):
    """Dummy Class for holding data."""

    def __init__(self, images=[], labels=[]):
        """Construct the dataset class.

        Args:
            images(list): images to store in the dataset.
            labels(list): labels to store in the dataset.
        """
        self._images = images
        self._labels = labels

    @property
    def images(self):
        """Return the images of the dataset."""
        return self._images

    @property
    def labels(self):
        """Return the labels of the dataset."""
        return self._labels

    @property
    def length(self):
        """Return length of the dataset."""
        assert len(self._images) == len(self._labels), (
            'len(images): %s len(labels): %s' % (
                len(self.images), len(self.labels)
            )
        )
        return len(self._images)

    def add_sample(self, image, label):
        """Add the image and label pair to the dataset."""
        self._images.append(image)
        self._labels.append(label)
        return self

    def permute(self):
        """Permute the dataset."""
        combined = list(zip(self._images, self._labels))
        random.shuffle(combined)
        self._images[:], self._labels[:] = zip(*combined)
        return self

    def get(self, i):
        """Return i-th sample if i is integer, otherwise list of samples."""
        # TODO: check if an array
        return (self.get_images(i), self.get_labels(i))
        # if(isinstance(i, int)):
        #     return self._images[i], self._labels[i]
        # else:
        #     return [(self.get_images[j], self.get_labels(j)) for j in i]

    def get_images(self, i):
        """Get i-th image if i is integer, otherwise list of images."""
        if(isinstance(i, int)):
            return self._images[i]
        else:
            return [self._images[j] for j in i]

    def get_labels(self, i):
        """Get i-th label if i is integer, otherwise list of labels."""
        if(isinstance(i, int)):
            return self._labels[i]
        else:
            return [self._labels[j] for j in i]
