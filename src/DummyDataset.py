from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random


class DummyDataset:
    def __init__(self):
        self._images = []
        self._labels = []

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    # @property
    # def length(self):
    #     return self.length

    def permute_dataset(self):
        combined = list(zip(self._images, self._labels))
        random.shuffle(combined)
        self._images[:], self._labels[:] = zip(*combined)

    def get(self, i):
        if(isinstance(i, int)):
            return self._images[i], self._labels[i]
        else:
            return [(self._images[i], self._labels[i]) for j in i]

    def get_images(self, i):
        if(isinstance(i, int)):
            return self._images[i]
        else:
            return [self._images[j] for j in i]

    def get_labels(self, i):
        if(isinstance(i, int)):
            return self._labels[i]
        else:
            return [self._labels[j] for j in i]

    def length(self):
        assert len(self._images) == len(self._labels), (
            'len(images): %s len(labels): %s' % (
                len(self.images), len(self.labels)
            )
        )
        return len(self._images)
