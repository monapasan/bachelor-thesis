"""This module provides indexes of noise data fot the GroupDataset."""
from numpy.random import choice


def _uniform(amount, size):
    return choice(amount, size, replace=False)


class IndexGenerator(object):
    """Responsible for generating indexes.

    The class generates indexes for 'cls_amount' classes.
    Amount of noise indexes is depends on 'noise_size'.
    When  keep_order is 'True' then generated indexes
    will be the same. That is, get_indexes will be
    determenistic function.
    """

    def __init__(
        self, noise_size, cls_amount, keep_order=False,
        distribution=_uniform, noise_index_order=None
    ):
        """Construct a new Index Generator.

        Args:
             noise_size(list) : should be array of size 'cls_amount'. Amount
                of noise data that one sample should have per class.
             cls_amount(int) : amount of classes the
                dataset should have.
             keep_order(Boolean) : if 'True' then order will be the same across
                all samples, if 'False' then indexes will be generated by using
                'disrtibution' function(default to False).
            distribution(int, int => list) : function of distribution
                to sample indexes where noise data
                should be placed at.
                Only used when keep_order is false (default uniform)
             noise_index_order(list) : when keep_order= True, then
                 'noise_index_order; is used as indexes where noise data
                 should be placed at.
        Raises:
            ValueError: If the `noise_size` is not one-dimensional array or
                if no noise_size was
                passed to the contructor.
        """
        if(noise_size is None):
            raise ValueError(
                'noise_size \
                should be passed to the constructor'
            )
        if(
            not isinstance(noise_size, int) and
            (hasattr(noise_size, "__len__")
                and len(noise_size) != cls_amount)
        ):
            raise ValueError(
                'noise_size should be either one-dimensional \
                or equal to amount of classes'
            )

        self.noise_size = [noise_size] * cls_amount if isinstance(
            noise_size, int) else noise_size
        self.cls_amount = cls_amount
        if(keep_order):
            self._build_generator_with_order(noise_index_order)
        else:
            self._build_generator(distribution)

    def _build_generator(self, distribution):
        self.distribution = distribution

        def generateOneSample():
            return [
                distribution(self.cls_amount + 1, noise)
                for noise in self.noise_size
            ]

        def generate(size):
            return [generateOneSample() for i in range(size)]
            # for i in range(size):
            #     yield generateOneSample()
        self._generate = generate

        def generate_for_class(class_number):
            return distribution(
                self.cls_amount + 1, self.noise_size[class_number]
            )
        self.__generate_for_class = generate_for_class

    def _build_generator_with_order(self, noise_index_order):
        self._order = noise_index_order
        self._check_order()

        def generate():
            return self._order

        def generate_for_class(class_number):
            return self._order[class_number]

        self._generate = generate
        self.__generate_for_class = generate_for_class

    def _check_order(self):
        for i, order in enumerate(self._order):
            if(len(order) == self.noise_size[i]):
                raise ValueError(
                    'The shape of the indexes object does not match with \
                     shape of noise_size'
                )

    def get_indexes(self, size=1):
        """Generate indexes for noise data of the size=sise for all classes.

        Args:
            size(int): amount of samples to generate.
        """
        return self._generate(size)

    def get_indexes_for_class(self, cls_number, size=1):
        """Generate indexes for noise data of the size with respect to the class.

        Args:
            cls_number(int): indexes is generated only for this class.
                Can't be bigger than 'cls_amount'
            size(int): amount of samples to generate.
        """
        if(cls_number >= self.cls_amount):
            raise ValueError(
                'Index is out of range. \
                cls parameter can not be higher then cls_amount: %i %i' % (
                    cls_number, self.cls_amount
                )
            )
        return [self.__generate_for_class(cls_number) for i in range(size)]
        # for i in range(size):
        #     yield self.__generate_for_class(cls_number)
