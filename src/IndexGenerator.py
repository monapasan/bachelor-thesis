from numpy.random import choice


def uniform(amount, size):
    return choice(amount, size, replace=False)


class IndexGenerator:
    """Responsible for generating indexes
    """
    def __init__(
        self, sample_size, noise_size, class_amount, keep_order=False,
        distribution=uniform, noise_index_order=None
    ):
        """Construct a new Index Generator.
        Args:
            distribution: function of distributions to samples noise indexes
             from.Only used when keep_order is false (default uniform)
            learning_rate: A `Tensor` or a floating point value.
            initial_accumulator_value: A floating point value.
        Raises:
            ValueError: If the `initial_accumulator_value` is invalid.
        """
        if(sample_size is None or noise_size is None):
            raise ValueError(
                'sample_size and noise_size should be passed to constructor'
            )
        if(
            not isinstance(noise_size, int) and
            (hasattr(noise_size, "__len__")
                and len(noise_size) != class_amount)
        ):
            raise ValueError(
                'noise_size should be either one-dimensional \
                or equal to amount of classes'
            )

        self.sample_size = sample_size
        self.noise_size = [noise_size] * class_amount if isinstance(
            noise_size, int) else noise_size
        self.class_amount = class_amount
        if(keep_order):
            self._build_generator_with_order(noise_index_order)
        else:
            self._build_generator(distribution)

    def _build_generator(self, distribution):
        self.distribution = distribution

        def generateOneSample():
            return [
                distribution(self.class_amount + 1, noise)
                for noise in self.noise_size
            ]
            # for i in range(class_amount):
            #     res[i] = distribution(self.class_amount, self.noise_size[i])
            # return res

        def generate(size):
            for i in range(size):
                yield generateOneSample()
        self._generate = generate

        def generate_for_class(class_number):
            if(class_number > self.class_amount):
                raise ValueError('cls can not be higher then class_amount')
            return distribution(
                self.class_amount + 1, self.noise_size[class_number]
            )
        self.generate_for_class = generate_for_class

    def _build_generator_with_order(self, noise_index_order):
        self.order = noise_index_order

        def generate():
            return None
        self._generator = generate

    def get_indexes(self, size=1):
        return self._generate(size)

    def get_indexes_for_class(self, cls_number, size=1):
        for i in range(size):
            yield self.generate_for_class(cls_number)


sample_size = 5
class_amount = 3
noise_size = [4, 3, 2]
myIndexGenerator = IndexGenerator(sample_size, noise_size, class_amount)
myIndexGenerator.get_indexes(1)
# myIndexGenerator.get_indexes(1) -->
# this should return an array of shape [1, class_amount, noise_size]
# [[1,5,3,1], [1,5,3], [3, 4]]
