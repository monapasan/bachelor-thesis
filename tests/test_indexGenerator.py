from .context import IndexGenerator
from nose.tools import raises
from tensorflow.examples.tutorials.mnist import input_data
mnist_raw = input_data.read_data_sets('../data/MNIST_data', one_hot=True)



images_per_sample = 5
class_amount = 3
noise_sizes = [4, 3, 2]
# myIndexGenerator.get_indexes(1) -->
# this should return an array of shape [1, class_amount, noise_sizes]
# [[1,5,3,1], [1,5,3], [3, 4]]

myIndexGenerator = IndexGenerator(images_per_sample, noise_sizes, class_amount)


def test_indexgenerator_length():
    length = len(list(myIndexGenerator.get_indexes(1)))
    assert length == 1

    length = len(list(myIndexGenerator.get_indexes(10)))
    assert length == 10


def test_indexgenerator_length_for_classes():
    length = len(list(myIndexGenerator.get_indexes_for_class(0)))
    assert length == 1

    length = len(list(myIndexGenerator.get_indexes_for_class(0, 9)))
    assert length == 9

    length = len(list(myIndexGenerator.get_indexes_for_class(1)))
    assert length == 1

    length = len(list(myIndexGenerator.get_indexes_for_class(1, 9)))
    assert length == 9

    length = len(list(myIndexGenerator.get_indexes_for_class(2)))
    assert length == 1

    length = len(list(myIndexGenerator.get_indexes_for_class(2, 9)))
    assert length == 9


def test_index_shape_for_classes():
    for cls_number, noise_size in enumerate(noise_sizes):
        assert_index_shape_for_class(cls_number, noise_size)


@raises(ValueError)
def test_index_out_of_range():
    return myIndexGenerator.get_indexes_for_class(3)


def assert_index_shape_for_class(cls_number, noise_size):
    for indexes in myIndexGenerator.get_indexes_for_class(cls_number):
        assert len(indexes) == noise_size

    for indexes in myIndexGenerator.get_indexes_for_class(cls_number, 10):
        assert len(indexes) == noise_size
