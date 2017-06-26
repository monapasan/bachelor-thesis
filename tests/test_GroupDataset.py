from .context import GroupDataset
from .context import IndexGenerator
# from nose.tools import raises
from tensorflow.examples.tutorials.mnist import input_data
mnist_raw = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

MNIST_size = 28
images_per_sample = 5
amount_of_classes = 3
MNIST_classes_n = 10


def init_indexGenerator():
    """Initilise IndexGenerator for testing purposes."""
    class_amount = 3
    noise_sizes = [4, 3, 2]
    # myIndexGenerator.get_indexes(1) -->
    # this should return an array of shape [1, class_amount, noise_sizes]
    # [[1,5,3,1], [1,5,3], [3, 4]]
    return IndexGenerator(noise_sizes, class_amount)


def init_raw_dataset():
    """Initilise the GroupDataset for testing purposes."""
    noise_label_index = [1, 2]
    # data_label_index = [0]
    data_label_index = [0, 3, 4, 5, 6, 7, 8, 9]
    n_samples_per_class = [15000, 15000, 15000]
    noise_quantity = [4, 3, 2]
    return GroupDataset(
        init_indexGenerator(), mnist_raw.train, noise_label_index,
        data_label_index, amount_of_classes, noise_quantity,
        n_samples_per_class, images_per_sample
    )


myGroupDataset = init_raw_dataset()


def test_images_shape():
    """Return expected shape of images."""
    assert myGroupDataset.images.shape == (45000, 5, 784)


def test_labels_shape():
    """Return expected shape of labels."""
    assert myGroupDataset.labels.shape == (45000, 1, 3)


def test_rawdataset_length():
    """Test whether the GroupDataset has the expected length."""
    assert myGroupDataset.length == 43000


def test_get_next_batch_forClass():
    """Test the function `next_batch`.

    As GroupDataset is use uniform distribution to choose the indexes
    of noise images, we check only the expected shape of return value.
    """
    size = 100
    images, labels = myGroupDataset.next_batch(size)
    assert images.shape == (size, 5, 784)
    assert labels.shape == (size, 1, 3)
    # images, labels = myGroupDataset.next_batch_for_class(class_n, size)
    # assert images.shape == (size, images_per_sample, MNIST_size * MNIST_size)
    # assert labels.shape == (size, images_per_sample, MNIST_classes_n)


# def test_get_next_batch():
#     """Tesh the function `next_batch` of the GroupDataset instance.
#
#     As GroupDataset is use uniform distribution to choose the indexes
#     of noise images, we check only the expected shape of return value.
#     """
#     size = 10
#     images, labels = myGroupDataset.next_batch(size)
#     assert images.shape == (
#         size, amount_of_classes, images_per_sample, MNIST_size * MNIST_size
#     )
#     assert labels.shape == (
#         size, amount_of_classes, images_per_sample, MNIST_classes_n
#     )


# myGroupDataset.next_batch(size = 1)

# [[size x images], [size x labels]]
# images  = [images_per_sample x image_size]
# labels = [images_per_sample x label_size]
# {labels: }[[image0,], [image], [image]]
