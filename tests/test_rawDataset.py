from .context import RawDataset
from .context import IndexGenerator
# from nose.tools import raises
from tensorflow.examples.tutorials.mnist import input_data
mnist_raw = input_data.read_data_sets('../data/MNIST_data', one_hot=True)


def init_indexGenerator():
    images_per_sample = 5
    class_amount = 3
    noise_sizes = [4, 3, 2]
    # myIndexGenerator.get_indexes(1) -->
    # this should return an array of shape [1, class_amount, noise_sizes]
    # [[1,5,3,1], [1,5,3], [3, 4]]
    return IndexGenerator(images_per_sample, noise_sizes, class_amount)


def init_raw_dataset():
    noise_label_index = [1, 2]
    data_label_index = [0]
    amount_of_classes = 3
    n_samples_per_class = [15000, 15000, 15000]
    noise_quantity = [4, 3, 2]
    sample_size = 5
    return RawDataset(
        init_indexGenerator(), mnist_raw.train, noise_label_index,
        data_label_index, amount_of_classes, noise_quantity,
        n_samples_per_class, sample_size
    )


myRawDataset = init_raw_dataset()


def test_rawdataset_length():
    assert myRawDataset.length == 15000


def test_get_next_batch_forClass():
    images, labels = myRawDataset.next_batch_for_class(1, 100)
    assert images.shape == (100, 5, 28 * 28)
    assert labels.shape == (100, 5, 10)

def test_get_next_batch():
    images, labels = myRawDataset.next_batch(10)
    assert images.shape == (10, 3, 5, 28 * 28)
    assert labels.shape == (10, 3, 5, 10)


# myRawDataset.next_batch(size = 1)

# [[size x images], [size x labels]]
# images  = [images_per_sample x image_size]
# labels = [images_per_sample x label_size]
# {labels: }[[image0,], [image], [image]]
