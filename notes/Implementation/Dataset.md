# MNIST, Dataset
#bachelor/Implementation/dataset


> **The original black** and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio.   
> **The resulting images** contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.  
>   
> With some classification methods (particularly template-based methods, such as SVM and K-nearest neighbors), the error rate improves when the digits are centered by bounding box rather than center of mass. **If you do this kind of pre-processing, you should report it in your publications.**  
- - - -
 [1]


#### Points to keep in mind

The basic idea of experiment to prove whether model is capable of identifying pictures that do not influence on making classification decision. Therefore following things are considered:
* whether the dataset set has a defined order between images, i.e. that noise picture will always be placed at one certain index in an array.
* Noise picture variations (see paragraph below)


#### Noise picture variations

<u>At the moment there are two variations acknowledged:</u>
Considering that each sample from dataset consists of 5 images.

1. It's possible to make one certain number picture - a noise picture. For example, we can achieve this by making number 0 be always a noise picture and this case number 0 will be irrelevant for classification decision. An example of this can be as follows:
	* Class one - has three 1 digits (f.i. `[[1, 0, 1, 1, 0], [0, 0, 1, 1, 1]]`)
	* Class two - has three  2 digits  (f.i. `[[0, 2, 2, 0, 2], [2, 2, 2, 0, 0]]`)
2. Another possibility would be to make a random number(or numbers) from dataset a noise picture. Considering an example where we have data consisting of 5 pictures in a sample and the data is divided into three classes:
	* Class one - has exactly one zero number (f.i. [0, 5, 3, 1, 9])
	* Class two - has exactly two zero number (f.i. [7, 8, 1, 0, 0])
	* Class three - has exactly three zero number (f.i. [0, 8, 1, 0, 0])

First variation should be simpler for model to learn as the noise picture is always the same digit.

### Amount of data

Let's calculate how many example are we able to generate with MNIST dataset using the second variation.
For one sample we need 5 digits.  Depends on the class of the sample one, two and three of the digest should be zero with respect to first, second and third class.

As we don't know yet how many parameters(weights) our network will have, let's consider a dataset of size 45000 is good enough for our problem.
In 45000 examples we will have:
* ~15000 belongs to class 1
* ~15000 belongs to class 2
* ~15000 belongs to class 3

MNIST training dataset has 5444 examples of digit 0.

The good thing about the dataset, that we can use any non-zero digit in one class what gives us a good amount of samples:
* One sample of class 1 can consist of the same numbers but  have different digit representations, that is that we can have different variations of the samples. I.e. [0, 5, 3, 1, 9]  can occur two times but have different pictures of digits in it.



Running simple python script we can get a sense of how often certain digits appear in the dataset:
```python
"""
for one hot encoded dataset
"""
def get_amount_of_digit_in_dataset(dataset, number):
    return np.array([x for x in dataset if x[number] == 1.0]).shape[0]

for number in range(10):
    number_of_examples = get_amount_of_digit_in_dataset(mnist_raw.train.labels, number)
    print('MNIST training set has %i examples of digit %i' % (number_of_examples, number))

for number in range(10):
    number_of_examples = get_amount_of_digit_in_dataset(mnist_raw.test.labels, number)
    print('MNIST test set has %i examples of digit %i' % (number_of_examples, number))
```

The output is:
```
MNIST training set has 5444 examples of digit 0
MNIST training set has 6179 examples of digit 1
MNIST training set has 5470 examples of digit 2
MNIST training set has 5638 examples of digit 3
MNIST training set has 5307 examples of digit 4
MNIST training set has 4987 examples of digit 5
MNIST training set has 5417 examples of digit 6
MNIST training set has 5715 examples of digit 7
MNIST training set has 5389 examples of digit 8
MNIST training set has 5454 examples of digit 9

MNIST test set has 980 examples of digit 0
MNIST test set has 1135 examples of digit 1
MNIST test set has 1032 examples of digit 2
MNIST test set has 1010 examples of digit 3
MNIST test set has 982 examples of digit 4
MNIST test set has 892 examples of digit 5
MNIST test set has 958 examples of digit 6
MNIST test set has 1028 examples of digit 7
MNIST test set has 974 examples of digit 8
MNIST test set has 1009 examples of digit 9
```


## Generation of the data


### Generation procedure

Requirements:
* It's not desired to have two identical pictures in the dataset
	* therefore we need to keep id of pictures, to prevent duplicated samples. or
	* build a dataset in such a way that the probability of having two duplicated samples across new dataset is 0.

There is a good variety of approaches that can be used to create a dataset described above.

This work will use the simplest approach which is described below.

To start with generation of the dataset, we first need to separate the MNIST dataset into two datasets: **zero digit dataset** and **non zero digit dataset**.

#### Let's consider first the training set.
Zero digit dataset consist of **5444** examples.
Non-zero digit dataset consist of `55000-5444` = **49556**

##### Building first class
To generate 15000 samples for the first class:
1. we need to have 15000 samples of zero digit
	* As the size of our zero digit dataset is 5444, we need to use same zero digit about 3 times(`15000/5444 = 2.775 ~= 3`).
	* Therefore the same zero digit can occur in two-three sample of the new dataset.
	* This is only applicable for the first class.
2. we need to have 15000 unordered 4-tuples of <u>non-zero digits</u>.
	* taking into account that we don't want to have two identical samples we have to make sure that we don't have identical non-zero digits in a sample with same zero digits.
		* or better to not have any identical sample at all
	* To achieve that we can build 15000 identical unordered 4-tuples from non-zero digit dataset.
	* Good news are that from 49556 samples in our non-zero digit dataset we can extract a huge number of combinations of size 4 (it's about 2.5E+17).
		* Counting combinations: from 49556 samples we want to build combinations of size 4:
		* using the formula: `n_C_r = n! / r! (n - r)!` n_C_r = 49556! / 4! (49556 - 4)! ~=2.5E+17
	* we can build at least 15000 combinations of 4-tuples from non-zero digits dataset (noise dataset), which is leading to non identical samples.
3. We concat the combinations from step 2 with zero digit from step 1.
4. Indexes of resulted array will be explained below.


##### Building second class
To generate 15000 samples for the second class where one sample should contain two zero digits in a sample:
1.  we need to have 15000 zero digit tuples.
	* In order to prevent that model can learn any dependencies between digits it's would be great to not use the same pair of zero digits in dataset more than one time.
	* Since number of combinations of size 2 from 5444 samples is equal to 14.815.846 we can easily gather 15000 non duplicated zero digit pairs.
		* Counting combinations: from 5444 sample we want to build combinations of size 2:
		* using the formula: `n_C_r = n! / r! (n - r)!` n_C_r = 5444! / 2! (5444 - 2)! = 5443 * 5444 / 2 = 14.815.846
	* building the these  combinations, will give us 15000 non-duplicated zero digit pairs.
2. We also need to have 15000 unordered triples of non-zero digits
	* As we have 49556 in our non-zero digit dataset, and 15000 x 3 = 45000 which less than 49556
	* To build the amount of triples it should be enough:
		* To randomly rearrange digits in the non-zero digit dataset
		* and just take sequentially three digit at a time based on a new order of non-zero digit dataset.
3. We concat the combinations from step 2 with zero digits from step 1.
4. Indexes of resulted array will be explained below.



##### Building third class

To generate 15000 samples for the third class where one sample should contain three zero digits in a sample:
1. we need to have 15000 unordered zero digit triples.
	* In order to prevent that model can learn any dependencies between digits it's would be great to not use the same pair of zero digits in dataset more than one time.
	* Since number of combinations of size 2 from 5444 samples is equal to 26,875,944,644 we can easily gather 15000 unordered non duplicated zero digit triples.
		* Counting combinations: from 5444 sample we want to build combinations of size 3:
		* using the formula: `n_C_r = n! / r! (n - r)!`.  n_C_r = 5444! / 3! (5444 - 3!) = 5442 * 5443 * 5444 / 6 = 26.875.944.644
	* building the these  combinations, will give us 15000 unordered non-duplicated zero digit triples.
2. We also need to have 15000 unordered pairs non-zero digits
	* As we have 49556 in our non-zero digit dataset, and 15000 x 2 = 30000 which less than 49556
	* To build the amount of triples it should be enough:
		* To randomly rearrange digits in the non-zero digit dataset
		* and just take sequentially two digit at a time based on a new order of non-zero digit dataset.
3. We concat the combinations from step 2 with zero digits from step 1.
4. Indexes of resulted array will be explained below.

#### Choosing the indexes of zero digit and noise pictures

As already mentioned above, we need to prove whether the order of noise pictures in a sample will make a difference for performance of the model.
We have two major possibilities to place our noise pictures in a sample:

1. with fixed place:
	* always have zero digit at certain place
	* we can put indexes at start of the array or at end of the array
	* Make experiments and investigate how the model will behave in these different cases.
	* in case of 2nd and 3rd classes we must have multiple indexes.
2. with random place:
	* we can sample indexes for zero digits from uniform distribution.
	* depends on the class we can sample multiple indexes.

### Implementation procedure

Steps to be done:
1. separate MNIST dataset into non-zero and zero ones
2. build combinations from new datasets.
3. with respect to the class take appropriate number of zero and non-zero digits
	1. first class - 1 zero and 4 non-zeros
	2. second class  - 2 zeros and 3 non-zeros
	3. third class  - 3 zeros and 2 non-zeros
4. choose the indexes according to points mentioned above
5. place taken digits into an array with respect to indexes


## Classes overview
The approach to generate the new dataset can be implemented by using two classes:
1. IndexGenerator
	* responsible for generating indexes
	* maybe split this one into three classes:
		* BaseIndexGenerator - abstract class
		* OrderedIndexGenerator - class when order is the same
		* UnorderedIndexGenerator - class when order is taken from distribution.
2. RawDataset
	* the dataset which holds the raw data
	* can accept index_generator class as parameter.

We can also create a class for convenient use in our model:
* Dataset
	* the dataset which holds tf tensors.
	* basically a wrapper for convenient use in out model.
	* has Raw_dataset as property.
		* or also as parameter


### IndexGenerator class

<u>Properties</u>:
* `keep_order` - whether order should be saved for all samples.
* `distribution` - distributions to use  when order only used when keep_order is false (default uniform)
* `noise_index_order` - order of noise (like random, start, end), only if keep_order is true. Maybe better to use a list of indexes.
* `sample_size` - the size of one sample( in the case above is 5)
* `noise_size` - the size of noise sample (in case of the first class is 4)
* `number_classes` - number of classes in dataset(in above case is 3)
	* if `number_classes` is more than 1 then all of parameters above can be passed as array.
	* when the props above is not arrays, then it's assumed that all properties are the same for all classes.

<u>Methods</u>:
* `get_indexes(size:int=1)`
	* will return an array of size `(size, number_classes, noise_size)`
		* or will return an array of size `(size, number_classes, sample_size)`. In this case we can make it one hot encoded.
* `get_indexes_for_class(class, size=1)`
	* will give you next batch of index with respect to class `class: int`
	* will return an array of size  `(size, noise_size)`

### RawDataset class

Constructor should accept following:
	* `IndexGenerator` - class described above
	* `dataset` - accept the dataset (MNIST)
		* pythonic way: dataset should have properties `images` and `labels`
	* `distribution` -
	* `noise_label_index` - noise label is expected at this index. All other indexes will be considered as places from which actual information comes from. One either specifies `noise_label_index` or `data_label_index`. These properties are mutually exclusive.
	* `data_label_index` - data label is expected at this index. Indexes of labels where actual information comes from.
	* `amount_of_classes` - amount of classes
	* `noise_quantity` -  amount of noise labels that should be putting into each of the class. Should be an array of size `amount_of_classes`. This equation should be fulfilled: `amount_of_classes >noise_size`
	* `amount_of_samples_in_class` - amount of sample in each of the class. Should be an array of size `amount_of_classes`.(take a look on class 3, maybe some technics)

Public properties:
* `images # /groups` -- array of groups of images.
* `labels` -- array of labels

Methods:
* `next_batch(size): (images, labels)` -
* `permute` - permute the order.

### Dataset class, work in progress

As for convenience sake public methods of the Dataset class should accept and return only Tensorflow's tensors. It will provide a good level of abstraction and make Tensorflow's tensors a first-class citizen of the prototype. Having the tensors first-class citizen will contribute to construction of coherent prototype.

Additional requirement to dataset are:
* This gonna be a bit complicated.

TensorFlow provide a class for importing the [MNIST dataset ](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).  We build our Dataset class upon the one from tensor flow.

Dataset class methods:
* WIP
Dataset class properties:
* WIP

## Utilities
* `itertools.combinations(iterable, r)` - nice way of building combinations. Does not build it by calling, but will return a generator.
* `numpy.random.uniform` - sampling from uniform distribution, provided by bumpy
* `numpy.random. randint` - sampling discrete values from uniform distribution
* `numpy.random.permutation` - or rather use this for distribution
* `np.random.choice` - this is will be equivalent to above
- - - -
[1] - MNIST handwritten digit database, Yann LeCun, Corinna Cortes and Chris Burges. (n.d.). Retrieved April 12, 2017, from http://yann.lecun.com/exdb/mnist/






### Questions:
* What are good metrics that one has a good dataset? How check for correlation and dependencies between data in the dataset?
* What will be good amount of samples? How is that dependent on weight variables?
* double check distribution of numbers in dataset
* As we do some random permutation on the dataset, does it make sense to store the dataset to have the same dataset at every training of the model? Or would it better to generate a new one per training?
* PEP 008 recommends using camelCase naming for methods and  properties names. But I have a feeling that this convention is not widely used. For instance, If one looks at [this](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py) it seems like tensorFlow guys for example don't follow it.

- - - -











Garbage:





1. take the digits based on order in dataset:
	* e.g. we have an array [4,1,4,5,1,3,6,8,1,9,4,1].
	* first we will take [4,1,4,5], then [1,3,6,8] and etc.
	* this will give us


With current approach we can easily build 15000 samples for the first class.

* firstly, we can take zero digits from the zero digit dataset based on the order of this dataset.
* This we give us about

To generate samples for the first class:
* take one zero digit from zero digit dataset
* take four digits from non-zero digit dataset
* build an array
	* choose indexes (see below)
