# MNIST, Dataset
#bachelor/Implementation/dataset

**The original black** and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio.
**The resulting images** contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.


With some classification methods (particularly template-based methods, such as SVM and K-nearest neighbors), the error rate improves when the digits are centered by bounding box rather than center of mass. **If you do this kind of pre-processing, you should report it in your publications.** [1]


#### Points to keep in mind

The basic idea of experiment to prove whether model is capable of identifying pictures that do not influence on making classification decision. Therefore following things are considered:
	* whether the dataset set has a defined order between images, i.e. that noise picture will always be placed at one certain index in an array.
	* Noise picture variations (see paragraph below)


#### Noise picture variations

<u>So far we acknowledged two variations:</u>
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

MNIST training dataset has **5444** examples of digit 0.

The good thing about the dataset, that we can use any non-zero digit in one class what gives us a good amount of samples:
* One sample of class 1 can consist of the same numbers but  have different digit representations, that is that we can have different variations of the samples.
* I.e. [0, 5, 3, 1, 9]  can occur two times but have different pictures of digits in it.



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
	* Good news are that having 49556 samples in our non-zero digit dataset gives us a huge number of combinations of size 4 (it's about 2.5E+17).
	* we can build 15000 combination of size 4 from non-digits dataset
	* and use these combinations.


##### Building second class
To generate 15000 samples for the second class where one sample should contain two zero digits in a sample:
1.  we need to have 15000 zero digit pair.
	* They shouldn't be identical as to prevent duplicates in the dataset we can use different  non-zero digits.
	* but to completely prevent that model can learn any dependencies between digits it's desired to not use the same pair of zero digits in dataset.
	* And since number of combinations is equal to 14.815.846 we can easily achieve 15000 non duplicated zero digit pairs.
		* Counting combinations: from 5444 sample we want to build combinations of size 2:
		* using the formula: `n_C_r = n! / r! (n - r)!` n_C_r = 5444! / 2! (5444 - 2)! = 5443 * 5444 / 2 = 14.815.846
	* building the array of combinations and sampling from it, will give us 15000 non-duplicated zero digit pairs.
2. We also need 15000 unordered triples non-zero digits
	* As we have 49556 in our non-zero digit dataset, and 15000 x 3 = 45000 which less than 49556
	* To build the amount of triples it should be enough:
		* To randomly rearrange digits in the non-zero digit dataset
		* and just take sequentially three digit at a time based on a new order of non-zero digit dataset.



##### Building third class

To generate 15000 samples for the third class where one sample should contain three zero digits in a sample:
1. we need to have 15000 unordered zero digit triples.
	* They shouldn't be identical as to prevent duplicates in the dataset we can use different  non-zero digits.
	* but to completely prevent that model can learn any dependencies between digits it's desired to not use the same pair of zero digits in dataset.
	* And since number of combinations is equal to 26,875,944,644 we can easily achieve 15000 non duplicated zero digit triples.
		* Counting combinations: from 5444 sample we want to build combinations of size 3:
		* using the formula: `n_C_r = n! / r! (n - r)!`.  n_C_r = 5444! / 3! (5444 - 3!) = 5442 * 5443 * 5444 / 6 = 26.875.944.644
	* building the array of combinations and sampling from it, will give us 15000 unordered set of non-duplicated zero digit triples.
2. We also need 15000 unordered pairs non-zero digits
	* As we have 49556 in our non-zero digit dataset, and 15000 x 2 = 30000 which less than 49556
	* To build the amount of triples it should be enough:
		* To randomly rearrange digits in the non-zero digit dataset
		* and just take sequentially two digit at a time based on a new order of non-zero digit dataset.


#### Choosing the indexes of zero digit and noise pictures

As already mentioned above, we need to prove whether the order of noise pictures in a sample will make a difference for performance of the model.
We have two major possibilities to place our noise pictures in a sample:

1. with fixed place:
	* always have zero digit at certain place
	* we can put indexes at start of the array or at end of the array
	* Make experiments and investigate how the model will behave in these different cases.
	* in case of 2 and 3 classes we must have multiple indexes.
2. with random place:
	* we can sample indexes of zero from uniform distribution.
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
I see here three classes
* Index_generator
	* responsible for generating indexes
* Raw_dataset
	* the dataset which holds the raw data
	* can accept index_generator class as parameter.
* Dataset
	* the dataset which holds tf tensors
	* has Raw_dataset as property.
		* or also as parameter

Index generator:
	* 	keep_order ? - wether order should be


### Index_generator class

### Raw_dataset class

### Dataset class

As for convenience sake public methods of the Dataset class should accept and return only Tensorflow's tensors. It will provide a good level of abstraction and make Tensorflow's tensors a first-class citizen of the prototype. Having the tensors first-class citizen will contribute to construction of coherent prototype.

Additional requirement to dataset are:


TensorFlow provide a class for importing the [MNIST dataset ](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).  We build our Dataset class upon the one from tensor flow.

Dataset class methods:
	*
Dataset class properties:
	*

- - - -
[1] - MNIST handwritten digit database, Yann LeCun, Corinna Cortes and Chris Burges. (n.d.). Retrieved April 12, 2017, from http://yann.lecun.com/exdb/mnist/



Questions:
* How to check whether data make sense?
* What is good amount of samples?
* double check distribution of number in dataset
* As we do some random permutation on the dataset, does it make sense to store the dataset to have the same dataset?

---

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
