# RAM

Modified from https://github.com/jlindsey15/RAM

Implementation of "Recurrent Models of Visual Attention" V. Mnih et al.

Run by `python ram.py` and it can reproduce the result on Table 1 (a) 28x28 MNIST

### Install dependencies
`pip install -r requirements.txt`

### Run tests
`make test`


### TODO:

* Run against OpenStack Style Guidelines and  docstrings (pep257) - https://atom.io/packages/linter-flake8
* RawDataset should extend dummydataset.
* Direct acces variable - https://github.com/monapasan/bachelor-thesis/blob/master/src/RawDataset.py#L111
* https://github.com/monapasan/bachelor-thesis/blob/master/src/RawDataset.py#L118, dataset should have property add_sample()
* https://github.com/monapasan/bachelor-thesis/blob/master/src/RawDataset.py#L167 , porpert get() : tuple()
