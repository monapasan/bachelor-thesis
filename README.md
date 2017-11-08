
## Description

This is the implementation of the prototype presented in bachelor thesis "Development and Evaluation of a Visual Attention Model with Python and Tensorflow".

To run the model, you need to have python3 and pip command available in terminal.
In order to run the model, you need to install the reuiqred packaged with pip.
Below you can find commands that you can execute.
___

You can change the default parameters of the model by providing the arguments on the command line.
Another way to change the parameters is to change the values in [/src/main.py](https://github.com/monapasan/bachelor-thesis/blob/master/src/main.py) and then run the model with `make run` or `python3 src/main.py`. You will also find in this file the descriptions of the parameters. 

___
## Execution

In the root diretory of the project execute following commands:

### To install dependencies
`pip install -r requirements.txt`

### To run tests
`make test`

### To start training of the model
`make run`

### print parameters of the model
`make help`

___

Without help of the  makefile it's also to possible execute the following commands:

### To run tests
`nosetests tests --nologcapture --rednose -d`

### To start training of the model
`python3 src/main.py`

### print parameters of the model
`python3 src/main.py --help`

