This is the implementation of the prototype presented in bachelor thesis "Development and Evaluation of a Visual Attention Model with Python and Tensorflow".

To run the model, you need to have python3 and pip command available in terminal.
Below you can find commands that you can execute with the help of makefile.
You can change the default parameters of the model by providing the arguments on the command line.
Another way to change the parameters is to change the values [here](https://github.com/monapasan/bachelor-thesis/blob/master/src/main.py) and then run the model with `make run`. You will also find in this file the descriptions of the parameters. 

## Execution

In the root diretory of the project execute following commands:

### To install dependencies
`pip install -r requirements.txt`

### To run tests
`make test`

### To start train the model
`make run`

### print parameters of the model
`make help`

Without help of the  make it's also to possible execute the following commands:

### To run tests
`nosetests tests --nologcapture --rednose -d`

### To start train the model
`python3 src/main.py`

### print parameters of the model
`python3 src/main.py --help`

