# R2RT, Recurrent Neural Networks in Tensorflow I
http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
_It won't hurt to lookup cross entropy of probabilities as well as backpropagation algorithm(e.g. truncated backpropagation)_
#reccurent-neural-networks/r2rtRNNinTF

In this post, we’ll be building a no frills RNN that accepts a binary sequence X and uses it to predict a binary sequence Y. The sequences are constructed as follows:
	* Input sequence (X): At time step Xt  has a 50% chance of being 1 (and a 50% chance of being 0). E.g., X might be [1, 0, 0, 1, 1, 1 … ].
	* Output sequence (Y): At time step t, Yt has a base 50% chance of being 1 (and a 50% base chance to be 0).
		* The chance of Yt being 1 is increased by 50% (i.e., to 100%) if Xt−3 is 1, and decreased by 25% (i.e., to 25%) if Xt−8 is 1. If both Xt−3 and Xt−8 are 1, the chance of Yt being 1 is 50% + 50% - 25% = 75%.

# Model architecture
The model will be as simple as possible:
	* at time step t, for t ∈ {0,1,…n} the model accepts:
		* a (one-hot) binary Xt vector
		* and a previous state vector, St−1, as inputs
	* and produces:
		* a state vector, St,
		* and a predicted probability distribution vector Pt, for the (one-hot) binary vector Yt.

Formally, the model is:
![](R2RT,%20Recurrent%20Neural%20Networks%20in%20Tensorflow%20I/16527E28-FFC8-403C-A1BF-9DBEDB3456DC.png)

#### How wide should our Tensorflow graph be?


To build models in Tensorflow generally, you first represent the model as a graph, and then execute the graph.
 _A critical question we must answer when deciding how to represent our model is_: **how wide should our graph be? How many time steps of input should our graph accept at once?**

There are two cases:
1. Each time step is a duplicate, so it might make sense to have our graph, G, represent a single time step: G(Xt,St−1)↦(Pt,St).
	* We can then execute our graph for each time step, feeding in the state returned from the previous execution into the current execution.
	* This would work for a model that was already trained, but there’s a problem with using this approach for training:
		* the gradients computed during backpropagation are graph-bound. We would only be able to backpropagate errors to the current timestep; we could not backpropagate the error to time step t-1.
		* this mean we won't be able to learn long-term dependency for our model.
2. We might make our graph as wide as our data sequence:
	* This often works, except that in this case, we have an arbitrarily long input sequence, so we have to stop somewhere.
	* Let’s say we make our graph accept sequences of length 10,000. This solves the problem of graph-bound gradients, and the errors from time step 9999 are propagated all the way back to time step 0.
	* **Unfortunately, such backpropagation is not only (often prohibitively) expensive, but also ineffective, due to the vanishing / exploding gradient problem**:
		* it turns out that backpropagating errors over too many time steps often causes them to vanish (become insignificantly small) or explode (become overwhelmingly large). 		

So we have to find a trade-off between these two.

**The usual pattern for dealing with very long sequences is therefore to “truncate” our backpropagation by backpropagating errors a maximum of n steps.**
 _We choose n as a hyperparameter to our model, keeping in mind the trade-off_:
	* higher n lets us capture longer term dependencies, but is more expensive computationally and memory-wise.


**A natural interpretation of backpropagating errors a maximum of n steps means that we backpropagate every possible error n steps.**
That is, if we have a sequence of length 49, and choose n=7, we would backpropagate 42 of the errors the full 7 steps.
**This is not the approach we take in Tensorflow. Tensorflow’s approach is to limit the graph to being n units wide.**

This means that:
	* we would take our sequence of length 49,
	* break it up into 7 sub-sequences of length 7
	* that we feed into the graph in 7 separate computations,
	* and that only the errors from the 7th input in each graph are backpropagated the full 7 steps. Therefore, even if you think there are no dependencies longer than 7 steps in your data, it may still be worthwhile to use n>7 so as to increase the proportion of errors that are backpropagated by 7 steps.


**Our graph will be n units (time steps) wide where each unit is a perfect duplicate, sharing the same variables.**
 The easiest way to build a graph containing these duplicate units is to build each duplicate part in parallel. This is a key point: **the easiest way to represent each type of duplicate tensor (the rnn inputs, the rnn outputs (hidden state), the predictions, and the loss) is as a list of tensors.**


## Model

```
"""
Placeholders
"""

x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

"""
RNN Inputs
"""

# Turn our x placeholder into a list of one-hot tensors:
# rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
x_one_hot = tf.one_hot(x, num_classes)
# x_one_hot.shape --> 200, 5 , 2
# so we have a batch size of 200, trunctated_step of 5, and one_hot =2 (which is either 0 or 1)

rnn_inputs = tf.unstack(x_one_hot, axis=1)
# what we do here is unroll/unstack x_one_hot into a list each of those which (200,2).
# i.e. list of 5 --> 200, 2
```

- - - -

## Sum up of the process

1. define parameters:
```
num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1
```

In details:
	* `learning_rate` - parameter for gradient descent algorithm or another approach.
	* `num_classes `  - number of classes (for one-hot encryption)
	* `batch_size` - batch size to train on
	* `state_size`  - the size of the RNN cell state. Should be big if truncated back-propagation required more steps to propagate back in order to store appropriate amount of information.
	* `num_steps` - number of truncated backdrop steps, is dependent of how deep one wants to propagate error, hence how deep you think data dependency should be
2. define the model:
	* define placeholder for input data as well as for the labels
	* define initial state
	* Define static model without tf api:
		* turn the input into one-hot tensors
		* unstack the input data into list of `num_steps` tensors with shape `[batch_size, num_classes]`
		* define RNN cell:
			* define weights and biases
			* define `tahn`  connected layer or with any function.
		* build a graph:
```
			state = init_state
			rnn_outputs = []
			for rnn_input in rnn_inputs:
			    state = rnn_cell(rnn_input, state)
			    rnn_outputs.append(state)
			final_state = rnn_outputs[-1]
```
	* Define loss function and predictions:
```
#logits and predictions
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# Turn our y placeholder into a list of labels
y_as_list = tf.unstack(y, num=num_steps, axis=1)

#losses and train_step
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
```
	* Train the network:
```
def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                                  feed_dict={x:X, y:Y, init_state:training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 250 steps:", training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0

    return training_losses
```


Using the tensor flow api this will look following:
```
cell = tf.contrib.rnn.BasicRNNCell(state_size)
rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)
```
So we don't create a connected layer by ourselves, but instead using class BasicRNNCell to create a cell and passing desired the size of state.

```
Final model using static api:
"""
Placeholders
"""

x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

"""
Inputs
"""

x_one_hot = tf.one_hot(x, num_classes)
rnn_inputs = tf.unstack(x_one_hot, axis=1)

"""
RNN
"""

cell = tf.contrib.rnn.BasicRNNCell(state_size)
rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

"""
Predictions, loss, training step
"""

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

y_as_list = tf.unstack(y, num=num_steps, axis=1)

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
```



## Using a dynamic RNN
Above, we added every node for every time-step to the graph before execution. This is called “static” construction. <u>We could also let Tensorflow dynamically create the graph at execution time, which can be more efficient.<u> 
>  To do this, instead of using a list of tensors (of length num_steps and shape  `[batch_size, features]`), we keep everything in a single 3-dimnesional tensor of shape ` [batch_size, num_steps, features]`, and use Tensorflow’s `dynamic_rnn` function. This is shown below.  
