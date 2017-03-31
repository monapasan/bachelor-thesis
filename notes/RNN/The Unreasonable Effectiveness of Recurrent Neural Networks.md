# The Unreasonable Effectiveness of Recurrent Neural Networks
#reccurent-neural-networks/karpathyExamples
Sum up over this lecture: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
Why would one use RNN:
	* The core reason that recurrent nets are more exciting is that they allow us to operate over sequences of vectors: Sequences in the input, the output, or in the most general case both.

![](The%20Unreasonable%20Effectiveness%20of%20Recurrent%20Neural%20Networks/4D271118-121B-4825-910F-C94814144D59.png)

Each rectangle is a vector and arrows represent functions (e.g. matrix multiply). Input vectors are in red, output vectors are in blue and green vectors hold the RNN's state (more on this soon). 
From left to right: 
	1. Vanilla mode of processing without RNN, from fixed-sized input to fixed-sized output (e.g. image classification). 
	2. Sequence output (e.g. image captioning takes an image and outputs a sentence of words). 
	3. Sequence input (e.g. sentiment analysis where a given sentence is classified as expressing positive or negative sentiment). 
	4. Sequence input and sequence output (e.g. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French). 
	5. Synced sequence input and output (e.g. video classification where we wish to label each frame of the video). 

Notice that in every case are no pre-specified constraints on the lengths sequences because the recurrent transformation (green) is fixed and can be applied as many times as we like.	


The takeaway is that even if your data is not in form of sequences, you can still formulate and train powerful models that learn to process it sequentially. You’re learning stateful programs that process your fixed-sized data.
- - - -

#### Vanila RNN
```
class RNN:
  # ...
  def step(self, x):
    # update the hidden state
    self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    # compute the output vector
    y = np.dot(self.W_hy, self.h)
    return y
```

So basically input just changes the hidden state. And output will depend on the updated hidden state.
`Whh` - what to throw out of the current hidden state
`Wxh` - what to take from the input to update the hidden state.


 There are two terms inside of the tanh: 
	* one is based on the previous hidden state and 
	* one is based on the current input. 


We initialize the matrices of the RNN with random numbers and the bulk of work during training goes into finding the matrices that give rise to desirable behavior, as measured with some loss function that expresses your preference to what kinds of outputs y you’d like to see in response to your input sequences x.

RNNs are neural networks and everything works monotonically better (if done right) if you put on your deep learning hat and start stacking models up like pancakes. For instance, we can form a 2-layer recurrent network as follows:

```
y1 = rnn1.step(x)
y = rnn2.step(y1)
```
	
In other words we have two separate RNNs: 
	* One RNN is receiving the input vectors 
	* and the second RNN is receiving the output of the first RNN as its input. Except neither of these RNNs know or care - it’s all just vectors coming in and going out, and some gradients flowing through each module during backpropagation.

**Getting fancy.** I’d like to briefly mention that in practice most of us use a slightly different formulation than what I presented above called a Long Short-Term Memory (LSTM) network. The LSTM is a particular type of recurrent network that works slightly better in practice, owing to its more powerful update equation and some appealing backpropagation dynamics. I won’t go into details, but everything I’ve said about RNNs stays exactly the same, except the mathematical form for computing the update (the line self.h = ... ) gets a little more complicated. From here on I will use the terms “RNN/LSTM” interchangeably but all experiments in this post use an LSTM.


At test time, we feed a character into the RNN and get a distribution over what characters are likely to come next. We sample from this distribution, and feed it right back in to get the next letter. Repeat this process and you’re sampling text! Lets now train an RNN on different datasets and see what happens

#### Hard and Soft attention 

Now, I don’t want to dive into too many details but a soft attention scheme for memory addressing is convenient because it keeps the model fully-differentiable, but unfortunately one sacrifices efficiency because everything that can be attended to is attended to (but softly). Think of this as declaring a pointer in C that doesn’t point to a specific address but instead defines an entire distribution over all addresses in the entire memory, and dereferencing the pointer returns a weighted sum of the pointed content (that would be an expensive operation!). This has motivated multiple authors to swap soft attention models for hard attention where one samples a particular chunk of memory to attend to (e.g. a read_write action for some memory cell instead of reading_writing from all cells to some degree). This model is significantly more philosophically appealing, scalable and efficient, but unfortunately it is also non-differentiable. This then calls for use of techniques from the Reinforcement Learning literature (e.g. REINFORCE) where people are perfectly used to the concept of non-differentiable interactions. 












