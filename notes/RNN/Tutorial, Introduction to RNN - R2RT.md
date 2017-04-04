# Tutorial, Introduction to RNN - R2RT
#reccurent-neural-networks/r2rt_introduction
http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html#fn2

Outline of the tutorial:
	1. We’ll define and describe RNNs generally, focusing on the limitations of vanilla RNNs that led to the development of the LSTM.
	2. We’ll describe the intuitions behind the LSTM architecture, which will enable us to build up to and derive the LSTM. Along the way we will derive the GRU. We’ll also derive a pseudo LSTM, which we’ll see is better in principle and performance to the standard LSTM.
	3. We’ll then extend these intuitions to show how they lead directly to a few recent and exciting architectures: highway and residual networks, and Neural Turing Machines.

**An RNN** is a composition of identical feedforward neural networks, one for each moment, or step in time, which we will refer to as “RNN cells”.

![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/FA1A9251-1C1D-4D14-92DA-9BE244BAE45D.png)

Analogy to how our brain functions:
	* We have some background, each of us has. Some thought that we always have in our mind. This can represented as Internal input to RNN cell.
	* We also have an ability to percept things. We can see, hear and etc. This can be though as an external input.
	* After we see something, read something, eat something. It has an further influence on our background, on our thoughts. Hence our thoughts are changing. We can think about this changed background as an internal output of RNN cell.
	* But we do not only changed our mind after we see something, we can also do actions based on what we see. This optional actions can be represented as external output from RNN cell.

You can think of the recurrent outputs as a “state” that is passed to the next timestep. Thus an RNN cell accepts a prior state and an (optional) current input and produces a current state and an (optional) current output.

Here is the algebraic description of the RNN cell:
![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/AAE73772-12B2-451A-9268-5DAC8AC810D5.png)
where:
![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/BD0D705F-E03A-4A2B-9D6A-A17C89B1C0C8.png)

Our brain operates in place: current neural activity takes the place of past neural activity. We can see RNNs as operating in place as well: _because RNN cells are identical, they can all be viewed as the same object, with the “state” of the RNN cell being overwritten at each time step_. Here is a diagram of this framing:

![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/498B0EA3-B9CB-4490-89BD-812CFEFBCF61.png)

When starting with the single cell loop framing, RNN’s are said to “unrolled” to obtain the sequential framing above.

In theory, RNN cell can do anything: _if we give the neural network inside each cell at least one hidden layer, each cell becomes a universal function approximator_


Alternatively, we might say that each word or each character is a time step. Here is an illustration of what an RNN translating “the cat sat” on a per word basis might look like:
![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/6AAAE7B6-B83F-4996-8CEE-5D4EBA689A48.png)

* After the first time step, the state contains an internal representation of “the”; after the second, of “the cat”; after the the third, “the cat sat”.
* The network does not produce any outputs at the first three time steps.
* It starts producing outputs when it receives a blank input, at which point it knows the input has terminated. When it is done producing outputs, it produces a blank output to signal that it’s finished.

In practice, even powerful RNN architectures like deep LSTMs might not perform well on multiple tasks (here there are two: reading, then translating). **To accomodate this, we can split the network into multiple RNNs, each of which specializes in one task**. In this example, we would use an “encoder” network that reads in the English (blue) and a separate “decoder” network that reads in the French (orange):

![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/76F76E87-BF2D-4DCC-B742-C79090EEB05B.png)
Additionally, as shown in the above diagram, **the decoder network is being fed in the last true value** (i.e., the target value during training, and the network’s prior choice of translated word during testing). For an example of an RNN encoder-decoder model,

Notice that having two separate networks still fits the definition of a single RNN: **we can define the recurring function as a split function that takes, alongside its other inputs, an input specifying which split of the function to use.**


_We can let the RNN decide when its ready to move on to the next input, and even what that input should be._ This is similar to how a human might focus on certain words or phrases for an extended period of time to translate them or might double back through the source.
To do this:
	* we use the RNN’s output (an external action) to determine its next input dynamically.
		* For example, we might have the RNN output actions like “read the last input again”, “backtrack 5 timesteps of input”, etc.
	* Successful attention-based translation models are a play on this:
		* they accept the entire English sequence at each time step
		* and their RNN cell decides which parts are most relevant to the current French word they are producing.

_The most basic RNN cell is a single layer neural network, the output of which is used as both the RNN cell’s current (external) output and the RNN cell’s current state_:

![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/BAA76D58-4660-486D-A436-B7187BECE7D8.png)

Note how _the prior state vector is the same size as the current state vector_. As discussed above, **this is critical for composition of RNN cells**. Here is the algebraic description of the vanilla RNN cell:
![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/57C82473-F5B4-48F7-BFE8-59F131938E65.png)


_There is a problem in practice:_ training vanilla RNNs with backpropagation algorithm turns out to be quite difficult, even more so than training very deep feedforward neural networks. This difficulty is due to the problems of information morphing and vanishing and exploding sensitivity caused by repeated application of the same nonlinear function.


## Backpropagation through time and vanishing sensitivity

1. We backpropagate errors through time:
	* _For RNNs we need to backpropagate errors from the current RNN cell back through the state, back through time, to prior RNN cells._
	* This allows the RNN to learn to capture long term time dependencies.
	* Because the model’s parameters are shared across RNN cells (each RNN cell has identical weights and biases),
	* we need to calculate the gradient with respect to each time step separately and then add them up. This is similar to the way we backpropagate errors to shared parameters in other models, such as convolutional networks.
2. There is a trade-off between weight update frequency and accurate gradients:

For all gradient-based training algorithms, there is an unavoidable trade-off between:
1. frequency of parameter updates (backward passes), and
2. accurate long-term gradients.


_We could compute more accurate gradients by doing fewer parameter updates (backward passes), but then we might be giving up training speed_ (which can be particularly harmful at the start of training).
**Note** the similarity to the trade off to the one faces by choosing a mini-batch size for mini-batch gradient descent: the larger the batch size, the more accurate the estimate of the gradient, but also the fewer gradient updates.

**Important**:
reread this section: Backpropagation through time and vanishing sensitivity after repeating Backpropagation.

Long story short:
**Why don't use vanilla RNN?**
	1. vanishing gradient problem:
		* meaning that we need to regulated our weight to ensure that gradient of those won't vanish over time.
	2. Exploding gradient problem:
		* If our gradient explodes backpropagation will not work because we will get NaN values for the gradient at early layers.
	3. Morphing problem:
		* information is morphed by RNN cells and the original message is lost. A small change in the original message may not have made any difference in the final message, or it may have resulted in something completely different.
- - - -

### Written memories: the intuition behind LSTMs

_How can we protect the integrity of messages?_ This is the fundamental principle of LSTMs: to ensure the integrity of our messages in the real world, we write them down. Writing is a delta to the current state: it is an act of creation (pen on paper) or destruction (carving in stone); the subject itself does not morph when you write on it and the error gradient on the backward-pass is constant.

**The fundamental principle of LSTMs: Write it down.**
Practically speaking, this means that any state changes are incremental, so that
![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/ED081B64-B11C-4729-BAB5-6EE3897438EF.png)

**First form of selectivity: Write selectively.**

To get the most out of our writings in the real world, we need to be selective about what we write; when taking class notes, we only record the most important points and we certainly don’t write our new notes on top of our old notes. In order for our RNN cells to do this, they need a mechanism for selective writing.

**Second form of selectivity: Read selectively.**

In order to perform well in the real-world, we need to apply the most relevant knowledge by being selective in what we read or consume. In order for our RNN cells to do this, they need a mechanism for selective reading.

Note the difference between reads and writes: If we choose not to read from a unit, it cannot affect any element of our state and our read decision impacts the entire state. If we choose not to write to a unit, that impacts only that single element of our state. This does not mean the impact of selective reads is more significant than the impact of selective writes: reads are summed together and squashed by a non-linearity, whereas writes are absolute, so that the impact of a read decision is broad but shallow, and the impact of a write decision is narrow but deep.

**Third form of selectivity: Forget selectively.**

In the real-world, we can only keep so many things in mind at once; in order to make room for new information, we need to selectively forget the least relevant old information. In order for our RNN cells to do this, they need a mechanism for selective forgetting.

### Gates as a mechanism for selectivity

**Selective reading, writing and forgetting involves separate read, write and forget decisions for each element of the state.** We will make these decisions by taking advantage of state-sized read, write and forget vectors with values between 0 and 1 specifying the percentage of reading, writing and forgetting that we do for each state element.

We call these read, write and forget vectors “gates”, and we can compute them using the simplest function we have, as we did for the vanilla RNN: the single-layer neural network.

Our three gates at time step t are denoted it:
	* It the input gate (for writing),
	* ot, the output gate (for reading) and
	* ft, the forget gate (for remembering).

From the names, we immediately notice that two things are backwards for LSTMs:
* Admittedly this is a bit of a chicken and egg, but I would usually think of first reading then writing. Indeed, this ordering is strongly suggested by the RNN cell specification– _we need to read the prior state before we can write to a new one_, so that even if we are starting with a blank initial state, we are reading from it. The names input gate and output gate suggest the opposite temporal relationship, which the LSTM adopts. We’ll see that this complicates the architecture.
* The forget gate is used for forgetting, but it actually operates as a remember gate. _E.g., a 1 in a forget gate vector means remember everything, not forget everything_. This makes no practical difference, but might be confusing.

![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/97DE189D-1A24-4079-962B-406CCC2AA103.png)


**Gluing gates together to derive a prototype LSTM**
### The Prototype LSTM
![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/C57994DD-D1A5-446D-87CC-74E8964811B8.png)
It - is our input gate and is responsible for what we will change in our state(what will write down in our state)
Ot - is our read gate or output gate. This gate is responsible for what our cell will output or let's say what we will read from our gate.
Ft - is our forget gate(which actually operates as remember gate) and is responsible for what we will remember from our prior state.
**remember** that our changes to the states are incremental:
![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/ED081B64-B11C-4729-BAB5-6EE3897438EF%201.png)
Therefore short explanation for our incremental change as following:
	* we multiply our prior state with our forget gate, which will decide what we want to remember from our state.
	* then we add this by our true _write_ which is element wise multiplication of our candidate write(including input to a state)  and our input gate(write gate)

In theory, this prototype should work, and it would be quite beautiful if it did.
_In practice:_
	* the selectivity measures taken are not (usually) enough to overcome the fundamental challenge of LSTMs:
		* the selective forgets and the selective writes are not coordinated at the start of training
		* which can cause the state to quickly become large and chaotic.
	* _Further, since the state is potentially unbounded,_ the gates and the candidate write will often become saturated, which causes problems for training.

By enforcing a bound on the state to prevent it from blowing up, we can overcome this problem. There are a few ways to do this, which lead to different models of the LSTM.
How we could solve this problem:

	* Soft bound
		* basically normalising state
	* gated recurrent unit (GRU): hard bound
		* coordinate our writes and forgets is to explicitly link them
		* mean we combine two gates: write and forget gate
		* by write gate = 1 - forget gate
	* The Pseudo LSTM: a hard bound via non-linear squashing
		*  we pass the state through the squashing function every time we need to use it for anything except making incremental writes to it.
		* By doing this, our gates and candidate write don’t become saturated and we maintain good gradient flow
		* and  impose a bound on the values used to compute the candidate write.
		* **To this point, our external output has been the same as our state, but here, the only time we don’t squash the state is when we make incremental writes to it.** Thus, our cell’s output and state are different.

The Pseudo LSTM:
![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/B96265A7-4CA7-42A1-B085-CDEA3CA22EA8.png)

#### Deriving the LSTM:
They all share one key difference with our pseudo LSTM: **the real LSTM places the read operation after the write operation.**

This difference in read-write order has the following important implication:
	* We need to read the state in order to create a candidate write.
	* But if creating the candidate write comes before the read operation inside our RNN cell,
	* we can’t do that unless we pass a pre-gated “shadow state” from one time step to the next along with our normal state. The write-then-read order thus forces the LSTM to pass a shadow state from RNN cell to RNN cell.

#### The basic LSTM

![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/C2330D65-093A-4454-9A40-CB11E1840F96.png)

Let's split it into parts:
	* Forget gate Ft - decide what we will remember from out state.
		* we add element wise the input and the out of the previous cell(hidden state)
		* then we multiply it with the state. Thus, by this action we decide what we will remember from the state
		* Note: this happens before actual read operation.
	* Write operation
		* we have candidate values Ct
			* which hidden state + input (element wise) squashed by some non linear function `phi`.
		* we have It, the same input gate layer as above.
			* Responsible for which value we will update from our candidate values
		* and then me multiply our new candidate values by input gate(which value should we take from candidate values)
		* this will be our WRITE  to the state. We add this WRITE to our state element wise.
	* Read operation,
		* We need to decide what from our state is important.
			* We do defined what is important for us:
				* hidden state + input (element wise) squashed by some non linear function
				* i.e. 1 -> take the value from the state, 0 -> don't
		* remember we always squash our state.
		* so it squashed state multiplied by the value above(value by importance)
		*



![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/8F4686F5-947E-4F25-8CD5-966C9B41C1D3.png)

### The LSTM with peepholes

 **Peepholes connections include the original unmodified prior state, ct−1 in the calculation of the gates.**


Note that  the outdated input to the read gate (due to LSTM Diff 1), and partially fixed it by moving the calculation of the read gate, ot, to come after the calculation of ct, so that ot uses ct instead of ct−1 in its peephole connection.


LSTM with peepholes:
![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/1913210B-3A77-46DD-B3FC-F070168B48FC.png)
**We let the gate layers look at the cell state.**
![](Tutorial,%20Introduction%20to%20RNN%20-%20R2RT/FF88EDE6-D5CD-41EF-A84D-4E4758DB647C.png)

Basically the idea is that you look up the current state while reading from the state.
AS well as the state while writing and forgetting.
