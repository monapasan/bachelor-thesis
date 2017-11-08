# Understanding LSTM Networks
#reccurent-neural-networks


Recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Consider what happens if we unroll the loop:


![](Understanding%20LSTM%20Networks/DE29A7F1-3228-40A3-B740-54468ACE986A.png)



This chain-like nature reveals that recurrent neural networks are intimately related to sequences and lists. They’re the natural architecture of neural network to use for such data.

Essential to these successes is the use of “LSTMs,” a very special kind of recurrent neural network which works, for many tasks, much much better than the standard version. Almost all exciting results based on recurrent neural networks are achieved with them. It’s these LSTMs that this essay will explore.
- - - -
#### The Problem of Long-Term Dependencies
One of the appeals of RNNs is the idea **that they might be able to connect previous information to the present task**, such as using previous video frames might inform the understanding of the present frame. 
The problem with usual RNN that they are not capable of handling long-term dependencies.
 
#### LSTM Networks

LSTMs are explicitly designed to avoid the long-term dependency problem. **Remembering information for long periods of time is practically their default behaviour**, not something they struggle to learn!


**All recurrent neural networks have the form of a chain of repeating modules of neural network.** In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.


![](Understanding%20LSTM%20Networks/718D14E3-E100-45B3-BC5A-E4F114CD9FC5.png)

LSTMs also have this chain like structure, but the repeating module has a different structure. _Instead of having a single neural network layer, there are four, interacting in a very special way._
![](Understanding%20LSTM%20Networks/8F4686F5-947E-4F25-8CD5-966C9B41C1D3.png)
In the above diagram, **each line carries an entire vector**, from the output of one node to the inputs of others.
![](Understanding%20LSTM%20Networks/LSTM2-notation.png)

#### The Core Idea Behind LSTMs

_The key to LSTMs is the cell state, the horizontal line running through the top of the diagram._
![](Understanding%20LSTM%20Networks/9C07C641-F342-4612-B04E-525FA20CD076.png)


**The LSTM does have the ability** to _remove or add information to the cell state_, carefully regulated by structures called gates.

Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

The sigmoid layer outputs numbers between zero and one, **describing how much of each component should be let through**. A value of zero means “let nothing through,” while a value of one means “let everything through!”

_An LSTM has three of these gates, to protect and control the cell state._

_The first step in our LSTM is to decide what information we’re going to throw away from the cell state_. This decision is made by a sigmoid layer called the **“forget gate layer.”**

![](Understanding%20LSTM%20Networks/8A70FD41-F11B-421A-A868-5A83ADC4A1AE.png)

_The next step is to decide what new information we’re going to **store in the cell state**._ 
This has two parts:
	* First, a sigmoid layer called the “input gate layer” decides which values we’ll update. 
	* Next, a tanh layer creates a vector of new candidate values, C̃ t, that could be added to the state. 
In the next step, we’ll combine these two to create an update to the state.

![](Understanding%20LSTM%20Networks/D8FA233E-C5B7-4FD0-A0F6-802F97BD72F6.png)


**Update the state:**
We multiply the old state by `ft`, forgetting the things we decided to forget earlier. Then we add `it∗C̃` . **This is the new candidate values, scaled by how much we decided to update each state value**.

![](Understanding%20LSTM%20Networks/986D47E1-96E5-418B-92BF-D884EDD51A99.png)
 First multiplication is forget gate, second is add gate.


Finally, we need to decide what we’re going to output. 
This output will be based on our cell state, but will be a filtered version:
	* First, we run a _sigmoid layer which decides what parts of the cell state we’re going to output_. 
	* Then, we put the cell state through tanh(to push the values to be between −1 and 1) and _multiply it by the output of the sigmoid gate, so that we only output the parts we decided to._

![](Understanding%20LSTM%20Networks/FBECD206-F802-4AB0-B58C-8BC7A092D8DD.png)

For the language model example, since it just saw a subject, _it might want to output information relevant to a verb, in case that’s what is coming next_. For example, it might output whether the subject is singular or plural, so that we know what form a verb should be conjugated into if that’s what follows next.

	


