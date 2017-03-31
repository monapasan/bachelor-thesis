# Lecture 6, Value Function Approximation
* #reinforcment-learning/lecture6

#### Why do we need it?
The problem is so far that we have to store all state that we've visited. For solving the real world problem we will need a huge state space, for some problems state space can be continuous. We want our function value to understand a generalisation of the state without need to store all of them. Therefore it is crucial to build up a good representation of our value function. 

## Value Function Approximation
The Idea is to build a function which will take a state S and some weight W as parameter, and based on that will say our how to behave in the current state:
![](Lecture%206,%20Value%20Function%20Approximation/AC09F663-FA69-4DF6-AE66-D1480FDBC0AE.png)

* Generalise from seen states to unseen states
	* the other point is that in that case it will be possible to generalise our function to unseen states based on seen states 
* Update parameter w using MC or TD learning

What it means to do a value approximation with a value function !?

Non-Stationary data have a lot variance, covariance and therefore almost impossible to model.

#### Types of Value Function Approximation

1. feeding a state to a function, return value gonna be how good it is to be in this state
2. feeding state S and action A, return value gonna be how good it is to be in this state and taking an action A
3. feeding state S, return values are how good it is to take a actions from this state S.
4. 
 
![](Lecture%206,%20Value%20Function%20Approximation/36FAA110-3BD0-4131-9D6D-1F114CE0AEF3.png)



We are trying to estimate a non-stationary sequence of value functions. 

The good thing is to use a linear combination of features within gradient descent algorithm because in that case it will always converge
- - - -

#### TD(λ) with Value Function Approximation

The λ-return Gλt is also a biased sample of true value vπ(s)
Can again apply supervised learning to “training data”:
![](Lecture%206,%20Value%20Function%20Approximation/9EBBB74B-B30A-46C9-8B2A-439FBF4A47CF.png)

* Forward view linear TD(λ):
![](Lecture%206,%20Value%20Function%20Approximation/F9C577A9-CE7A-4155-A061-01A61B1642E9.png)

* Backward view linear TD(λ):
![](Lecture%206,%20Value%20Function%20Approximation/6A930875-1192-437B-ACF6-A9908143B66A.png)

Eligibility traces are based now on feature vector that we have and have size of parameters, not the size of the state space. It's kinda remembering all the features that we've seen so far and accumulating the things that happen most and decay things that happen rare.

Forward view and backward view linear TD(λ) are equivalent.
- - - -


#### Action-Value Function Approximation
 Once again: the reason behind using action value function is that in this case we don't need a model.
* Approximate the action-value function:
![](Lecture%206,%20Value%20Function%20Approximation/B61999CE-35E9-49B9-8A76-673940A23BAE.png)

basically the same ideas, but function approximation will now represent a action value function.

#### Linear Action-Value Function Approximation
Represent state and action by a feature vector:
![](Lecture%206,%20Value%20Function%20Approximation/AD24B789-9A5C-4257-8590-7FD67BEE6C48.png)

Intuition: how far I would be from the wall if I take an action moving forward.

In theory using td for non linear function may not converge to our real value function.




- - - -
# Batch Reinforcement Learning

* Gradient descent is simple and appealing
* But it is not sample efficient
	* meaning we sample a reward, we are updating our function and then throwing the data away.
	* **We haven't make the maximum use of this data!**
* Batch methods seek to find the best fitting value function
	* The question to ask is: How  we can get most value from the data that we see to find the best possible value function hence the policy. 
	* best fitting value function to **ALL** of the data we have seen in the batch.
* Given the agent’s experience (“training data”)

The main idea is that we don't throw away our data, we cache/store it instead.

So we basically store all the data we see, randomly chose sample from this data and then update our weight in the direction of this random sample.

So we just keep doing that again, again and again until we come to the solution.

DQN uses **experience replay** and **fixed Q-targets**:
	* Take action at according to -greedy policy
	* Store transition (st, at,rt+1,st+1) in replay memory D
	* Sample random mini-batch of transitions (s, a,r,s0) from D
	* Compute Q-learning targets w.r.t. old, fixed parameters w−
	* Optimise MSE between Q-network and Q-learning targets:
	
![](Lecture%206,%20Value%20Function%20Approximation/A52F1DAF-9EC0-4572-861C-6F8A03B5F065.png)

Why this will converge:
	* because with experience replay we make sure that we stabilise the neural network because it decorelates it's trajectory.
	* fixed Q-targets:
		* we keep two weights.
		* one weights is kind frozen
		* We remember the old parameters (for example 5 steps ago)
		* and we will update one of our fresh weights with respect to old weights f.i. from 5 steps before
		* So we don't use our freshest weights to update weights
		* And this will give us the stable update
		* So basically we keep weights for a certain number of iterations, and only after this number of iteration we update our _frozen_ weights
		* In practice it stabilising everything










