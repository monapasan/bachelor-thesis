# Lecture 4, Model-Free Prediction
#reinforcment-learning/lecture4

Model Free prediction means that we don't know the exact process of MDP in an environment,  but still need to figure out how to behave in this environment.

**Model Free - no one tells us the environment**

The assumption that the MDP is given to us  is unrealistic. Therefore we need to estimate a value function from only interactions of agents with the environment (Model-free Prediction) . 

- - - -
#### Monte-Carlo Reinforcement Learning
 
_MC Simplified_:
	* Start from some state
	* sample episode return.
	* look over sample return and average over them
	* Based on the average value we can already estimate state.

_Properties of MC_:
	* MC methods learn directly from episodes of experience
	* MC is model-free: no knowledge of MDP transitions / rewards
	* MC learns from complete episodes: no bootstrapping
	* MC uses the simplest possible idea: value = mean return
	* Caveat: can only apply MC to episodic MDPs
		* All episodes must terminate	


#### First-Visit Monte-Carlo Policy Evaluation

	* To evaluate state s
	* The first time-step t that state s is visited in an episode,
	* Increment counter `N(s) ← N(s) + 1`
		* Counter persist over the set of episodes
		* as well as return
	* Increment total return `S(s) ← S(s) + Gt`
	* Value is estimated by mean return `V(s) = S(s)/N(s)`
	* By law of large numbers, `V(s) → vπ(s) as N(s) → ∞`

it's guaranteed that we will visit all states that are reachable following the policy p

#### Every-Visit Monte-Carlo Policy Evaluation

Basically the same idea as First-visit MC policy evaluation but we increment the counter each time the we go through the state and start a new return from that state once again.


The mean µ1, µ2, ... of a sequence x1, x2, ... can be computed incrementally: 

![](Lecture%204,%20Model-Free%20Prediction/94B8B2A3-1F82-4366-B0F4-A2B7248ED26F.png)


Intuition:
	* we compute error between:
		* what we think the value gonna be u k-1
		* and the new value coming Xk
	* and we then slightly move our average towards this error.
	* Correct this prediction in the direction of the error.


#### Incremental Monte-Carlo Updates

replace updating the value of the policy with an incremental form of average.

	* Update V(s) incrementally after episode S1, A1, R2, ..., ST
	* For each state St with return Gt
![](Lecture%204,%20Model-Free%20Prediction/A54D4826-DF8B-4831-9694-2FD4D22FA664.png)


In non-stationary problems, it can be useful to track a running mean, i.e. forget old episodes:
![](Lecture%204,%20Model-Free%20Prediction/E6C82523-0793-401F-8152-C4CBC22A32F5.png)

To get rid of the N(S), i.e. counter of how much a state was visited introducing a fixed alpha value which will act as rate of how much we want to change our value.
Advantage of this form is in non-stationary problems, where we always want to improve our policy(or value function)


- - - -
#### Temporal-Difference Learning


* TD methods learn directly from episodes of experience
* TD is model-free: no knowledge of MDP transitions / rewards
* TD learns from incomplete episodes, by bootstrapping
	* Bootstrapping
		* start with a guess of how much reward I gonna get from episode(e.g. going to the wall)
		* then do a part of an episode (e.g. going one part further to the wall)
		* Update a guess based on the reward from the part of episode
		* 
* TD updates a guess towards a guess

#### TD vs MC

* TD can learn before knowing the final outcome
	* based on a guess
* TD can learn online after every step
	* online - means that we don't wait until the end of the episode and update the value function straight off.
* MC must wait until end of episode before return is known
* TD can learn without the final outcome
* TD can learn from incomplete sequences
* MC can only learn from complete sequences
* TD works in continuing (non-terminating) environments
* MC only works for episodic (terminating) environments

In TD each goes is back up by another guess therefore comes to a true value function.

#### Bias/Variance Trade-Of
* Return `Gt = Rt+1 + γRt+2 + ... + γT−1*RT` is unbiased estimate of vπ(St)
* True TD target `Rt+1 + γvπ(St+1)` is unbiased estimate of vπ(St)
* TD target `Rt+1 + γV(St+1)` is biased estimate of `vπ(St)`
* TD target is much lower variance than the return:
	* Return depends on many random actions, transitions, rewards
	* TD target depends on one random action, transition, reward


* MC has high variance, zero bias
	* Good convergence properties
	* (even with function approximation)
	* Not very sensitive to initial value
	* Very simple to understand and use
* TD has low variance, some bias
	* Usually more efficient than MC
	* TD(0) converges to vπ(s)
	* (but not always with function approximation)
	* More sensitive to initial value


		
#### Certainty Equivalence

* MC converges to solution with minimum mean-squared error
	* Best fit to the observed returns
* TD(0) converges to solution of max likelihood Markov model
	* Solution to the MDP hS, A,Pˆ, Rˆ, γi that best fits the data
	* it creates internally an MDP,  and search solution for MDP
	* It's also possible to put the MDP created from TD to MC.

TD implicitly build a MDP model, and based on this model it's giving the value which describes the data TD has seen so far. TD build a MDP and then solves it.
How TD does it?
![](Lecture%204,%20Model-Free%20Prediction/6DD82657-7D54-4F11-9DA4-DE228B350D25.png)
It builds first a sort of matrix transition, from state s to state s'

Advantages and Disadvantages of MC vs. TD (3)

* TD exploits Markov property
	* Usually more efficient in Markov environments
* MC does not exploit Markov property
	* Usually more effective in non-Markov environments

- - - -
#### Bootstrapping and Sampling

* Bootstrapping: update involves an estimate or guess, backing up the value
	* MC does not bootstrap
	* DP bootstraps
	* TD bootstraps
* Sampling: update samples an expectation
	* MC samples
	* DP does not sample
		* it does an exhaustive update or full width back up over the all possible states.
		* i.e. considers all possible  
	* TD samples


### TD(λ)

Idea is to make the algorithm more flexible by introducing a lambda, which says after how many steps we want to update our value function. In order to avoid sticking either to shallow backups or deep backups.()
![](Lecture%204,%20Model-Free%20Prediction/5582344B-B285-4A0A-B9A9-FCAA0A175EEC.png)





#### Averaging n-Step Returns
**Problem**:  using different `n` we can get different error rate which may be slightly better on one `n` . Therefore it is not clear how to chose `n`  appropriately. 
**The idea** is that we don't  have to choose any `n`. We can just calculate different return for different `n` and the average those. Therefore it makes our prediction more robust because it's kind gets the best of the both of this cases.
But we still need to chose those `n`

Is there anything that will consider all different  `n`   without increasing the complexity?
Yes:
#### TD(λ), λ-return

Constant λ is between 0 and 1, and tell us the weighting that we have for each successive n.
TD(λ) algorithm simplifies TD(n) algorithm by exempting from choose of n and decaying all returns start from state till the finish of episode. 

![](Lecture%204,%20Model-Free%20Prediction/92C310E0-A9CC-4302-B57B-8983F2106955.png)
![](Lecture%204,%20Model-Free%20Prediction/2EAA2200-6840-4521-B284-144E9D4244C9.png)



#### Eligibility Traces

Eligibility traces - we look over time at the state that we visited, increase value of eligibility trace for this state, and over time when we are not visiting state, we decrease value of eligibility trace:
![](Lecture%204,%20Model-Free%20Prediction/AF0459EF-4D09-4261-BF70-4B441A30B266.png)


#### Backward View TD(λ)

* Keep an eligibility trace for every state s
	* Just a simple function,  not really expensive.
* Update value V(s) for every state s
* In proportion to TD-error δt and eligibility trace Et(s):
![](Lecture%204,%20Model-Free%20Prediction/69E63E29-413B-4B6B-BFCE-B62B3C59A944.png)
Basically we updating out value with respect to eligibility trace. According to eligibility we can say whether this state is responsible for the error. (for example if Et(s) = 1, means that state "responsible" for the error)

Intuition is that:  eligibility says which state is responsible for the TD-error

























