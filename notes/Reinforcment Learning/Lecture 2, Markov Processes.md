# Lecture 2, Markov Processes 
#reinforcment-learning/lecture2


**Markov decision processes** formally describe an environment for reinforcement learning.
	* where the environment is fully observable
	* Almost all RL problems can be formalised as MDPs, e.g.
		* Optimal control primarily deals with continuous MDPs
		* Partially observable problems can be converted into MDPs
		* Bandits are MDPs with one state


**Markov property**: the future is independent of the past given the present. 

- - - -

#### State Transition Matrix
For a Markov state s and successor state S0, the state transition
probability is defined by:

![](Lecture%202,%20Markov%20Processes/A34A810F-68B8-4C25-A8C0-8013457B2EF8.png)

State transition matrix P defines transition probabilities from all
states s to all successor states S0 : 

![](Lecture%202,%20Markov%20Processes/ACD11BD7-7BDC-4884-A33F-9D494C6E3FE8.png)

Where each row sums to 1.

- - - -
### Markov process

A **Markov process** is a memoryless random process, i.e. a sequence of random states S1, S2, ... with the Markov property.
![](Lecture%202,%20Markov%20Processes/BEFA4777-452C-4061-A6BA-9381A08CCE20.png)


- - - -
# Markov Reward Process

A **Markov reward** process is a Markov chain(Process) with values.

![](Lecture%202,%20Markov%20Processes/CC3D98FF-9C6C-4803-9941-2CBA8A3438EF.png)

This is just immediate Reward for what I get for being in a particular state.


#### Return
The return Gt is the total discounted reward from time-step t.
![](Lecture%202,%20Markov%20Processes/BD9A6012-E318-4D81-9AA9-12D3BADDB82F.png)

* The discount **γ** ∈ [0, 1] is **the present** value of future rewards
* The value of receiving reward R after k + 1 time-steps is γ^k * R.
* This values immediate reward above delayed reward.
		* γ close to 0 leads to ”myopic” evaluation
		* γ close to 1 leads to ”far-sighted” evaluation

The goal is to maximise the return


#### Value Function

The state value function v(s) of an MRP is the expected return starting from state s.
![](Lecture%202,%20Markov%20Processes/AEFEF951-8986-45C0-A3CA-2A8C2CFE46EB.png)

Value function from state  S1:
	* Sample returns from state S1 on.
	* Calculate returns 
	* Take the average of it.
	
- - - -
#### Bellman Equation for MRPs

The value function can be decomposed into two parts:
	* immediate reward Rt+1
	* discounted value of successor state γv(St+1)


#### Bellman Equation in Matrix Form
![](Lecture%202,%20Markov%20Processes/EB8D5D0D-01E5-4E56-BD50-856A443612B2.png)


The Bellman equation is a linear equation and can be solved directly:
v = R + y * P * v
v(1 - y * P) = R
V = (1-y*p)^-1  * R

* Computational complexity is O(n^3) for n state
* Direct solution only possible for small MRPs

- - - -


# Markov Decision Process
What is actually used in RL.
A Markov decision process (MDP) is a Markov reward process **with decisions**. It is an environment in which all states are Markov.

![](Lecture%202,%20Markov%20Processes/A47477C4-D15E-4AC4-BB62-0476AA160011.png)

State where one ends up is also now dependent on the actions. So now there is n ability to chose state.
 So now agent has more control over the process. 

#### Policies

A policy π is a distribution over actions given states:
`π(a|s) = P [At = a | St = s]`

	* A policy fully defines the behaviour of an agent
	* MDP policies depend on the current state (not the history)
	* i.e. Policies are _stationary_ (time-independent),
Policy is a mapping from state to probability of going to next possible state which is controlled by agent.
It's stochastic function. It's usable to make it stochastic since it give us possibility to make things like exploration. 

#### Value Function
with Fixed Policy:

The **state-value** function vπ(s) of an MDP is the expected return starting from state s, and then following policy π.

`vπ(s) = Eπ [Gt| St = s]`
State value function says us how good is to be in a state St.

The **action-value** function qπ(s, a) is the expected return starting from state s, taking action a, and then following policy π:
`qπ(s, a) = Eπ [Gt| St = s, At = a]`

Action-value function tell us how good is to take a particular action.

#### Bellman Expectation Equation for Vπ

Considering we are in state S. We might take action left and action right. We can calculate the equation below to understand How goos it is to be in the state S.
![](Lecture%202,%20Markov%20Processes/1352EC3C-DB56-4E59-B426-0B20BC6DAACF.png)


Bellman Expectation Equation for Qπ.

Here we want to understand how good is it to take an action a. Considering situation below where after taking an action environment can throw agent  to left and to the right, so we need to consider all of state(in this case left and right).

![](Lecture%202,%20Markov%20Processes/891780D5-F902-4A27-9B28-ECD1653CDB44.png)

The value action at the current time step is equal to immediate reward + the value action where you end up.


#### Optimal Value Function

The optimal state-value function v∗(s) is the maximum value function over all policies:
![](Lecture%202,%20Markov%20Processes/66F4731F-A734-462F-B9A5-C6541D36C305.png)


The optimal action-value function q∗(s, a) is the maximum action-value function over all policies:
![](Lecture%202,%20Markov%20Processes/909D11C9-4581-4E0D-9505-C2FB5983CAE1.png)


If you have q start, you're done. This will say you basically if you have two actions to chose from. If you chose action A you will get Return of 90, and if you chose action B of 30. You can compare those and just decide about an action which will bring you more reward.
- - - -

#### Define a partial ordering over policies
![](Lecture%202,%20Markov%20Processes/F7791F61-5D24-48C4-8D16-1E487E76A361.png)

Policy 1is better than policy 2, if value function of policy 1 is equal or better in all states.

For any Markov Decision Process:
	* There exists an optimal policy π∗ that is better than or equal to all other policies, π∗ ≥ π, ∀π
	* All optimal policies achieve the optimal value function
	* All optimal policies achieve the optimal action-value function, `qπ∗(s, a) = q∗(s, a)`

An optimal policy can be found by maximising over `q∗(s, a)`,
![](Lecture%202,%20Markov%20Processes/422930EA-6351-4C71-BF74-59CA0F82DD55.png)

Means that if we know that one action brings more reward than we want to take this action with probability of 1.
	* There is always a deterministic optimal policy for any MDP
	* If we know `q∗(s, a)`, we immediately have the optimal policy
- - - -
### How to find an optimal policy

#### Bellman Optimality Equation for v∗

The optimal value functions are recursively related by the Bellman optimality equations:
![](Lecture%202,%20Markov%20Processes/27FAF908-6544-49F1-A5D5-F4C24D93CE18.png)

We are in state S. Have two actions and instead of taking average of the two action value function, we take a maximum of those.

















