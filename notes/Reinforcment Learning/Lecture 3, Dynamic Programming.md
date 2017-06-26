# Lecture 3, Dynamic Programming
#reinforcment-learning/lecture3


Dynamic Programming is a very general solution method for problems which have two properties:
	* Optimal substructure
		* Principle of optimality applies
		* Optimal solution can be decomposed into subproblems
	* Overlapping subproblems
		* Subproblems recur many times
		* Solutions can be cached and reused
	* Markov decision processes satisfy both properties
		* Bellman equation gives recursive decomposition
		* Value function stores and reuses solutions

Dynamic programming assumes full knowledge of the MDP.
It is used for **planning** in an MDP.
For prediction: Output: value function vπ
Output: optimal value function v∗ and: optimal policy π∗
- - - -
## Iterative Policy Evaluation
How to figure out the value of my current policy.

**Problem**: evaluate a given policy π
**Solution**: iterative application of Bellman expectation backup

Using synchronous backups,
	* At each iteration k + 1
	* For all states s ∈ S
	* Update vk+1(s) from vk (s0) 
	* where s0 is a successor state of s.

![](Lecture%203,%20Dynamic%20Programming/28662001-D87F-4716-B572-ACBEDE98B673.png)
How to find a value of policy:
	- [ ] Initialise a value with some initial value (i.e. 0)
	- [ ] Use update equation given above where k+1 is the new update. 
Even though we evaluating one policy, we can use the results to find a new policy.



### Policy Iteration

**Any value function can be used to compute the better value function.**

Given a policy π
Evaluate the policy π
`vπ(s) = E [Rt+1 + γRt+2 + ...|St = s]`
Improve the policy by acting greedily with respect to vπ, `π' = greedy(vπ)`
	* In Small Gridworld improved policy was optimal, `π' = π∗`
	* In general, need more iterations of improvement / evaluation
	* **But this process of policy iteration always converges to π∗**

Definition of acting greedy: look ahead for all possible actions, compare the value of the states, chose the best ones possible.

Convergence rate is independent of initial value and policy function.

The idea is based on cycle of evaluation and improvement.
* evaluate Policy
* Improve it.
* is value function optimal ? go to the first step : stop

Acting greedily means:
	* we look at the value of being in particular step and doing a particular action and then following your policy after that.
	* pick action in a way that give us the maximum action value
	* the actions we are picking, getting us most `qπ`


Why don't we have a maximum optima in the improvement cycle:
	* Partial ordering over the policies.
	* So we have different policies, which have different values
	* And then we just compare it based on order.

#### Principle of Optimality

**Why?** 
	* Evaluation steps can be a waste of time when policy stays the same.
Can we shorten our evaluation process and using approximate policy evaluation in a loop?

Idea is to **modified policy iteration**:
	* stop condition 
		* define an E, delta
		* if policy update(improvement) is less than above defined delta, stop evaluation of a policy
	* simply define number of iteration n 
		* after n iteration, we stop evaluation of a policy
	* or define n=1, and update a policy after every iteration.

 Principle of Optimality from react programming:
	* 	you don't have to consider the whole problem at once
	* you can divide it into a bunch of similar subproblem
	* and solve this subproblem  

Any optimal policy can be subdivided into two components:
	* An optimal first action A∗
	* Followed by an optimal policy from successor state S	

![](Lecture%203,%20Dynamic%20Programming/26262323-63C3-4AE0-AF8A-BAD67F58A6B1.png)



### Deterministic Value Iteration

* If we know the solution to subproblems v∗(s0)
* Then solution v∗(s) can be found by one-step lookahead:
![](Lecture%203,%20Dynamic%20Programming/1A239130-312B-4CC7-B7AA-19AE8279D1F2.png)

* The idea of value iteration is to apply these updates iteratively
* Intuition: start with final rewards and work backwards
	* imagine you are at the "end" of MDP, and have only two possible action where you different reward and it's kinda clear what to do there.
	* from this state one can do a one-step lookahead, to identify best possible value function from step before. So basically working backwards in MDP
* Still works with loopy, stochastic MDPs
	* The problem can be that there is no goal in MDP, or no goal in MDP that is known to us. Because we  update every value in every iteration, this makes goal reachable anyway and can find a solution for MDP.



##  Value Iteration

**Problem**: find optimal policy π
**Solution**: iterative application of Bellman optimality backup
	v1 → v2 → ... → v∗
* Using synchronous backups
	* At each iteration k + 1
	* For all states s ∈ S
	* Update vk+1(s) from vk (s0)
* Unlike policy iteration, there is no explicit policy 
* Intermediate value functions may not correspond to any policy
	* In contrast to policy iteration, the update value function may not have any policy which corresponds to this value function.
* it combines generating greedy policy and evaluation it into one single step


#### Synchronous Dynamic Programming Algorithms

![](Lecture%203,%20Dynamic%20Programming/BC47D1C0-8A10-40BE-9B77-8136B2166FB0.png)

Problem:
	* Prediction
		* given MDP with a policy
		* How much reward do I get from a given policy
		* output: a value function
	* Control
		* given MDP
		* output optimal policy


#### Asynchronous Dynamic Programming
Why? With synchronous update there can be too many computations, so to reduce it we can use asynchronous value update.

	* DP methods described so far used synchronous backups
	* i.e. all states are backed up in parallel
	* Asynchronous DP backs up states individually, in any order
	* For each selected state, apply the appropriate backup
	* Can significantly reduce computation
	* Guaranteed to converge if all states continue to be selected

	
**Different method** to pick the state, from which method will update:
Three simple ideas for asynchronous dynamic programming:
	* In-place dynamic programming
	* Prioritised sweeping
	* Real-time dynamic programming

#### Prioritised Sweeping
The idea is to keep a priority queue :

* Use magnitude of Bellman error to guide state selection, or any other error which will make sense
* Backup the state with the largest remaining Bellman error
* Update Bellman error of affected states after each backup
* Requires knowledge of reverse dynamics (predecessor states)
* Can be implemented efficiently by maintaining a priority queue

#### Real-Time Dynamic Programming

Select the state that we actually visited, i.e. sample over a MDP. Let agent explore the states, chose most relevant ones and update those:
* Idea: only states that are relevant to agent
* Use agent’s experience to guide the selection of states
* After each time-step St, At, Rt+1
* Backup the state St:
![](Lecture%203,%20Dynamic%20Programming/9BF61D71-6E36-47F2-B9C0-EACEFEFEDD3E.png)



#### Full-Width Backups

We can not 
* DP uses full-width backups
* For each backup (sync or async)
* Every successor state and action is considered
* Using knowledge of the MDP transitions and reward function
* DP is effective for medium-sized problems(millions of states)
* For large problems DP suffers Bellman’s curse of dimensionality
	* Number of states n = |S| grows exponentially with number of state variables
* **Even one backup can be too expensive**
	* Solution for this is sampling. Sample over the MDP.
	* Because we sampling from the environment we don't need to know the model of the environment.



In this example, there are four rewarding states (apart from the walls), one worth +10 (at position (9,8); 9 across and 8 down), one worth +3 (at position (8,3)), one worth -5 (at position (4,5)) and one -10 (at position (4,8)). In each if these states the agent gets the reward when it carries out an action in that state (when it leaves the state, not when it enters). 
You can see these when you press "step" once after a "reset". 
	* If "Absorbing states" is checked, the positive reward states are absorbing; 
		* the agent gets no more rewards after entering those states. 
	* If the box is not checked,
		* when an agent reaches one of those states, no matter what it does at the next step, it is flung, at random, to one of the 4 corners of the grid world.

 **[Does this make a difference? Try the non-discounted case (i.e., where discount=1.0).]**

The initial discount rate is 0.9. You can either type in a new number or increment or decrement it by 0.1. 

It is interesting to try the value iteration at different discount rates. Try 0.9, 0.8, 0.7, 0.6, (and 0.99, 0.995 and 1.0 when there is an absorbing state). Look, in particular, at the policy for the points 2&3-across and 6&7-down, and around the +3 reward. 

**Try 0.9, 0.8, 0.7, 0.6, (and 0.99, 0.995 and 1.0 when there is an absorbing state)**

* _Can you explain why the optimal policy changes as the value function is built?_
	* Because some cell have a impact of the value of others.
	* Reward +10 at one cell is propagated to cells close to it first.
	* Then cells which more far away, getting this impact as well.
	* System starts to "understand" that it's better to avoid cells which has negative reward, and that is better to move in the direction of the cells with a positive reward.
* _Can you explain the direction in the optimal policy for each discount rate?_
	* 1.0
		* with absorbing state
			* with discount rate, means that future rewards are equally important for the system
			* therefore system will prefer to achieve cell with maximum reward and try to avoiding any cell that can end the game.
	* 0.9
		* 
	* 0.8
	* 0.7
	* 0.6
with values less than 1.0, system is being more careful, and start taking into account the difference between big loss and small loss. System tries to avoid huge losses in contrast to small losses. 


You can also change the initial value for each state (i.e, the value that value iteration starts with). 
_See if changing this value affects the final value or the rate that we approach the final value._ 




	









	