# Lecture 5,  Model-Free Control
#reinforcment-learning/lecture5


### On and Off-Policy Learning

* On-policy learning
	* “Learn on the job”
	* Learn about policy π from experience sampled from π
* Off-policy learning
	* “Look over someone’s shoulder”
	* Learn about policy π from experience sampled from µ
	* as an example can be a robot who observes a human behaviour and learns from this behaviour. 
	*


- - - -

#### Greedy in the Limit with Infinite Exploration (GLIE)

All state-action pairs are explored infinitely many times, to guarantee that we didn't miss out anything. (e greedy policy has this property, every state should be visited)

![](Lecture%205,%20%20Model-Free%20Control/62E4F06C-C8AF-4A54-89D5-AF59B7480538.png)

The policy converges on a greedy policy,

![](Lecture%205,%20%20Model-Free%20Control/C117AE10-9A0D-4695-A58F-4A4E1001FCED.png)


- - - -

### MC vs. TD Control

TD can be important by using it for  off-policy learning

#### Updating Action-Value Functions with Sarsa

We are asking about the state S and a particular action A, then we sampled from the environment what reward we can get from that and what state we end up in. Then we sample at our own policy at the next step:
![](Lecture%205,%20%20Model-Free%20Control/1449DD19-1362-40EF-BFE4-5BAE6E8D7CF4.png)

Sarsa Update:
![](Lecture%205,%20%20Model-Free%20Control/F60C53A2-FD31-48C9-94B2-D8A16493DE92.png)

#### On-Policy Control With Sarsa
Every single time step we will improve out value function.
We are only updating a value function for state action pair at one time step.
We just storing a Q value so we've got a huge table of Q values for all states and actions.
Policy implicitly represented by this Q values.

#### Convergence of Sarsa

	* GLIE sequence of policies `πt(a|s)`
		* meaning that our our policy will eventually converge to an optimal policy
		* i.e. e greedy policy. make sure that we eventually end up greedy.
	* Robbins-Monro sequence of step-sizes `αt`
		* make sure the step sizes are sufficient large
			* so our q value can updated itself for any initial value
		* the changes should become smaller and smaller and eventually vanish.
		* otherwise you still have a noise and jumping around in our policy
		* 
![](Lecture%205,%20%20Model-Free%20Control/2B9C874B-4FA5-41BD-978F-5C79C11592DC.png)
In practice we don't worry about the points above, normally it works anyway.

#### n-Step Sarsa
We have a parameter lambda which says how much we prefer the short-term reward compare to long-term  one. labmda = 1 (MC), lambda = 0. ()

![](Lecture%205,%20%20Model-Free%20Control/8DEBC1D7-B3AD-4E92-A740-A46199989D4B.png)


we have to wait until the end of episode in order to compute our Q:
![](Lecture%205,%20%20Model-Free%20Control/5FE9DD05-DF69-40BF-B0E7-510503C79684.png)




#### Sarsa(λ) Algorithm

At every step we update our value function with respect to eligibility trace. Therefore every pair of action and state can be responsible for an td error.

The Idea of backward view is that once we will reach a state with reward it will have an influence of value function for state before that. I.e. it will propagate reward backward. That way we get much faster flow of information through the time.
**Lambda** set the property of how steadily you want your eligibility traces decay over the state sequence.
More back you look, more variance you will have, because the can be a lot of the random state which aren't influenced on getting the reward --> reducing the bias.

Backward and Forward basically propose the same 
- - - -


### Off-policy Learning

on-policy learning  - the policy I am following is the policy I am learning about.
Yes, by using off-policy learning.
**Why is this important?**
	* Learn from observing humans or other agents
	* Re-use experience generated from old policies π1, π2, ..., πt−1
		* _Is it possible to make use of this additional data(e.g. policy that already learned about) to get a better estimate for a final policy?_
	* Learn about optimal policy while following exploratory policy
		* exploration as we want and and at the same time find a better policy.
	* Learn about multiple policies while following one policy
		* How to learn about different policies without trying this policy out.

* First of all, there's no reason that an agent has to do the greedy action; Agents can explore or they can follow options. This is not what separates on-policy from off-policy learning.

* **The reason that Q-learning is off-policy** is that it updates its Q-values using the Q-value of the next state s′s′ and the greedy action a′a′. In other words, it estimates the return (total discounted future reward) for state-action pairs assuming a greedy policy were followed despite the fact that it's not following a greedy policy.

* **The reason that SARSA is on-policy** is that it updates its Q-values using the Q-value of the next state s′s′ and the current policy's action a″a″. It estimates the return for state-action pairs assuming the current policy continues to be followed.

* **The distinction disappears if the current policy is a greedy policy.** However, such an agent would not be good since it never explores.

We are trying 


By observing some other policy, figuring out how to do better, i.e. find a better  policy.

MC is really bad idea for off-policy.


#### Q-Learning
* We now consider off-policy learning of action-values Q(s, a)
* No importance sampling is required
* Next action is chosen using behaviour policy At+1 ∼ µ(·|St)
* But we consider alternative successor action A0 ∼ π(·|St)
* And update Q(St, At) towards value of alternative action

We have two policies here:
	* behaviour  policy is policy which we are actually following, i.e. taking an actual action 
	* target policy is a policy we want to learn
So the idea behind Q-learning is that we will actually take an Action from our behaviour policy, but we gonna update Q value in the direction that we are taking an action from our target policy.

We take an action from behaviour policy. Observing new State S, reward R. After that we take an another action from our  behaviour policy **But** we will update our q value based on an action that we would take if the target policy is followed.
	

Special thing about Q-learning that  both behaviour and target policy will improve
Q-learning is a special case where we will learn about out greedy policy while following an exploratory policy.


We can think of TD as of  sample of bellman expectation(optimality equation)  which Dynamic programming uses the exhaustive search on it.











