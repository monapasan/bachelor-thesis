# Lecture 7, Policy Gradient Methods
	#reinforcment-learning/lecture7

here we are talking how to improve our policy directly by working with policy  instead of q value function.
We adjust our policy in the direction which makes it better.

Ideas is that we will have now a function approximation directly for the policy.

We control the parameters that control the distribution which we choose action from.
Motivation why we want to do that:
	* We want to scale to large complicated environment
	*  for policy we need to store less things therefore make it more compact
		* for example if you 5 metres from the wall you may want to alway go right
		* This can be represented more simple with policy than with value function. 
Here we use gradient ascent algorithm to maximise the reward. 

but the slides on "**compatible function approximation**" is not in the right order and the slides on "**deterministic policy gradient**" is missing.﻿
- - - -
#### Advantages of Policy-Based RL

_Advantages:_
	* Better convergence properties
		* because we working directly with policy 
		* updates are more stable and smooth
	* Effective in high-dimensional or continuous action spaces
		* in Q-learning or Sarsa we always choose action over action space which maximise reward.
		* But what if this procedure itself is expensive when f.i. we can continuous action space.
	* Can learn stochastic policies
_Disadvantages:_
	* Typically converge to a local rather than global optimum
		* value based approaches 
	* Evaluating a policy is typically inefficient and high variance




If you in partial observer MDP then it can be optimal to use a stochastic policy in which case policy based method can do better than value based method where you just act greedy.
Value based policy is near deterministic policy, hence whenever the state aliases occurs it's better to have a stochastic policy computed from policy based methods.

if you remember from Bellman optimality equation that every MDP has an optimal deterministic policy, but in Partially Observer MDP  or when to use function approximation where there is vector holding features(which is equivalent to partially observer environment)  then it can be optimal to use a stochastic policy in which policy search methods method can do better than value search methods. 

#### Policy Objective Functions
	* Goal: given policy πθ(s, a) with parameters θ, find best θ
	* But how do we measure the quality of a policy πθ?
	* In episodic environments we can use the start value
![](Lecture%207,%20Policy%20Gradient%20Methods/672A1BD1-B65D-4A60-AD6A-04D695790CE5.png)
	* In continuing environments we can use the average value: 
![](Lecture%207,%20Policy%20Gradient%20Methods/EA1E9877-182C-409E-BDCA-3A901F5321F5.png)

	* where dπθ (s) is stationary distribution of Markov chain for πθ
		* probability that agent occurs on state S following policy πθ
	* Or the average reward per time-step
		
![](Lecture%207,%20Policy%20Gradient%20Methods/0B9F944B-D629-457F-8BB7-EA23236065C2.png)
**explained**: probability that agent is in state **s** multiplied by sum over actions where sum is probability that agent will take an action **a** from state **s** multiplied by reward observed after taking this action.

πθ(s,a)  - probability that from state S, I will take action a.
####  Policy Optimisation
	* Policy based reinforcement learning is an optimisation problem
	* Find θ that maximises J(θ)
	* Some approaches do not use gradient
		* Hill climbing
		* Simplex / amoeba / Nelder Mead
		* Genetic algorithms
	* Greater efficiency often possible using gradient
		* Gradient descent
		* Conjugate gradient
		* Quasi-newton
	* We focus on gradient descent, many extensions possible
	* And on methods that exploit **sequential structure**
		* We would like to learn within a lifetime of an agent, and not first when agent  is dead.

### Computing Gradients By Finite Differences

* To evaluate policy gradient of πθ(s, a)
* For each dimension k ∈ [1, n]
	Estimate kth partial derivative of objective function w.r.t. θ
	* By perturbing θ by small amount  in kth dimension
![](Lecture%207,%20Policy%20Gradient%20Methods/8774BE98-DB56-4450-98D9-DC1D66BC3C65.png)
	where uk is unit vector with 1 in kth component, 0 elsewhere
* Uses n evaluations to compute policy gradient in n dimensions
* Simple, noisy, inefficient - but sometimes effective
* Works for arbitrary policies, even if **policy is not differentiable**



#### Score Function

We want to take a gradient of our policy. And then we want to take an expectation value of our gradient.

* We now compute the policy gradient analytically
* Assume policy πθ is differentiable whenever it is non-zero 
* and we know the gradient ∇θπθ(s, a)
* Likelihood ratios exploit the following identity:
         
![](Lecture%207,%20Policy%20Gradient%20Methods/268C3B44-9ED8-467B-83FF-E6F89E5399D6.png)

If we rewrite the gradient in this way we are able to take expectations.
Compute expectation in the first way is hard. Compute the expectation in latter case with log is easy, because this is the actual policy we're following, hence this is actually something that we sample from.

#### Example 1, Softmax Policy

	* We will use a softmax policy as a running example
	* Weight actions using linear combination of features φ(s, a)>θ
	* Probability of action is proportional to exponentiated weight:
![](Lecture%207,%20Policy%20Gradient%20Methods/0BF5D20F-322B-4FF0-B52D-D7B03F4D36BD.png)

This tells us how frequently should choose an action for every discrete set of actions.

Softmax in general is a policy that proportional to some exponentiated value.
**What we do here** is just multiply our features vector by weights that we have. And since we have a policy we normalise mentioned before equation by exponentiating it to get a distribution over actions. 

Score function represents **How much more of this feature I do  than usual.** 


####  One-Step MDPs
* Consider a simple class of one-step MDPs
	* Starting in state s ∼ d(s)
	* Terminating after one time-step with reward r = Rs,a
* Use likelihood ratios to compute the policy gradient

![](Lecture%207,%20Policy%20Gradient%20Methods/F58A77F6-6E3A-4977-B70D-77C2CBE7818C.png)

_What is our objective function?_
	* it's **expected** reward that we get following the policy.
	* We want to improve our objective function by applying gradient ascent, hence change the parameters to have a more expected return.
	* Why do we use log form ?
		* We are starting with something which is expectation.
		* after taking a gradient of it we recover something which is still an expectation.
		* In order to have an expectation in our gradient.
		* this is the whole point of it.
		* in details:
			* first sum over S, d(s) is the expectation over state from our policy
			* second sum over a, p(s,a) is the expectation over discrete action space from state s 
	* That means if we want to improve our policy we need to update our objective in direction of of the score function multiplied by reward.
	* Which means that we need only reward that we experiences to compute the score function since we know the policy because we're following it.
	* This give us update our parameters without knowing model(model-free)


#### Policy Gradient Theorem

* The policy gradient theorem generalises the likelihood ratio approach to multi-step MDPs
* Replaces instantaneous reward r with long-term value Qπ(s, a)
* Policy gradient theorem applies to start state objective, average reward and average value objective
**Theorem:**
	* For any differentiable policy πθ(s, a),
	* for any of the policy objective functions J = J1, JavR, or 1/(1−γ) * JavV , the policy gradient is
![](Lecture%207,%20Policy%20Gradient%20Methods/762A09BA-5B80-46BE-867D-FB167691FBA4.png)
_It tells us:_
	* How to adjust the policy to get more or less of that particular action multiplied by q value of that action.
	* Meaning:
		* that you get a positive reward that you will try to update this particular action in positive direction
		* or if you get a negative reward that decrease this particular action in our policy.

The equation  above is similar to  supervised learning but instead of updating the gradient in direction of q value, in supervised learning the parameters are updated in direction of what the teacher tells you.


#### Monte-Carlo Policy Gradient (REINFORCE)

* Update parameters by stochastic gradient ascent
* Using policy gradient theorem
* Using return vt as an unbiased sample of Qπθ (st, at)
* 
![](Lecture%207,%20Policy%20Gradient%20Methods/B8922BDE-7F03-4A93-9D05-830A6D03FD77.png)

Algorithm: 

![](Lecture%207,%20Policy%20Gradient%20Methods/7C7F7EDE-F2EC-43E0-8E53-63ADA34CDB16.png)

Procedure:
	* Run the episode, 
	* store all states, actions, returns
	* after end of episode update the parameters in direction of the return.
	* vt is here reward starting from state St.
- - - -
### Reducing Variance Using a Critic

* Monte-Carlo policy gradient still has high variance
	* Using MC takes also a lot of time to train a model.
* We use a critic to estimate the action-value function,

![](Lecture%207,%20Policy%20Gradient%20Methods/C4E4A1B7-5318-4A52-8CE8-4BE92F966C1A.png)

* Actor-critic algorithms maintain two sets of parameters
	* _Critic_ Updates action-value function parameters w
	* _Actor_ Updates policy parameters θ, in direction suggested by critic
* Actor-critic algorithms follow an approximate policy gradient:
![](Lecture%207,%20Policy%20Gradient%20Methods/7B5388AA-F229-40C5-ACD6-31CB8791C5B3.png)

Now we need to store two sets of parameters:
	* one is for q value, a Critic
		* given by function approximator. 
		* which says or is responsible for how good our reward is.
		* e.g. which of the feature is good, or bad.
		* Suggesting the actor the directions in which policy parameters should be updated.
	* one is for policy, an Actor
		* decide which action to pick
		* will update the policy parameters by suggested value of Critic.

#### Estimating the Action-Value Function

	* The critic is solving a familiar problem: policy evaluation
		* we are not looking for q*, an optimal action value function
		* but just making an evaluation of a current policy.
	* How good is policy πθ for current parameters θ?
	* This problem was explored in previous two lectures, e.g.
		* Monte-Carlo policy evaluation
		* Temporal-Difference learning
		* TD(λ)
	* Could also use e.g. least-squares policy evaluation	



#### Action-Value Actor-Critic


* Simple actor-critic algorithm based on action-value critic
* Using linear value fn approx. Qw (s, a) = φ(s, a)>w
	* Critic Updates w by linear TD(0)
	* Actor Updates θ by policy gradient
![](Lecture%207,%20Policy%20Gradient%20Methods/12278C2B-1F1B-4BA7-86A9-A36DF78E2815.png)


**Note:** if we increase `α` - step size to infinite then we will get a greedy policy. Step size controls this smoothness


#### Reducing Variance Using a Baseline.
We want to make things better by reducing the variance of the solution:
	* We subtract a baseline function B(s) from the policy gradient
	* This can reduce variance, without changing expectation 
![](Lecture%207,%20Policy%20Gradient%20Methods/90E9F0AB-583E-468C-AAA1-07FAD907D373.png)
Baseline function is independent from θ, therefore we can pull it forward before the gradient. Then we can also pull the gradient before the sum sign and as policy over all action always sums up to one and gradient over constant is zero, the whole term is equal to zero.


**The idea is to subtract from our gradient the form above that won't change an expected value but may reduce a variance.**
One reward may be 1000 and another is -1. We want to rescale things by introducing the baseline function.

* A good baseline is the state value function `B(s) = Vπθ (s)`
	* as you may remember the value function says how good it is to be in a particular state independent of action that you will take from that state.
* So we can rewrite the policy gradient using the advantage function A πθ (s, a)
![](Lecture%207,%20Policy%20Gradient%20Methods/E25EC4FB-0ABD-44BE-A79D-95C4B6E7A42C.png)
**Which basically says how good is that to take an action a than any action in general.**
**Then update: is how to adjust our policy to achieve than action a**
	
There are two ways of estimating the advantage function:

#### Estimating the Advantage Function(1)
	* The advantage function can significantly reduce variance of policy gradient
	* So the critic should really estimate the advantage function
	* For example, by estimating both Vπθ (s) and Qπθ (s, a)
	* Using two function approximators and two parameter vectors:
![](Lecture%207,%20Policy%20Gradient%20Methods/7C50EFF9-7D8C-49DF-9C44-20085A77FB1B.png)
And updating both value functions by e.g. TD learning.

This way introduces a new parameter and therefore new bias. Can be the a better solution ?

#### Estimating the Advantage Function (2)

For the true value function `Vπθ (s)`, the TD error `δπθ`:
![](Lecture%207,%20Policy%20Gradient%20Methods/DD0E0A01-83C7-4665-BFBA-C9D144B3FD5F.png)
_is an unbiased estimate of the advantage function_
![](Lecture%207,%20Policy%20Gradient%20Methods/63EB269B-E983-47CC-9C21-A1A9541B7647.png)
`Vπθ (s)` conditioned on S is fixed and has no expected value therefore we can pull it out of the expectation. The First before - is a definition of q value function from bellman equation.

So we can use the TD error to compute the policy gradient:
![](Lecture%207,%20Policy%20Gradient%20Methods/7ECF52F5-D9BE-4C13-BE7A-32A4F1A920E4.png)
In practice we can use an approximate TD error:
![](Lecture%207,%20Policy%20Gradient%20Methods/41634A36-8B4A-4F9B-80EC-6845F84324A6.png)
This approach only requires one set of critic parameters v

- - - -

#### Critics at Different Time-Scales

Critic can estimate value function Vθ(s) from many targets at different time-scales From last lecture...
we can replace return Vt with every target that we learned in previous lectures:
	* td target
	* forward view - td lambda target
	* For backward-view TD(λ), we use eligibility traces:
```
	δt = rt+1 + γV(st+1) − V(st)
	et = γλet−1 + φ(st)
	∆w = α * δt * et
```

#### Actors at Different Time-Scales
The policy gradient can also be estimated at many time-scales:
* Like backward-view TD(λ), we can also use eligibility traces:
	* By equivalence with TD(λ), substituting φ(s) = ∇θ log πθ(s, a)
![](Lecture%207,%20Policy%20Gradient%20Methods/1C6125E1-7DBF-40F5-9490-B1647FE71982.png)
We update our traces in direction of our score function, which keeps larger value in action which were most largely, most frequently.


- - - -
#### Bias in Actor-Critic Algorithms

* Approximating the policy gradient introduces bias
	* as well as approximating the q value function.
	* since we don't use a true value function anymore and instead using the approximation of it and hope that it is still the right gradient to follow.
* A biased policy gradient may not find the right solution
	* e.g. if Qw (s, a) uses aliased features, can we solve gridworld example?
* Luckily, if we choose value function approximation carefully
* Then we can avoid introducing any bias
* i.e. We can still follow the exact/ true policy gradient

#### Compatible Function Approximation
the features that we use are themselves are score function, then we can actually guarantee that we don't affect our true policy gradient.
_**If** the following two conditions are satisfied:_
	1. Value function approximator is compatible to the policy
		The features that we use in action value function approximation are themselves are score function.![](Lecture%207,%20Policy%20Gradient%20Methods/7F5A79EF-1916-468F-8D0A-302CCC152418.png)
	2. Value function parameters w minimise the mean-squared error
![](Lecture%207,%20Policy%20Gradient%20Methods/CF15C81A-5E0F-43DA-A671-4BEF4B19455A.png)

**Then** the policy gradient is exact	:
![](Lecture%207,%20Policy%20Gradient%20Methods/F65D3DD8-AC22-499C-AA71-A5F1EC434089.png)



#### Deterministic Policy 

Why?
	* We take a policy (gaussian) which just has some mean and variance
	* and in particular just a noise 
	* We estimating our policy by sampling  that noise again and again
	* And we're taking an expectation of the noise
	* And it can be in some case a really bad idea.
	* As we go narrow in finding our right policy, the noise can blow up to infinity. 


### Summary of Policy Gradient Algorithms

![](Lecture%207,%20Policy%20Gradient%20Methods/98F71C4F-CC88-4D7F-8DB4-8C6AEF493A7E.png)

* Each leads a stochastic gradient ascent algorithm
* Critic uses policy evaluation (e.g. MC or TD learning) to estimate `Qπ(s, a), Aπ(s, a) or Vπ(s	`
All of these above is essentially  different variance of the same algorithm(idea). 

