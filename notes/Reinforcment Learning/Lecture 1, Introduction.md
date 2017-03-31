# Lecture 1, Introduction
#reinforcment-learning/lecture1

**What makes reinforcement learning different from other machine learning paradigms?**
	* There is no supervisor, only a reward signal
	* Feedback is delayed, not instantaneous
	* Time really matters (sequential, non i.i.d data)
	* Agent’s actions affect the subsequent data it receives


Rewards:
	* A reward Rt is a scalar feedback signal
	* Indicates how well agent is doing at step t
	* The agent’s job is to maximise cumulative reward

#### Reward hypothesis
	_All goals can be described by the maximisation of expected cumulative reward_


#### Sequential Decision Making
	**Goal**:  _select actions to maximise total future reward_
	* Actions may have long term consequences
	* Reward may be delayed
	* It may be better to sacrifice immediate reward to gain more long-term reward


### Agent and Environment
	* At each step t the **agent**:
		* Executes action At
		* Receives observation Ot
		* Receives scalar reward Rt
	* The **environment**:
		* Receives action At
		* Emits observation Ot+1
		* Emits scalar reward Rt+1
	* t increments at env. step

![](Lecture%201,%20Introduction/76989E9A-B858-491C-8EBD-106B7603F33C.png)



The history is the sequence of observations, actions, rewards:
`Ht = O1, R1, A1, ..., At−1, Ot, Rt`


What happens next depends on the history:
	* The agent selects actions
	* The environment selects observations/rewards

**State** is the information used to determine what happens next.

Formally, state is a function of the history:
	` St = f (Ht) `

The **environment state**  `Set` is the environment’s private representation:
	* i.e. whatever data the environment uses to pick the next observation/reward
	* The environment state is not usually visible to the agent
	* Even if `Set` is visible, it may contain irrelevant information

#### Agent State
	* The agent state `Sat` is the agent’s internal representation
	* i.e. whatever information the agent uses to pick the next action
	* i.e. it is the information used by reinforcement learning algorithms
	* It can be any function of history:
		 `Sat = f (Ht)`

An information state (a.k.a. **Markov state**) contains all useful information from the history.
A state `St` is **Markov** if and only if
	`P[St+1 | St] = P[St+1 | S1, ..., St]`

**“The future is independent of the past given the present”**
	* Once the state is known, the history may be thrown away
	* i.e. The state is a sufficient statistic of the future
	* The environment state `Set` is Markov The history `Ht` is Markov


#### Partially Observable Environments
	* Partial observability: agent indirectly observes environment:
		* A robot with camera vision isn’t told its absolute location
		* A trading agent only observes current prices
		* A poker playing agent only observes public cards
	* Now agent state != environment state

An RL agent may include one or more of these components:
	* **Policy**: agent’s behaviour function
	* **Value function**: how good is each state and/or action
	* **Model**: agent’s representation of the environment

	
#### Policy
	* A policy is the agent’s behaviour
	* It is a map from state to action, e.g.
	* Deterministic policy: `a = π(s)`
	* Stochastic policy: `π(a|s) = P[At = a|St = s]`

#### Value Function
	* Value function is a prediction of future reward
	* Used to evaluate the goodness/badness of states
	* And therefore to select between actions, e.g.
	`vπ(s) = Eπ [Rt+1 + γ * Rt+2 + γ^2 * Rt+3 + ... | St = s ]`
	* Value function is bound to a policy of an agent
	* γ is discount coefficient , if it is equal to 0 , so we care only about immediate reward
	* How much reward in **the future**  we can get from this state forward if we will follow this particular policy
	* compare those is a basis for a making good decisions


#### Model

* A model predicts what the environment will do next
* P predicts the next state
* R predicts the next (immediate) reward, e.g.

![](Lecture%201,%20Introduction/271AB50E-008F-4EAF-AEB5-1319BAAB2596.png)



### Categorising RL agents

#### Value Based
	* No Policy (Implicit)
	* Value Function
	* always pick you acton greedy with respect to your value function, since there is no policy
#### Policy Based
	* Policy
	* No Value Function
	* Without explicitly storing the value function
#### Actor Critic
	* Policy
	* Value Function
	* combined two above agents together


Two fundamental distinctions in RL:
	* **Model Free**
		* Policy and/or Value Function
		* No Model
		* without building a representation of environment
	* **Model Based**
		* Policy and/or Value Function
		* Model	

#### RL Agent Taxonomy

![](Lecture%201,%20Introduction/31CDF483-1576-41B6-8D41-D44B56DEE23B.png)



#### Two fundamental problems in sequential decision making

**Reinforcement Learning:**
	* The environment is initially unknown
	* The agent interacts with the environment
	* The agent improves its policy
	* Rules are unknown, learns about environment by interacting with it.
**Planning:**
	* A model of the environment is known
	* The agent performs computations with its model (without any external interaction)
	* The agent improves its policy
		* a.k.a. deliberation, reasoning, introspection, pondering, thought, search
	* Basically we told in advance about environment, we can take a look in advance what will happen if we will execute an particular action.
	* Plan ahead to find optimal policy (e.g. tree search)

	
	

#### Exploration and Exploitation
	* Reinforcement learning is like trial-and-error learning
	* The agent should discover a good policy
	* From its experiences of the environment
	* Without losing too much reward along the way

* **Exploration** finds more information about the environment
* **Exploitation** exploits known information to maximise reward
* One has to balance those two things out in order to find a best reward policy.





