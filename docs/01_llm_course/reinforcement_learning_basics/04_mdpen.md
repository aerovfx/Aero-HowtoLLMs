
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [01 llm course](../index.md) > [reinforcement learning basics](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
First, MDPs are used to model decision-making problems where outcomes depend probabilistically on the current state. They are central to reinforcement learning because they provide a framework for agents to make optimal decisions in uncertain environments.

Alright, so let's break it down. An MDP is typically defined by five elements: the set of states $S$, the set of actions $A$, the transition probabilities (P(s,a)), the reward function (R(s,a)), and the discount factor $γ$. Let me write that down:

- **States $S$**: These are the different situations or environments the agent might encounter. For example, in a game, states could represent positions on a board.
  
- **Actions $A$**: These are the possible moves or decisions an agent can make from any given state. Continuing with the game example, actions could be moving pieces or making strategic choices.

- **Transition Probabilities (P(s,a))**: This represents the probability of transitioning from one state to another when a specific action is taken. Mathematically, P(s', s, a) denotes the probability of moving to state s' from state s upon taking action a.

- **Reward Function (R(s,a))**: This function assigns a numerical reward to each possible state-action pair. It quantifies how good or bad it is for the agent to take an action in a particular state. R(s,a) could be +1 for winning a game, -1 for losing, etc.

- **Discount Factor $γ$**: A parameter that determines the importance of future rewards relative to immediate ones. It's usually between 0 and 1. For example, γ = 0.9 means that future rewards are valued at 90% of their face value.

Now, how do these elements come together? In reinforcement learning, an agent interacts with its environment by taking actions in states and receiving rewards based on the transitions and outcomes. The goal is for the agent to learn a policy $π$ that maximizes the cumulative reward over time.

A policy π: S → A is a function that takes a state and returns the probability distribution of actions to take from that state. Sometimes policies are deterministic, meaning they select one action with certainty, or stochastic, selecting actions probabilistically.

The objective in reinforcement learning is often to maximize the expected cumulative reward starting from an initial state s0. This cumulative reward can be represented as:

$$
G = R_1 + γR_2 + γ^2R_3 + ...
$$

Where $R_t$ is the reward received at time t.

To find the optimal policy, we often need to evaluate the value of a given policy. The value function V$s$ for a state s under a policy π is defined as the expected cumulative reward starting from state s and following policy π thereafter:

$$
V_π(s) = E[R_t | s_0 = s, π]
$$

This can be expanded using the Bellman equation, which incorporates transitions and rewards:

$$
V_π(s) = R(s, a) + γE[V_π(s') | s, a, π]
$$

Where $s'$ is the next state after taking action a from state s.

In dynamic programming approaches to solving MDPs, we use methods like value iteration and policy iteration. Value iteration updates an estimate of the value function iteratively:

$$
V_{k+1}(s) = max_a [ R(s,a) + γ \sum_{s'} P(s'|s,a) V_k(s') ]
$$

Here, $a$ is chosen to maximize the expected return.

Policy iteration alternates between evaluating a policy and improving it. Evaluation involves computing the value function for the current policy, while improvement involves changing the policy based on this evaluation until it converges to an optimal policy.

Another important concept is Q-learning, which uses the Q-value function Q(s,a) that represents the expected cumulative reward starting from state s, taking action a, and then following policy π:

$$
Q_π(s,a) = R(s,a) + γ \sum_{s'} P(s'|s,a) V_π(s')
$$

In practice, Q-learning can be implemented without knowing the exact transition probabilities, making it suitable for environments where such information is not readily available.

Let me think about a concrete example. Suppose we're training an agent to play a simple game like a grid maze. The states could be the positions on the grid, actions could be moving up, down, left, or right, rewards could be given when reaching certain positions, and transitions depend on whether the move is possible (e.g., not hitting walls).

Using Q-learning, the agent would update its Q-values based on experiences:

$$
Q(s,a) = Q(s,a) + α [ r + γ max_{a'} Q(s',a') - Q(s,a) ]
$$

Here, α is the learning rate, determining how much new information overrides old data.

This iterative process allows the agent to learn the optimal strategy by exploring the environment and updating its understanding of state-action-reward relationships.

In summary, MDPs provide a formal structure for reinforcement learning problems. They define the probabilistic interactions between an agent and an environment, enabling systematic approaches to discover optimal policies that maximize cumulative rewards.
</think>

An **MDP (Markov Decision Process)** is a mathematical framework used to model decision-making processes in environments where outcomes are uncertain. It is central to reinforcement learning, providing a structured approach for agents to make optimal decisions.

### Components of an MDP:
1. **States $S$:** The set of possible situations or contexts the agent can encounter.
2. **Actions $A$:** The set of possible decisions or actions the agent can take in any given state.
3. **Transition Probabilities (P(s,a)):** The probabilities of moving from one state to another when a specific action is taken.
4. **Reward Function (R(s,a)):** A function that assigns rewards to each state-action pair, indicating how good or bad taking an action in a state is.
5. **Discount Factor $γ$:** A parameter (0 ≤ γ ≤ 1) that determines the weight of future rewards compared to immediate ones.

### Objective:
The goal for the agent is to learn a policy $π$, which maps states to actions, maximizing the expected cumulative reward over time.

### Key Equations and Concepts:

- **Cumulative Reward:**

$$
G = R_1 + γR_2 + γ^2R_3 + ...
$$

- **Value Function (V(s)):** The expected cumulative reward starting from state s under policy π:

$$
V_π(s) = E[R_t | s_0 = s, π]
$$

- **Bellman Equation:**

$$
V_π(s) = R(s,a) + γE[V_π(s') | s, a, π]
$$

- **Value Iteration Update:**

$$
V_{k+1}(s) = max_a [ R(s,a) + γ \sum_{s'} P(s'|s,a) V_k(s') ]
$$

- **Q-value Function (Q(s,a)):** Represents the expected cumulative reward starting from state s, taking action a, and following policy π:

$$
Q_π(s,a) = R(s,a) + γ \sum_{s'} P(s'|s,a) V_π(s')
$$

- **Q-Learning Update:**

$$
Q(s,a) = Q(s,a) + α [ r + γ max_{a'} Q(s',a') - Q(s,a) ]
$$

Here, α is the learning rate.

### Example:
In a grid maze game, states are positions on the grid, actions are movements (up, down, left, right), rewards are given for reaching certain positions, and transitions depend on avoiding walls. Using Q-learning, the agent updates its Q-values based on experiences to learn optimal strategies.

### Summary:
MDPs formalize decision-making in uncertain environments, enabling systematic approaches like value iteration and Q-learning to discover optimal policies that maximize cumulative rewards.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [01 whatisreinforcementlearningen](01_whatisreinforcementlearningen.md) | [Xem bài viết →](01_whatisreinforcementlearningen.md) |
| [01 whatisreinforcementlearningvi](01_whatisreinforcementlearningvi.md) | [Xem bài viết →](01_whatisreinforcementlearningvi.md) |
| [02 bellman equationvi](02_bellman_equationvi.md) | [Xem bài viết →](02_bellman_equationvi.md) |
| [02 bellmanequationen](02_bellmanequationen.md) | [Xem bài viết →](02_bellmanequationen.md) |
| [03 the plan in plankton sattacken](03_the_plan_in_plankton_sattacken.md) | [Xem bài viết →](03_the_plan_in_plankton_sattacken.md) |
| [03 the plan in plankton sattackvi](03_the_plan_in_plankton_sattackvi.md) | [Xem bài viết →](03_the_plan_in_plankton_sattackvi.md) |
| 📌 **[04 mdpen](04_mdpen.md)** | [Xem bài viết →](04_mdpen.md) |
| [04 mdpvi](04_mdpvi.md) | [Xem bài viết →](04_mdpvi.md) |
| [05 policyvsplanvi](05_policyvsplanvi.md) | [Xem bài viết →](05_policyvsplanvi.md) |
| [📘 Khóa học: Học Sâu Học Tăng Cường (Deep Reinforcement Learning)](06_deep_reinforcement_learning_course.md) | [Xem bài viết →](06_deep_reinforcement_learning_course.md) |
| [📂 Module: Reinforcement_Learning_Basics](README.md) | [Xem bài viết →](README.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
