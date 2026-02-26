
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../index.md) > [01 LLM Course](../../index.md) > [Reinforcement Learning Basics](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../../index.md)
- [📚 Module 01: LLM Course](../../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
**Bellman's Equation**
=====================

Bellman's Equation is a mathematical formula used in Reinforcement Learning to compute the optimal value of a policy (policy) in an environment. The equation was first developed by Richard Bellman in 1957.

**Bellman's Equation**
---------------------

Bellman's Equation can be written as:

V(s) = max[ r + γ V(s') ]

where:

* `V(s)` is the optimal value of the policy in state `s`
* `r` is the reward received in state `s`
* `γ` is the discount factor, which represents the weight of future rewards
* `s'` is the next state after taking action `a` in state `s`

**Interpretation**
-----------------

Bellman's Equation represents the process of finding the optimal value of a policy in an environment. The equation shows that the optimal value of a policy in state `s` is computed by adding the reward received in state `s` to the optimal value of the policy in the next state `s'`, and multiplying it by the discount factor `γ`.

**Example**
-----------

Let's consider a simple environment with two states: `s1` and `s2`, and two actions: `a1` and `a2`. We want to find the optimal value of the policy at state `s1`.

Bellman's Equation would be written as:

V(s1) = max[ r1 + γ V(s2), r2 + γ V(s1) ]

In this case, we need to compute the optimal value of the policy at state `s1` by comparing the values of two possible policies: one that takes action `a1` in state `s2`, and another that takes action `a2` in state `s1`.

**Using Bellman's Equation**
---------------------------

Bellman's Equation is widely used in Reinforcement Learning to find the optimal value of a policy in an environment. It can be used to solve problems such as:

* Finding the optimal policy in a specific environment
* Evaluating the performance of a policy in an environment
* Discovering new policies by computing their optimal values

However, Bellman's Equation also has some limitations, such as:

* Not being able to handle environments with many states or actions
* Requiring specialized techniques to handle identical cases

In summary, Bellman's Equation is a fundamental formula in Reinforcement Learning that helps us find the optimal value of a policy in an environment.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
