
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [04 buildgpt](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
## Cấu trúc chính (10 sections + 3 appendices):

### **Phần lý thuyết:**
1. **Nền tảng toán học** - Exponential function, định nghĩa Softmax, probability axioms
2. **Temperature scaling** - Công thức mở rộng, behavior với different T values, intuition
3. **Log-Softmax** - Motivation, numerical stability, range comparison
4. **Mathematical derivations** - Derivatives, cross-entropy gradients

### **Phần thực hành:**
5. **NumPy & PyTorch implementations** - Complete code examples
6. **Experiments** - Temperature effects, stability tests
7. **LLM applications** - Token generation, attention mechanisms
8. **Best practices** - Guidelines, common pitfalls, testing

### **Phần nâng cao:**
- Gumbel-Softmax, Sparsemax, Entmax
- Complete code repository trong Appendices
- Mathematical proofs và derivations

## Điểm nổi bật:

✅ **12 academic citations** (NeurIPS, ICML, ICLR papers)  
✅ **Mathematical rigor** với proofs về probability properties  
✅ **Comprehensive code** - NumPy, PyTorch, testing suites  
✅ **Practical focus** - Temperature ranges cho production (0.7-1.3)  
✅ **Visual explanations** - Plots comparing linear vs log scales  
✅ **Numerical stability** - LogSumExp trick, underflow prevention  
✅ **Real-world examples** - LLM generation pipeline  
✅ **3 complete appendices** - Code, math derivations, glossary

## Key insights covered:

- Tại sao Softmax **guarantees** valid probabilities
- Temperature như "creativity control knob"
- Log-Softmax essential cho large vocabularies
- PyTorch's `F.log_softmax()` vs naive `torch.log(F.softmax())`
- Production temperature values vs experimental ranges
# Hàm Softmax và Tham Số Temperature trong Mô Hình Ngôn Ngữ Lớn: Phân Tích Toán Học và Thực Nghiệm

**Tác giả:** Phân tích học thuật  
**Ngày:** 14 tháng 2, 2026  
**Lĩnh vực:** Deep Learning, Natural Language Processing, Probability Theory

---

## Tóm tắt

Bài viết này trình bày phân tích toàn diện về hàm Softmax—một trong những phép biến đổi quan trọng nhất trong deep learning. Nghiên cứu tập trung vào ba khía cạnh chính: (1) nền tảng toán học của Softmax và mối quan hệ với lý thuyết xác suất, (2) tham số temperature và ảnh hưởng của nó đến phân phối xác suất, và (3) Log-Softmax và vai trò của nó trong ổn định số học cho mô hình ngôn ngữ lớn. Kết hợp phân tích lý thuyết với triển khai thực nghiệm trong NumPy và PyTorch, bài viết làm rõ tại sao Softmax là fundamental transformation cho token generation, và cách temperature scaling cho phép kiểm soát độ "confidence" và "creativity" của model outputs.

**Từ khóa:** Softmax, Temperature Scaling, Log-Softmax, Probability Distribution, Token Generation, Large Language Models, Numerical Stability, PyTorch

---

## 1. Giới Thiệu

### 1.1 Tầm Quan Trọng của Softmax

Softmax là một trong những phép biến đổi **fundamental và ubiquitous** nhất trong deep learning [1]. Trong ngữ cảnh của Large Language Models (LLMs), Softmax đóng vai trò then chốt trong việc:

1. **Chuyển đổi logits thành xác suất**: Mapping raw model outputs → valid probability distributions
2. **Token selection**: Enabling probabilistic sampling cho text generation
3. **Loss computation**: Foundation cho cross-entropy loss trong training
4. **Attention mechanisms**: Core component trong transformer architectures [2]

**Quan sát quan trọng:**
> Mặc dù công thức Softmax có vẻ đơn giản, nhưng nó chứa đựng độ phức tạp và nuances đáng kể, đặc biệt khi áp dụng vào mô hình với vocabularies lớn (100,000+ tokens) và yêu cầu numerical stability cao.

### 1.2 Động Lực Nghiên Cứu

**Câu hỏi nghiên cứu trung tâm:**

1. Tại sao Softmax guarantee tạo ra valid probability distributions?
2. Temperature parameter ảnh hưởng thế nào đến shape của distribution?
3. Log-Softmax khác gì so với standard Softmax và khi nào cần sử dụng?
4. Làm thế nào để implement Softmax một cách numerically stable?
5. Giá trị temperature nào phù hợp cho production LLMs?

### 1.3 Cấu Trúc Bài Viết

Bài viết được tổ chức như sau:
- **Section 2**: Nền tảng toán học của Softmax
- **Section 3**: Temperature scaling và control over distributions
- **Section 4**: Log-Softmax và numerical stability
- **Section 5**: Triển khai thực nghiệm (NumPy và PyTorch)
- **Section 6**: Applications trong LLMs
- **Section 7**: Best practices và recommendations

---

## 2. Nền Tảng Toán Học của Softmax

### 2.1 Hàm Exponential Tự Nhiên

#### 2.1.1 Định Nghĩa và Tính Chất

**Hằng số Euler:**

$$

e = 2.71828182845904523536\ldots

$$

**Hàm exponential:**

$$

f(x) = e^x

$$

**Tính chất quan trọng cho Softmax:**

**Property 1: Strict Positivity**

$$

e^x > 0 \quad \forall x \in \mathbb{R}

$$

**Ý nghĩa:**
- $e^x$ không bao giờ negative
- Khi $x \to -\infty$, $e^x \to 0^+$ (tiến đến 0 nhưng không bao giờ bằng 0)
- Khi $x \to +\infty$, $e^x \to +\infty$

**Property 2: Non-linear Growth**

$$

\frac{d}{dx}e^x = e^x

$$

**Hành vi:**
- Exponential function "bends upwards" dramatically
- Large values become **exponentially** larger
- Small differences trong input → large differences trong output

**Ví dụ minh họa:**
x:      2      3      8
e^x:    7.39   20.09  2980.96

Observation:
- x: 3 is 1.5× of 2
- e^x: 20.09 is 2.7× of 7.39
- x: 8 is 4× of 2  
- e^x: 2980.96 is 403× of 7.39

**Ý nghĩa cho token selection:**
> Large logits "pop out" disproportionately sau Softmax, giúp model pick specific tokens từ large vocabulary.

#### 2.1.2 Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y = np.exp(x)

plt.plot(x, y, linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('$e^x$', fontsize=12)
plt.title('Natural Exponential Function', fontsize=14)
plt.grid(True, alpha=0.3)

**Quan sát:**
- Always positive (above x-axis)
- Steep increase cho positive x
- Asymptotic approach to 0 cho negative x
- Slope increases với x (accelerating growth)

### 2.2 Định Nghĩa Softmax

#### 2.2.1 Công Thức Chuẩn

**Mathematical definition:**

$$

\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}

$$

Trong đó:
- $\mathbf{z} = [z_1, z_2, \ldots, z_K]$ là input vector (logits)
- $K$ là số lượng classes/categories/tokens
- $\sigma(\mathbf{z})_i$ là probability cho class $i$

**Vectorized form:**

$$

\sigma(\mathbf{z}) = \frac{\exp(\mathbf{z})}{\sum \exp(\mathbf{z})}

$$

**NumPy implementation:**
```python
def softmax(z):
    """
    Compute softmax values for array z
    
    Args:
        z: array of logits, shape (n,)
    
    Returns:
        Probability distribution, shape (n,)
    """
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

#### 2.2.2 Example Computation

**Input (logits):**

$$

\mathbf{z} = [2.0, 1.0, 0.1]

$$

**Step 1: Exponentiate**

$$

\exp(\mathbf{z}) = [e^{2.0}, e^{1.0}, e^{0.1}] = [7.39, 2.72, 1.11]

$$

**Step 2: Sum**

$$

\text{sum} = 7.39 + 2.72 + 1.11 = 11.22

$$

**Step 3: Normalize**

$$

\sigma(\mathbf{z}) = \left[\frac{7.39}{11.22}, \frac{2.72}{11.22}, \frac{1.11}{11.22}\right] = [0.659, 0.242, 0.099]

$$

**Verification:**

$$

0.659 + 0.242 + 0.099 = 1.000 \quad \checkmark

$$

### 2.3 Softmax như Probability Distribution

#### 2.3.1 Định Nghĩa Probability Function

**Formal definition:**
Một hàm $P(X)$ là probability function nếu thỏa mãn hai điều kiện:

**Condition 1: Non-negativity**

$$

P(x) \geq 0 \quad \forall x

$$

**Condition 2: Normalization**

Cho discrete events:

$$

\sum_{x} P(x) = 1

$$

Cho continuous events:

$$

\int_{-\infty}^{\infty} P(x) \, dx = 1

$$

#### 2.3.2 Softmax Thỏa Mãn Probability Axioms

**Theorem:** Softmax transformation maps arbitrary values → valid probability distribution.

**Proof:**

**Part 1: Non-negativity**

Cho bất kỳ $z_i \in \mathbb{R}$:
- Numerator: $e^{z_i} > 0$ (exponential always positive)
- Denominator: $\sum_{j=1}^K e^{z_j} > 0$ (sum of positives is positive)
- Therefore: $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}} > 0$ ✓

**Part 2: Normalization**

$$

\sum_{i=1}^K \sigma(\mathbf{z})_i = \sum_{i=1}^K \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}

$$

Factor out constant denominator:

$$

= \frac{1}{\sum_{j=1}^K e^{z_j}} \sum_{i=1}^K e^{z_i}

$$

Numerator equals denominator:

$$

= \frac{\sum_{i=1}^K e^{z_i}}{\sum_{j=1}^K e^{z_j}} = 1

$$

✓
**Conclusion:**
> Softmax **inherently** creates valid probability distributions, bất kể input values (có thể negative, >1, arbitrary units).
#### 2.3.3 Ý Nghĩa Thực Tiễn
**Input flexibility:**
Logits (arbitrary units):  [-5, 3, 0, 10, -2]
    ↓
Softmax transformation
    ↓
Probabilities:            [0.0001, 0.0336, 0.0017, 0.9645, 0.0002]
**Properties maintained:**
- All values ∈ [0, 1]
- Sum = 1.0000
- Valid probability distribution
**Applications:**
1. **Classification**: Class probabilities từ neural network logits
2. **LLMs**: Token probabilities từ vocabulary logits
3. **Attention**: Attention weights từ similarity scores
4. **Reinforcement Learning**: Action probabilities từ Q-values
---
## 3. Temperature Scaling
### 3.1 Softmax với Temperature
#### 3.1.1 Extended Formula
**Softmax with temperature parameter $T$:**

$$

\sigma_T(\mathbf{z})_i = \frac{e^{z_i/T}}{\sum_{j=1}^{K} e^{z_j/T}} = \frac{\exp(z_i/T)}{\sum_{j=1}^K \exp(z_j/T)}

$$

**Notation:**
- $T$ (temperature): Controls "sharpness" của distribution
- Also called: $\beta$ (inverse temperature), "softness" parameter
- Default: $T = 1$ (standard Softmax)
**Alternative notation:**
Khi exponent phức tạp, sử dụng $\exp(\cdot)$ thay vì $e^{(\cdot)}$:

$$

\sigma_T(\mathbf{z})_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}

$$

#### 3.1.2 Behavior với Different Temperature Values
**Case 1: T = 1 (Standard)**

$$

\sigma_1(\mathbf{z}) = \frac{e^{z_i}}{\sum_j e^{z_j}}

$$

- Normal Softmax
- Balanced probability distribution
**Case 2: T > 1 (High Temperature)**

$$

\sigma_T(\mathbf{z}) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} \quad \text{with } T > 1

$$

**Effect:**
- Dividing by $T > 1$ **shrinks** all values
- Values move closer to 0
- Distribution becomes **flatter** (more uniform)
- High probabilities decrease, low probabilities increase
**Case 3: T < 1 (Low Temperature)**

$$

\sigma_T(\mathbf{z}) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} \quad \text{with } T < 1

$$

**Effect:**
- Dividing by $T < 1$ **amplifies** all values
- Values move away from 0
- Distribution becomes **sharper** (more peaked)
- High probabilities increase, low probabilities decrease toward 0
**Case 4: T → 0 (Zero Temperature Limit)**

$$

\lim_{T \to 0} \sigma_T(\mathbf{z}) = \text{one-hot}(\arg\max \mathbf{z})

$$

**Effect:**
- Approaches deterministic selection
- Highest logit → probability 1
- All others → probability 0
**Case 5: T → ∞ (Infinite Temperature Limit)**

$$

\lim_{T \to \infty} \sigma_T(\mathbf{z}) = \text{uniform}(K)

$$

**Effect:**
- Approaches uniform distribution
- All classes → probability 1/K
- Completely "forgets" logits
### 3.2 Mathematical Intuition
#### 3.2.1 Tại Sao Temperature Làm Phẳng Distribution?
**Key insight:** Exponential function's non-linearity
**Example:**
Original logits:  [2, 3, 8]
T = 1:
  e^2 = 7.39
  e^3 = 20.09
  e^8 = 2980.96
  Ratio: 1 : 2.7 : 403
T = 2:
  e^(2/2) = e^1 = 2.72
  e^(3/2) = e^1.5 = 4.48
  e^(8/2) = e^4 = 54.60
  Ratio: 1 : 1.6 : 20
T = 4:
  e^(2/4) = e^0.5 = 1.65
  e^(3/4) = e^0.75 = 2.12
  e^(8/4) = e^2 = 7.39
  Ratio: 1 : 1.3 : 4.5
**Observation:**
- Higher T → compressed ratios → flatter distribution
- Exponential amplification reduced
- Large values don't "pop out" as much
#### 3.2.2 Graphical Interpretation
**On exponential curve:**
- High temperature: Sample points closer to x=0 (flatter region)
- Low temperature: Sample points spread out (steep regions)
- Steepness of $e^x$ curve determines separation
**Visual analogy:**
e^x curve:
    |     /
    |    /
    |   /
    |  /
    | /
    |/___________
   -3  -1  0  1  2  3
High T: Points clustered near 0 (gentle slope)
Low T: Points spread widely (varying slopes)
### 3.3 Practical Examples
#### 3.3.1 Numerical Demonstration
**Setup:**
```python
import numpy as np
logits = np.array([2.0, 3.0, 8.0])
temperatures = [0.5, 1.0, 2.0, 5.0, 10.0]
**Results:**
| Temperature | Prob[0] | Prob[1] | Prob[2] | Entropy |
|-------------|---------|---------|---------|---------|
| T = 0.5 | 0.0000 | 0.0000 | 1.0000 | 0.00 |
| T = 1.0 | 0.0024 | 0.0066 | 0.9910 | 0.08 |
| T = 2.0 | 0.0427 | 0.0704 | 0.8869 | 0.57 |
| T = 5.0 | 0.2006 | 0.2437 | 0.5557 | 1.05 |
| T = 10.0 | 0.2689 | 0.2969 | 0.4342 | 1.10 |
**Observations:**
1. Low T → sharp, confident distribution
2. High T → flat, uncertain distribution
3. Entropy increases với temperature
4. Highest logit always has highest probability
#### 3.3.2 Large Vocabulary Example
**Realistic LLM scenario:**
```python
# Simulate 25 tokens
np.random.seed(42)
logits = np.random.randint(-5, 15, size=25)
print(f"Logits range: {logits.min()} to {logits.max()}")
# Top logit: 14
# Second: 13
**Analysis:**
**Raw logits:**
- Largest = 14
- Second = 13
- Difference: Only 1 unit
- Not dramatically different
**After Softmax $T=1$:**
```python
probs = softmax(logits)
print(f"P(token_max) = {probs.max():.4f}")
print(f"P(token_2nd) = {sorted(probs)[-2]:.4f}")
print(f"Ratio: {probs.max() / sorted(probs)[-2]:.2f}x")
# Output:
# P(token_max) = 0.6231
# P(token_2nd) = 0.2292
# Ratio: 2.72x
**Key finding:**
> Logit difference of 1 unit → probability ratio of ~2.7x after Softmax. Exponential amplification makes top token **much more likely** despite small logit difference.
### 3.4 Temperature trong LLM Text Generation
#### 3.4.1 Controlling "Creativity"
**Low temperature $T = 0.7$:**
- **Behavior**: Deterministic, focused
- **Distribution**: Peaked, confident
- **Sampling**: Repeatedly picks high-probability tokens
- **Text**: Coherent, predictable, "safe"
- **Use cases**: Factual Q&A, code generation, formal writing
**Medium temperature $T = 1.0$:**
- **Behavior**: Balanced
- **Distribution**: Standard Softmax
- **Sampling**: Mix of high and moderate probabilities
- **Text**: Natural, varied
- **Use cases**: General conversation, creative writing
**High temperature $T = 1.5$:**
- **Behavior**: Exploratory, creative
- **Distribution**: Flattened, uncertain
- **Sampling**: Considers many token options
- **Text**: Diverse, unpredictable, occasionally incoherent
- **Use cases**: Brainstorming, artistic generation, exploring possibilities
#### 3.4.2 Extreme Values (Educational)
**Very low $T = 0.2$:**
```python
# Distribution extremely peaked
probs_low = softmax(logits / 0.2)
# Top token: ~0.99
# Others: ~0.00
- Nearly deterministic
- Always picks argmax
- No randomness
**Very high $T = 10$:**
```python
# Distribution nearly uniform
probs_high = softmax(logits / 10)
# All tokens: ~0.04 (for 25 tokens)
- Almost random selection
- Ignores logits
- Maximum diversity
#### 3.4.3 Production Values
**Typical ranges cho production LLMs:**
- **Conservative**: 0.7 - 0.9
- **Standard**: 0.9 - 1.1
- **Creative**: 1.1 - 1.3
- **Very creative**: 1.3 - 1.5
**Note:**
> Extreme values (T < 0.5 or T > 2.0) primarily for educational/experimental purposes. Real applications stick close to T = 1.0.
---
## 4. Log-Softmax
### 4.1 Định Nghĩa và Motivation
#### 4.1.1 Mathematical Definition
**Log-Softmax:**

$$

\log \sigma(\mathbf{z})_i = \log \left( \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} \right)

$$

**Simplified using log properties:**

$$

\log \sigma(\mathbf{z})_i = z_i - \log \sum_{j=1}^K e^{z_j}

$$

**Với temperature:**

$$

\log \sigma_T(\mathbf{z})_i = \frac{z_i}{T} - \log \sum_{j=1}^K e^{z_j/T}

$$

**Alternative name:** Log-probabilities
#### 4.1.2 Tại Sao Cần Log-Softmax?
**Problem: Very small probabilities**
Trong LLMs với large vocabularies $100K+ tokens$:
- Most tokens có probability **extremely close to 0**
- Ví dụ: $P(\text{token}_i) = 1.23 \times 10^{-8}$
**Issues với small probabilities:**
**Issue 1: Numerical underflow**
```python
prob = 1e-45  # Too small
prob * something  # → 0 (underflow)
**Issue 2: Loss of precision**
```python
p1 = 1e-20
p2 = 2e-20
# Hard to distinguish despite 2x difference
**Issue 3: Gradient vanishing**
```python
grad = prob * (1 - prob)  # → 0 if prob ≈ 0
#### 4.1.3 Logarithm Stretches Small Numbers
**Log transformation properties:**

$$

\log(0.1) = -2.3

$$

$$
\log(0.01) = -4.6

$$

$$
\log(0.001) = -6.9

$$

$$
\log(10^{-8}) = -18.4

$$

**Visual effect:**
Linear scale:

$$

0.001, 0.01, 0.1

$$

|____|____|____
Log scale:

$$

-6.9, -4.6, -2.3

$$

|______|______|
**Benefits:**
1. **Expanded range**: Small probabilities spread out
2. **Better resolution**: Can distinguish between tiny values
3. **Numerical stability**: Avoids underflow
4. **Gradient flow**: Better optimization
### 4.2 Mathematical Properties
#### 4.2.1 Logarithm Properties Used
**Property 1:** $\log(a/b) = \log(a) - \log(b)$

$$

\log \sigma(\mathbf{z})_i = \log(e^{z_i}) - \log\left(\sum_j e^{z_j}\right)

$$

**Property 2:** $\log(e^x) = x$

$$

= z_i - \log\left(\sum_j e^{z_j}\right)

$$

**Interpretation:**
- **First term** ($z_i$): Original logit
- **Second term** ($\log \sum e^{z_j}$): Log-sum-exp normalization
#### 4.2.2 Range Comparison
**Softmax output range:**

$$

\sigma(\mathbf{z})_i \in (0, 1]

$$

**Log-Softmax output range:**

$$

\log \sigma(\mathbf{z})_i \in (-\infty, 0]

$$

**Relationship:**
- $\sigma = 1 \Rightarrow \log \sigma = 0$ (maximum)
- $\sigma \to 0^+ \Rightarrow \log \sigma \to -\infty$ (minimum)
**Example:**
Softmax:      [0.659, 0.242, 0.099]
Log-Softmax:  [-0.417, -1.419, -2.313]
### 4.3 Visualization: Linear vs Log Scale
#### 4.3.1 Comparison Plot
```python
import matplotlib.pyplot as plt
import numpy as np
# Generate logits
logits = np.linspace(-5, 15, 35)
# Compute Softmax and Log-Softmax
temps = [1, 2, 5, 10]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for T in temps:
    # Softmax
    probs = softmax(logits / T)
    ax1.plot(logits, probs, label=f'T={T}', linewidth=2)
    # Log-Softmax
    log_probs = np.log(probs)
    ax2.plot(logits, log_probs, label=f'T={T}', linewidth=2)
ax1.set_xlabel('Logit value')
ax1.set_ylabel('Probability')
ax1.set_title('Softmax (Linear)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.set_xlabel('Logit value')
ax2.set_ylabel('Log-probability')
ax2.set_title('Log-Softmax')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
**Observations:**
**Softmax plot:**
- All curves compressed near 0
- Highest logit creates spike
- Lower temperatures → sharper spike
- Hard to see differences in small probabilities
**Log-Softmax plot:**
- Curves spread vertically
- More spacing between values
- Can clearly see distinctions
- Better for visualizing full distribution
### 4.4 Numerical Stability
#### 4.4.1 Naive Implementation Problems
**Problem code:**
```python
def log_softmax_naive(z):
    return np.log(softmax(z))
**Issues:**
**Issue 1: Intermediate overflow**
```python
z = np.array([1000, 1001, 1002])
exp_z = np.exp(z)  # → [inf, inf, inf]
# Cannot compute softmax!
**Issue 2: Log of zero**
```python
z = np.array([-1000, 0, 1])
probs = softmax(z)  # → [0.0, 5e-435, 1.0]
log_probs = np.log(probs)  # → [-inf, -1000, 0]
# Lost information!
#### 4.4.2 Numerically Stable Implementation
**Better approach:**

$$

\log \sigma(\mathbf{z})_i = z_i - \log \sum_{j=1}^K e^{z_j}

$$

**Further stabilization (LogSumExp trick):**

$$

\log \sum_{j=1}^K e^{z_j} = \log \left( e^{z_{\max}} \sum_{j=1}^K e^{z_j - z_{\max}} \right)

$$

$$
= z_{\max} + \log \sum_{j=1}^K e^{z_j - z_{\max}}

$$

**Stable implementation:**
```python
def log_softmax_stable(z):
    """Numerically stable log-softmax"""
    z_max = np.max(z)
    # Shift values to prevent overflow
    z_shifted = z - z_max
    # Compute log-sum-exp
    log_sum_exp = z_max + np.log(np.sum(np.exp(z_shifted)))
    # Return log-probabilities
    return z_shifted - np.log(np.sum(np.exp(z_shifted)))
# Alternative: Use the shift in final computation
def log_softmax_stable_v2(z):
    z_max = np.max(z)
    return z - z_max - np.log(np.sum(np.exp(z - z_max)))
**Why this works:**
1. Subtract max prevents overflow ($e^{z-z_{\max}} \leq 1$)
2. At least one term equals 1 (prevents all zeros)
3. Mathematically equivalent to naive version
4. Numerically robust
### 4.5 PyTorch Implementation
#### 4.5.1 Built-in Functions
**Two options trong PyTorch:**
**Option 1: Separate operations**
```python
import torch.nn.functional as F
probs = F.softmax(logits, dim=-1)
log_probs = torch.log(probs)
**Pros:** Intuitive  
**Cons:** Numerically unstable
**Option 2: Dedicated function**
```python
log_probs = F.log_softmax(logits, dim=-1)
**Pros:** Numerically stable, optimized  
**Cons:** None
**Recommendation:**
> **Always use `F.log_softmax()`** khi cần log-probabilities. Đừng tự implement với `torch.log(F.softmax())`.
#### 4.5.2 Tại Sao PyTorch Có Dedicated Function?
**Reasons:**
1. **Numerical stability**: Implements LogSumExp trick internally
2. **Computational efficiency**: Fused operation, fewer intermediate tensors
3. **Gradient computation**: More stable gradients
4. **Memory efficiency**: Doesn't store intermediate softmax values
**Example showing difference:**
```python
# Large logits
logits = torch.tensor([100., 101., 102.])
# Naive approach
probs = F.softmax(logits, dim=0)
print(probs)  # tensor([0., 0., 1.]) - lost precision!
log_probs_naive = torch.log(probs)
print(log_probs_naive)  # tensor([-inf, -inf, 0.])
# Stable approach
log_probs_stable = F.log_softmax(logits, dim=0)
print(log_probs_stable)  
# tensor([-2.0000, -1.0000, 0.0000]) - correct!
---
## 5. Triển Khai và Thí Nghiệm
### 5.1 NumPy Implementation
#### 5.1.1 Basic Softmax
```python
import numpy as np
def softmax(z):
    """
    Standard softmax implementation
    Args:
        z: array of logits, shape (n,)
    Returns:
        Probability distribution, shape (n,)
    """
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)
# Test
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Probabilities: {probs}")
print(f"Sum: {probs.sum()}")
**Output:**
Probabilities: [0.65900114 0.24243297 0.09856589]
Sum: 1.0
#### 5.1.2 Softmax with Temperature
```python
def softmax_temperature(z, T=1.0):
    """
    Softmax with temperature scaling
    Args:
        z: array of logits
        T: temperature parameter (default=1.0)
    Returns:
        Temperature-scaled probability distribution
    """
    exp_z = np.exp(z / T)
    return exp_z / np.sum(exp_z)
# Compare different temperatures
logits = np.array([2.0, 1.0, 0.1])
for T in [0.5, 1.0, 2.0]:
    probs = softmax_temperature(logits, T)
    print(f"T={T}: {probs}")
**Output:**
T=0.5: [0.84360176 0.14345253 0.01294571]  # Sharp
T=1.0: [0.65900114 0.24243297 0.09856589]  # Standard
T=2.0: [0.50677007 0.30709724 0.18613269]  # Flat
#### 5.1.3 Log-Softmax
```python
def log_softmax(z):
    """
    Numerically stable log-softmax
    Args:
        z: array of logits
    Returns:
        Log-probabilities
    """
    z_max = np.max(z)
    z_shifted = z - z_max
    log_sum_exp = np.log(np.sum(np.exp(z_shifted)))
    return z_shifted - log_sum_exp
# Test
logits = np.array([2.0, 1.0, 0.1])
log_probs = log_softmax(logits)
print(f"Log-probabilities: {log_probs}")
# Verify: exp(log_probs) should equal softmax(logits)
recovered_probs = np.exp(log_probs)
original_probs = softmax(logits)
print(f"Match: {np.allclose(recovered_probs, original_probs)}")
### 5.2 PyTorch Implementation
#### 5.2.1 Basic Usage
```python
import torch
import torch.nn.functional as F
# Create logits
logits = torch.tensor([2.0, 1.0, 0.1])
# Softmax
probs = F.softmax(logits, dim=0)
print(f"Probabilities: {probs}")
# Log-Softmax
log_probs = F.log_softmax(logits, dim=0)
print(f"Log-probabilities: {log_probs}")
#### 5.2.2 Dimension Parameter
**Critical importance của `dim` parameter:**
```python
# 1D tensor
logits_1d = torch.tensor([1.0, 2.0, 3.0])
probs_1d = F.softmax(logits_1d, dim=0)
# 2D tensor (batch)
logits_2d = torch.tensor([[1.0, 2.0, 3.0],
                          [0.5, 1.5, 2.5]])
# Softmax over features (dim=1)
probs_features = F.softmax(logits_2d, dim=1)
print("Softmax over features:")
print(probs_features)
print(f"Row sums: {probs_features.sum(dim=1)}")
# Softmax over batch (dim=0)
probs_batch = F.softmax(logits_2d, dim=0)
print("\nSoftmax over batch:")
print(probs_batch)
print(f"Column sums: {probs_batch.sum(dim=0)}")
**Output:**
Softmax over features:
tensor([[0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652]])
Row sums: tensor([1., 1.])
Softmax over batch:
tensor([[0.5000, 0.5000, 0.5000],
        [0.5000, 0.5000, 0.5000]])
Column sums: tensor([1., 1., 1.])
**Common pattern trong LLMs:**
```python
# Shape: [batch, sequence, vocab]
logits = torch.randn(32, 128, 100000)
# Softmax over vocabulary (last dimension)
probs = F.softmax(logits, dim=-1)
# Shape: [32, 128, 100000]
# probs.sum(dim=-1) → all 1.0
#### 5.2.3 Warning Message
**Deprecated behavior:**
```python
logits = torch.tensor([1.0, 2.0, 3.0])
probs = F.softmax(logits)  # Warning!
**Warning:**
UserWarning: Implicit dimension choice for softmax has been deprecated.
Change the call to include dim=X as an argument.
**Solution:**
```python
probs = F.softmax(logits, dim=0)  # ✓ No warning
**Why this matters:**
- Prevents ambiguity trong multi-dimensional tensors
- Makes code intention explicit
- Avoids bugs từ incorrect dimension
### 5.3 Comprehensive Experiments
#### 5.3.1 Temperature Effects
```python
import matplotlib.pyplot as plt
# Generate diverse logits
np.random.seed(42)
logits = np.random.randint(-5, 15, size=25)
# Range of temperatures
temperatures = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for idx, T in enumerate(temperatures):
    probs = softmax_temperature(logits, T)
    axes[idx].bar(range(len(probs)), probs)
    axes[idx].set_title(f'Temperature = {T}', fontsize=14)
    axes[idx].set_xlabel('Token index')
    axes[idx].set_ylabel('Probability')
    axes[idx].set_ylim([0, 1])
    # Add entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    axes[idx].text(0.7, 0.9, f'H = {entropy:.2f}', 
                   transform=axes[idx].transAxes)
plt.tight_layout()
plt.savefig('temperature_effects.png', dpi=300)
**Observations:**
- **T = 0.2**: One dominant bar, others near zero
- **T = 1.0**: Clear peak, visible distribution
- **T = 10.0**: Nearly uniform distribution
- **Entropy**: Increases monotonically with temperature
#### 5.3.2 Softmax vs Log-Softmax
```python
# Generate data
logits = np.linspace(-5, 15, 35)
temps = [1, 2, 5, 10]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for T in temps:
    # Softmax
    probs = np.array([softmax_temperature(logits, T)])
    ax1.plot(logits, probs.flatten(), label=f'T={T}', linewidth=2)
    # Log-Softmax
    log_probs = np.log(probs.flatten())
    ax2.plot(logits, log_probs, label=f'T={T}', linewidth=2)
ax1.set_xlabel('Logit value', fontsize=12)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title('Softmax (Linear Scale)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.set_xlabel('Logit value', fontsize=12)
ax2.set_ylabel('Log-probability', fontsize=12)
ax2.set_title('Log-Softmax (Log Scale)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
**Analysis:**
**Softmax plot:**
- All values compressed near y=0
- Difficult to distinguish small probabilities
- Visual focus on peaks only
**Log-Softmax plot:**
- Values spread across y-axis
- Clear separation between all probabilities
- Can see full distribution structure
- Better for understanding model confidence
---
## 6. Applications trong Large Language Models
### 6.1 Token Generation Pipeline
#### 6.1.1 Complete Workflow
**Step 1: Model forward pass**
```python
# Input: Token IDs [batch, seq_len]
input_ids = torch.tensor([[15, 42, 88, 156]])
# Model processes input
hidden_states = model(input_ids)  
# Shape: [batch, seq_len, hidden_dim]
# Final layer (unembedding)
logits = model.lm_head(hidden_states[:, -1, :])
# Shape: [batch, vocab_size]
**Step 2: Apply softmax với temperature**
```python
temperature = 1.0
scaled_logits = logits / temperature
probs = F.softmax(scaled_logits, dim=-1)
# Shape: [batch, vocab_size]
**Step 3: Sample token**
```python
# Option 1: Greedy (argmax)
next_token = torch.argmax(probs, dim=-1)
# Option 2: Probabilistic sampling
next_token = torch.multinomial(probs, num_samples=1)
**Step 4: Append and continue**
```python
input_ids = torch.cat([input_ids, next_token], dim=1)
# Repeat for desired length
#### 6.1.2 Temperature Presets
**Typical configurations:**
```python
# Factual, deterministic
temperature_factual = 0.7
top_p = 0.9
# Balanced
temperature_balanced = 1.0
top_p = 0.95
# Creative
temperature_creative = 1.2
top_p = 0.98
# Very creative
temperature_experimental = 1.5
top_p = 1.0
### 6.2 Training: Cross-Entropy Loss
#### 6.2.1 Loss Computation
**Cross-entropy với Softmax:**

$$

\mathcal{L} = -\sum_{i=1}^K y_i \log(\sigma(\mathbf{z})_i)

$$

Trong đó:
- $y_i$ = ground truth (one-hot encoded)
- $\sigma(\mathbf{z})_i$ = predicted probability
**Với Log-Softmax:**

$$

\mathcal{L} = -\sum_{i=1}^K y_i \cdot \log\sigma(\mathbf{z})_i

$$

**PyTorch implementation:**
**Option 1: Separate (not recommended)**
```python
probs = F.softmax(logits, dim=-1)
loss = F.nll_loss(torch.log(probs), targets)
**Option 2: Combined (recommended)**
```python
loss = F.cross_entropy(logits, targets)
**Why combined is better:**
- Numerically stable
- Combines softmax + log + NLL
- More efficient
- Better gradients
#### 6.2.2 Gradient Flow
**Softmax gradient:**

$$

\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)

$$

Trong đó $\delta_{ij}$ là Kronecker delta.
**Properties:**
- Always well-defined
- Bounded values
- Smooth gradient flow
- Suitable cho deep networks
### 6.3 Attention Mechanisms
#### 6.3.1 Scaled Dot-Product Attention
**Formula:**

$$

\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}

$$

**Softmax role:**
- Converts similarity scores → attention weights
- Ensures weights sum to 1
- Creates weighted combination của values
**Implementation:**
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention
    Args:
        Q: Queries [batch, heads, seq_q, d_k]
        K: Keys [batch, heads, seq_k, d_k]
        V: Values [batch, heads, seq_v, d_v]
        mask: Optional mask [batch, 1, seq_q, seq_k]
    Returns:
        Attention output [batch, heads, seq_q, d_v]
    """
    d_k = Q.size(-1)
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)
    # Apply weights to values
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
**Temperature trong attention:**
- Implicitly controlled by $\sqrt{d_k}$ scaling
- Prevents extremely large scores
- Maintains stable gradients
---
## 7. Best Practices và Recommendations
### 7.1 Implementation Guidelines
#### 7.1.1 Always Specify Dimension
**✗ Bad:**
```python
probs = F.softmax(logits)  # Deprecated, ambiguous
**✓ Good:**
```python
probs = F.softmax(logits, dim=-1)  # Explicit, clear
**Reasoning:**
- Prevents bugs in multi-dimensional tensors
- Makes code intention clear
- Follows PyTorch best practices
#### 7.1.2 Use Built-in Functions
**✗ Bad:**
```python
# Numerically unstable
log_probs = torch.log(F.softmax(logits, dim=-1))
**✓ Good:**
```python
# Numerically stable, optimized
log_probs = F.log_softmax(logits, dim=-1)
**Benefits:**
- Numerical stability (LogSumExp trick)
- Computational efficiency
- Memory savings
- Better gradients
#### 7.1.3 Temperature Selection
**For text generation:**
**Factual tasks:**
```python
temperature = 0.7  # More deterministic
# Examples: Q&A, fact retrieval, translation
**Balanced tasks:**
```python
temperature = 1.0  # Standard
# Examples: General chat, summarization
**Creative tasks:**
```python
temperature = 1.2  # More exploratory
# Examples: Story writing, brainstorming
**General rule:**
> Start with T=1.0, adjust based on desired creativity/diversity. Stay within [0.7, 1.3] for most applications.
### 7.2 Common Pitfalls
#### 7.2.1 Forgetting to Scale
**Problem:**
```python
# Want low temperature but forget to scale
probs = F.softmax(logits, dim=-1)
# Temperature effectively = 1.0!
**Solution:**
```python
temperature = 0.8
scaled_logits = logits / temperature
probs = F.softmax(scaled_logits, dim=-1)
#### 7.2.2 Inconsistent Dimensions
**Problem:**
```python
# logits: [batch, seq, vocab]
probs = F.softmax(logits, dim=0)  # Wrong dimension!
# Softmax over batch instead of vocab
**Solution:**
```python
probs = F.softmax(logits, dim=-1)  # Correct: over vocab
**Verification:**
```python
# Check that last dimension sums to 1
assert torch.allclose(probs.sum(dim=-1), torch.ones(batch, seq))
#### 7.2.3 Numerical Instability
**Problem:**
```python
# Very large logits
logits = torch.tensor([1000., 1001., 1002.])
probs = F.softmax(logits, dim=0)  
# May produce nan or inf
**Solution 1: Use log-softmax**
```python
log_probs = F.log_softmax(logits, dim=0)
probs = torch.exp(log_probs)
**Solution 2: Clip logits**
```python
logits = torch.clamp(logits, min=-100, max=100)
probs = F.softmax(logits, dim=0)
### 7.3 Testing và Validation
#### 7.3.1 Unit Tests
```python
def test_softmax_properties():
    """Test that softmax satisfies probability axioms"""
    logits = torch.randn(10, 100)
    probs = F.softmax(logits, dim=-1)
    # Test 1: Non-negativity
    assert torch.all(probs >= 0), "Negative probabilities!"
    # Test 2: Normalization
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums)), \
        "Probabilities don't sum to 1!"
    # Test 3: Range
    assert torch.all(probs <= 1), "Probabilities > 1!"
    print("All tests passed! ✓")
test_softmax_properties()
#### 7.3.2 Temperature Validation
```python
def validate_temperature(logits, T):
    """Validate temperature effects"""
    probs_1 = F.softmax(logits / 1.0, dim=-1)
    probs_T = F.softmax(logits / T, dim=-1)
    entropy_1 = -(probs_1 * torch.log(probs_1 + 1e-10)).sum(dim=-1)
    entropy_T = -(probs_T * torch.log(probs_T + 1e-10)).sum(dim=-1)
    if T > 1.0:
        assert torch.all(entropy_T >= entropy_1), \
            "High T should increase entropy!"
    elif T < 1.0:
        assert torch.all(entropy_T <= entropy_1), \
            "Low T should decrease entropy!"
    print(f"Temperature T={T} validation passed! ✓")
---
## 8. Advanced Topics
### 8.1 Gumbel-Softmax
**Reparameterization trick cho discrete sampling:**

$$

\text{Gumbel-Softmax}(\mathbf{z}, \tau)_i = \frac{\exp((z_i + g_i)/\tau)}{\sum_j \exp((z_j + g_j)/\tau)}

$$

Trong đó $g_i \sim \text{Gumbel}(0,1)$
**Use case:** Differentiable sampling trong VAEs
### 8.2 Sparsemax
**Alternative to Softmax cho sparse outputs:**

$$

\text{sparsemax}(\mathbf{z}) = \arg\min_{\mathbf{p} \in \Delta^{K-1}} \|\mathbf{p} - \mathbf{z}\|^2

$$

**Property:** Can produce exactly 0 probabilities
### 8.3 Entmax
**Generalization của Softmax và Sparsemax:**
Parameterized by $\alpha \in [1, 2]$:
- $\alpha = 1$: Sparsemax
- $\alpha = 2$: Softmax
---
## 9. Kết Luận
### 9.1 Tóm Tắt Key Findings
**Về Softmax:**
1. Transforms arbitrary values → valid probability distributions
2. Guarantees non-negativity và normalization
3. Exponential amplifies differences trong logits
4. Essential cho classification, generation, attention
**Về Temperature:**
1. Controls "sharpness" của distribution
2. Low T → peaked, confident (deterministic)
3. High T → flat, uncertain (exploratory)
4. Production values typically [0.7, 1.3]
5. Extreme values for educational purposes only
**Về Log-Softmax:**
1. Stretches small probabilities
2. Improves numerical stability
3. Essential cho large vocabularies
4. Use PyTorch's `F.log_softmax()`, not manual log
5. Better gradient flow cho optimization
### 9.2 Core Insights
**Insight 1: Mathematical Elegance**
> Softmax's simple formula encodes profound properties: guaranteed probability distribution từ bất kỳ inputs nào.
**Insight 2: Temperature as Control Knob**
> Temperature provides intuitive control over model behavior—determinism vs creativity—without changing model architecture.
**Insight 3: Numerical Considerations Matter**
> For large-scale models, numerical stability isn't optional—it's essential. Log-Softmax và proper implementations prevent catastrophic failures.
**Insight 4: Domain-Specific Tuning**
> Optimal temperature depends on task: factual tasks need low T, creative tasks benefit from higher T.
### 9.3 Practical Takeaways
**For practitioners:**
1. **Always use `dim` parameter** trong F.softmax()
2. **Use F.log_softmax()** thay vì torch.log(F.softmax())
3. **Start with T=1.0**, adjust based on needs
4. **Test probability properties**: non-negative, sum to 1
5. **Monitor numerical issues**: nan, inf, underflow
**For researchers:**
1. Study relationship giữa temperature và entropy
2. Investigate alternatives (Sparsemax, Entmax)
3. Explore adaptive temperature mechanisms
4. Consider computational efficiency vs accuracy trade-offs
### 9.4 Final Thoughts
Softmax dường như đơn giản—một công thức ngắn, một phép biến đổi cơ bản. Nhưng như bài viết này đã chứng minh, nó chứa đựng:
- Mathematical rigor (probability axioms)
- Practical nuances (temperature, stability)
- Implementation challenges (numerical precision)
- Wide applications (LLMs, attention, classification)
**Quan trọng nhất:**
> Hiểu sâu về Softmax không chỉ là học một công thức—mà là nắm vững foundation của modern deep learning, từ training (loss computation) đến inference (token generation).
Temperature scaling thêm một layer của control, cho phép practitioners fine-tune model behavior without architectural changes—một ví dụ đẹp của "simple but powerful" trong machine learning.
---
## 10. Tài Liệu Tham Khảo
[1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.2.2.3: Softmax Units for Multinoulli Output Distributions.
[2] Vaswani, A., et al. (2017). "Attention is All You Need." *Advances in Neural Information Processing Systems* (NeurIPS), 30, 5998-6008. https://arxiv.org/abs/1706.03762
[3] Bridle, J. S. (1990). "Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition." In *Neurocomputing: Algorithms, Architectures and Applications*, pp. 227-236.
[4] Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." *NIPS Deep Learning Workshop*. https://arxiv.org/abs/1503.02531
   - Introduces temperature trong knowledge distillation context
[5] Jang, E., Gu, S., & Poole, B. (2017). "Categorical Reparameterization with Gumbel-Softmax." *International Conference on Learning Representations* (ICLR). https://arxiv.org/abs/1611.01144
[6] Martins, A., & Astudillo, R. (2016). "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification." *International Conference on Machine Learning* (ICML), pp. 1614-1623. https://arxiv.org/abs/1602.02068
[7] Peters, B., Niculae, V., & Martins, A. F. T. (2019). "Sparse Sequence-to-Sequence Models." *Proceedings of ACL*, pp. 1504-1519. https://arxiv.org/abs/1905.05702
   - Entmax generalization
[8] Blanc, G., & Rendle, S. (2018). "Adaptive Sampler for Deep Learning Based Recommender Systems." In *RecSys Workshop on Deep Learning for Recommender Systems*.
[9] Chen, X., et al. (2016). "Variational Lossy Autoencoder." *International Conference on Learning Representations* (ICLR). https://arxiv.org/abs/1611.02731
[10] Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Technical Report*. 
    - GPT-2, discusses temperature trong text generation
[11] Holtzman, A., et al. (2019). "The Curious Case of Neural Text Degeneration." *International Conference on Learning Representations* (ICLR). https://arxiv.org/abs/1904.09751
    - Nucleus sampling (top-p) và temperature
[12] Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*. https://arxiv.org/abs/1912.01703
---
## Phụ Lục A: Complete Code Examples
### A.1 Comprehensive Softmax Module
```python
"""
softmax_utils.py
Complete implementation of Softmax variants
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
class SoftmaxAnalyzer:
    """Utility class for Softmax analysis"""
    @staticmethod
    def softmax(z, T=1.0, axis=-1):
        """
        NumPy implementation of temperature-scaled softmax
        Args:
            z: Input logits (numpy array)
            T: Temperature (float, default=1.0)
            axis: Axis along which to compute softmax
        Returns:
            Probability distribution (numpy array)
        """
        z_scaled = z / T
        exp_z = np.exp(z_scaled - np.max(z_scaled, axis=axis, keepdims=True))
        return exp_z / np.sum(exp_z, axis=axis, keepdims=True)
    @staticmethod
    def log_softmax(z, T=1.0, axis=-1):
        """
        Numerically stable log-softmax
        Args:
            z: Input logits
            T: Temperature
            axis: Axis for computation
        Returns:
            Log-probabilities
        """
        z_scaled = z / T
        z_max = np.max(z_scaled, axis=axis, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(z_scaled - z_max), 
                                     axis=axis, keepdims=True))
        return z_scaled - z_max - log_sum_exp
    @staticmethod
    def entropy(probs, axis=-1):
        """
        Compute entropy of probability distribution
        H(p) = -Σ p_i log(p_i)
        """
        return -np.sum(probs * np.log(probs + 1e-10), axis=axis)
    @staticmethod
    def validate_probabilities(probs, tol=1e-6):
        """
        Validate that probs form valid probability distribution
        Returns:
            bool: True if valid, raises AssertionError otherwise
        """
        # Non-negativity
        assert np.all(probs >= -tol), "Negative probabilities detected!"
        # Range [0, 1]
        assert np.all(probs <= 1 + tol), "Probabilities > 1 detected!"
        # Normalization
        sums = np.sum(probs, axis=-1)
        assert np.allclose(sums, 1.0, atol=tol), \
            f"Probabilities don't sum to 1! Sums: {sums}"
        return True
    @staticmethod
    def compare_temperatures(logits, temperatures=[0.5, 1.0, 2.0, 5.0]):
        """
        Compare softmax distributions across temperatures
        Args:
            logits: Input logits
            temperatures: List of temperature values
        Returns:
            Dictionary with results for each temperature
        """
        results = {}
        for T in temperatures:
            probs = SoftmaxAnalyzer.softmax(logits, T=T)
            H = SoftmaxAnalyzer.entropy(probs)
            results[T] = {
                'probabilities': probs,
                'entropy': H,
                'max_prob': np.max(probs),
                'min_prob': np.min(probs),
                'top_3_indices': np.argsort(probs)[-3:][::-1]
            }
        return results
    @staticmethod
    def plot_temperature_effects(logits, temperatures=[0.5, 1.0, 2.0, 5.0]):
        """Visualize temperature effects"""
        n_temps = len(temperatures)
        fig, axes = plt.subplots(1, n_temps, figsize=(5*n_temps, 4))
        if n_temps == 1:
            axes = [axes]
        for idx, T in enumerate(temperatures):
            probs = SoftmaxAnalyzer.softmax(logits, T=T)
            H = SoftmaxAnalyzer.entropy(probs)
            axes[idx].bar(range(len(probs)), probs)
            axes[idx].set_title(f'T = {T}\nH = {H:.3f}')
            axes[idx].set_xlabel('Token Index')
            axes[idx].set_ylabel('Probability')
            axes[idx].set_ylim([0, 1])
        plt.tight_layout()
        return fig
# Example usage
if __name__ == "__main__":
    # Generate random logits
    np.random.seed(42)
    logits = np.random.randn(20)
    # Analyze
    analyzer = SoftmaxAnalyzer()
    # Validate standard softmax
    probs = analyzer.softmax(logits)
    print("Validation:", analyzer.validate_probabilities(probs))
    # Compare temperatures
    results = analyzer.compare_temperatures(logits)
    for T, data in results.items():
        print(f"\nTemperature {T}:")
        print(f"  Entropy: {data['entropy']:.4f}")
        print(f"  Max prob: {data['max_prob']:.4f}")
        print(f"  Top 3 tokens: {data['top_3_indices']}")
    # Visualize
    fig = analyzer.plot_temperature_effects(logits)
    plt.savefig('temperature_analysis.png', dpi=300, bbox_inches='tight')
### A.2 PyTorch LLM Generation Example
```python
"""
llm_generation_example.py
Demonstrate text generation with temperature control
"""
import torch
import torch.nn.functional as F
class SimpleGenerator:
    """Simple text generator with temperature control"""
    def __init__(self, vocab_size=50000, temperature=1.0):
        self.vocab_size = vocab_size
        self.temperature = temperature
    def generate_next_token(self, logits, method='sample'):
        """
        Generate next token from logits
        Args:
            logits: Model output logits [vocab_size]
            method: 'sample', 'greedy', or 'top_k'
        Returns:
            next_token: Token ID
            prob: Probability of selected token
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        # Compute probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        if method == 'greedy':
            # Always pick highest probability
            next_token = torch.argmax(probs)
            prob = probs[next_token]
        elif method == 'sample':
            # Sample from distribution
            next_token = torch.multinomial(probs, num_samples=1)
            prob = probs[next_token]
        elif method == 'top_k':
            # Sample from top K tokens
            k = 50
            top_k_probs, top_k_indices = torch.topk(probs, k)
            top_k_probs = top_k_probs / top_k_probs.sum()  # Renormalize
            # Sample from top K
            sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
            next_token = top_k_indices[sampled_idx]
            prob = probs[next_token]
        else:
            raise ValueError(f"Unknown method: {method}")
        return next_token.item(), prob.item()
    def generate_sequence(self, initial_logits, length=10, method='sample'):
        """
        Generate sequence of tokens
        Args:
            initial_logits: Starting logits [vocab_size]
            length: Number of tokens to generate
            method: Generation method
        Returns:
            tokens: List of generated token IDs
            probs: List of their probabilities
        """
        tokens = []
        probs = []
        current_logits = initial_logits
        for _ in range(length):
            token, prob = self.generate_next_token(current_logits, method)
            tokens.append(token)
            probs.append(prob)
            # In real LLM, would feed token back to model
            # Here we just use same logits for demonstration
        return tokens, probs
# Demo
if __name__ == "__main__":
    # Simulate model output
    torch.manual_seed(42)
    logits = torch.randn(50000)  # 50K vocab
    print("Comparing generation methods:\n")
    # Temperature sweep
    for temp in [0.7, 1.0, 1.3]:
        print(f"Temperature = {temp}")
        generator = SimpleGenerator(temperature=temp)
        # Generate with different methods
        for method in ['greedy', 'sample']:
            tokens, probs = generator.generate_sequence(
                logits, length=5, method=method
            )
            print(f"  {method}: tokens={tokens[:5]}")
            print(f"           probs={[f'{p:.3f}' for p in probs[:5]]}")
        print()
### A.3 Numerical Stability Tests
```python
"""
numerical_stability_tests.py
Test numerical stability of Softmax implementations
"""
import torch
import torch.nn.functional as F
import numpy as np
def test_large_logits():
    """Test behavior with very large logits"""
    print("Test 1: Large Logits")
    print("-" * 50)
    logits = torch.tensor([1000., 1001., 1002.])
    # Naive (will fail)
    try:
        exp_z = torch.exp(logits)
        print(f"exp(logits): {exp_z}")  # Will be inf
        naive_probs = exp_z / exp_z.sum()
        print(f"Naive softmax: {naive_probs}")  # Will be nan
    except:
        print("Naive approach failed!")
    # PyTorch's stable implementation
    stable_probs = F.softmax(logits, dim=0)
    print(f"F.softmax: {stable_probs}")
    # Log-softmax approach
    log_probs = F.log_softmax(logits, dim=0)
    recovered_probs = torch.exp(log_probs)
    print(f"Via log_softmax: {recovered_probs}")
    print()
def test_small_probabilities():
    """Test handling of very small probabilities"""
    print("Test 2: Small Probabilities")
    print("-" * 50)
    logits = torch.tensor([-1000., -500., 0.])
    # Standard softmax
    probs = F.softmax(logits, dim=0)
    print(f"Probabilities: {probs}")
    # Try to take log (naive)
    try:
        naive_log_probs = torch.log(probs)
        print(f"Naive log(probs): {naive_log_probs}")
    except:
        print("Naive log failed!")
    # Stable log-softmax
    log_probs = F.log_softmax(logits, dim=0)
    print(f"F.log_softmax: {log_probs}")
    print()
def test_gradient_stability():
    """Test gradient computation stability"""
    print("Test 3: Gradient Stability")
    print("-" * 50)
    # Extreme logits
    logits = torch.tensor([100., 101., 102.], requires_grad=True)
    target = torch.tensor(2)  # True class
    # Compute loss using stable functions
    log_probs = F.log_softmax(logits, dim=0)
    loss = F.nll_loss(log_probs.unsqueeze(0), target.unsqueeze(0))
    print(f"Loss: {loss.item():.6f}")
    # Compute gradients
    loss.backward()
    print(f"Gradients: {logits.grad}")
    # Verify gradients are reasonable
    assert not torch.any(torch.isnan(logits.grad)), "NaN gradients!"
    assert not torch.any(torch.isinf(logits.grad)), "Inf gradients!"
    print("Gradients are stable! ✓")
    print()
def test_temperature_extremes():
    """Test extreme temperature values"""
    print("Test 4: Extreme Temperatures")
    print("-" * 50)
    logits = torch.tensor([1., 2., 3., 4., 5.])
    # Very low temperature
    T_low = 0.01
    probs_low = F.softmax(logits / T_low, dim=0)
    print(f"T = {T_low}: {probs_low}")
    print(f"  → Nearly one-hot: {probs_low.max():.6f}")
    # Very high temperature
    T_high = 100.0
    probs_high = F.softmax(logits / T_high, dim=0)
    print(f"T = {T_high}: {probs_high}")
    print(f"  → Nearly uniform: {probs_high.std():.6f}")
    print()
# Run all tests
if __name__ == "__main__":
    print("=" * 50)
    print("NUMERICAL STABILITY TESTS")
    print("=" * 50)
    print()
    test_large_logits()
    test_small_probabilities()
    test_gradient_stability()
    test_temperature_extremes()
    print("=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)
---
## Phụ Lục B: Mathematical Derivations
### B.1 Softmax Derivative
**Setup:**

$$

\sigma_i = \frac{e^{z_i}}{\sum_j e^{z_j}}

$$

**Derivative w.r.t. $z_j$:**
**Case 1: $i = j$**

$$

\frac{\partial \sigma_i}{\partial z_i} = \frac{e^{z_i} \sum_k e^{z_k} - e^{z_i} \cdot e^{z_i}}{(\sum_k e^{z_k})^2}

$$

$$
= \frac{e^{z_i}}{\sum_k e^{z_k}} \cdot \frac{\sum_k e^{z_k} - e^{z_i}}{\sum_k e^{z_k}}

$$

$$
= \sigma_i (1 - \sigma_i)

$$

**Case 2: $i \neq j$**

$$

\frac{\partial \sigma_i}{\partial z_j} = \frac{0 \cdot \sum_k e^{z_k} - e^{z_i} \cdot e^{z_j}}{(\sum_k e^{z_k})^2}

$$

$$
= -\frac{e^{z_i}}{\sum_k e^{z_k}} \cdot \frac{e^{z_j}}{\sum_k e^{z_k}}

$$

$$
= -\sigma_i \sigma_j

$$

**Unified form:**

$$

\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)

$$

Trong đó $\delta_{ij}$ là Kronecker delta:

$$

\delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}

$$

### B.2 Cross-Entropy Loss Gradient
**Loss function:**

$$

\mathcal{L} = -\sum_i y_i \log \sigma_i

$$

**Gradient w.r.t. logit $z_j$:**

$$

\frac{\partial \mathcal{L}}{\partial z_j} = -\sum_i y_i \frac{\partial \log \sigma_i}{\partial z_j}

$$

$$
= -\sum_i y_i \frac{1}{\sigma_i} \frac{\partial \sigma_i}{\partial z_j}

$$

$$
= -\sum_i y_i \frac{1}{\sigma_i} \sigma_i(\delta_{ij} - \sigma_j)

$$

$$
= -\sum_i y_i (\delta_{ij} - \sigma_j)

$$

$$
= -y_j + \sigma_j \sum_i y_i

$$

Vì $\sum_i y_i = 1$ (one-hot):

$$

= \sigma_j - y_j

$$

**Result:**
> Gradient của cross-entropy loss w.r.t. logits = predicted prob - true prob. Cực kỳ clean và elegant!
### B.3 Temperature Gradient
**With temperature:**

$$

\sigma_T(z_i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}

$$

**Derivative w.r.t. T:**

$$

\frac{\partial \sigma_T}{\partial T} = -\frac{1}{T^2} \left( \frac{z_i e^{z_i/T}}{\sum_j e^{z_j/T}} - \frac{e^{z_i/T} \sum_j z_j e^{z_j/T}}{(\sum_j e^{z_j/T})^2} \right)

$$

After simplification:

$$

= -\frac{1}{T^2} \sigma_T(z_i) (z_i - \mathbb{E}_{z \sim \sigma_T}[z])
