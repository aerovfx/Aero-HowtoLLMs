
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [04 buildGPT](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
## Các phương pháp lấy mẫu token (Token Sampling Methods) trong sinh văn bản. 

## Highlights của bài viết:

### **Cấu trúc chính (11 sections + 2 appendices):**

#### **Phần lý thuyết nền tảng:**
1. **Mathematical foundations** - Probability distributions, decoding problem definition
2. **Chi tiết 4 phương pháp** - Multinomial, Greedy, Top-K, Top-P với công thức đầy đủ
3. **So sánh toàn diện** - Trade-offs, computational costs, hyperparameter sensitivity

#### **Phần implementation:**
4. **Complete PyTorch code** - Production-ready sampler với safeguards
5. **Generation loops** - Full text generation pipeline
6. **Evaluation metrics** - Diversity, quality, repetition scoring

#### **Phần ứng dụng:**
7. **Task-specific recommendations** - Q&A, code, creative writing, chat, translation
8. **Real-world examples** - ChatGPT, GitHub Copilot, AI Dungeon configurations
9. **Configuration matrix** - Temperature + method combinations

#### **Phần critical analysis:**
10. **The Hallucination Problem** - Root causes, risk analysis, mitigation strategies
11. **Best practices** - Decision framework, tuning guide, common mistakes

### **Nội dung đặc biệt:**

✅ **15 academic citations** từ ICLR, NeurIPS, ACL papers  
✅ **Mathematical rigor** - Formal definitions, probability proofs  
✅ **Production code** - Complete implementation với error handling  
✅ **Comprehensive comparisons** - 4 methods analyzed from every angle  
✅ **Real examples** - Actual use cases và configurations  
✅ **Hallucination analysis** - Fundamental problem explored deeply  
✅ **Decision flowcharts** - Practical guidance cho method selection  
✅ **Advanced topics** - Beam search, contrastive decoding, speculative decoding

## Key insights covered:

**Core findings:**
- **No universal best method** - Task-dependent optimization
- **Top-P currently dominant** - Adaptive nucleus wins in practice
- **Trade-offs fundamental** - Diversity vs accuracy inherent tension
- **Hallucination unresolved** - Stochasticity enables creativity but risks errors
- **Hyperparameters critical** - Temperature + sampling method tuned together

**Practical value:**
- Task-specific configurations (factual: greedy, creative: top-p high)
- Production settings (typically T=0.7-1.3, p=0.9-0.95)
- Common mistakes và how to avoid
- Complete evaluation framework

**Research contributions:**
- Systematic comparison framework
- Mathematical formalization
- Trade-off analysis
- Future directions identified

# Phương Pháp Lấy Mẫu Token trong Sinh Văn Bản: Phân Tích So Sánh Greedy, Top-K, Top-P và Multinomial Sampling

**Tác giả:** Phân tích học thuật  
**Ngày:** 14 tháng 2, 2026  
**Lĩnh vực:** Natural Language Processing, Text Generation, Large Language Models

---

## Tóm tắt

Bài viết này trình bày phân tích toàn diện về các phương pháp lấy mẫu token (decoding strategies) trong text generation của Large Language Models (LLMs). Nghiên cứu tập trung vào bốn phương pháp chính: Multinomial Sampling, Greedy Decoding, Top-K Sampling, và Top-P (Nucleus) Sampling. Thông qua phân tích toán học, so sánh thực nghiệm, và đánh giá trên các use cases khác nhau, bài viết làm rõ trade-offs giữa determinism và stochasticity, diversity và quality, creativity và accuracy. Kết quả cho thấy không có "best" method universally—lựa chọn optimal phụ thuộc vào application context, từ factual Q&A (greedy/low temperature) đến creative writing (top-p/high diversity). Nghiên cứu cũng đề cập đến fundamental challenge của LLMs: balancing between hallucination risk và output diversity.

**Từ khóa:** Token Sampling, Decoding Strategies, Greedy Search, Top-K Sampling, Nucleus Sampling, Text Generation, Large Language Models, Stochastic Decoding

---

## 1. Giới Thiệu

### 1.1 Bối Cảnh và Động Lực

#### 1.1.1 Text Generation Pipeline

Trong Large Language Models, text generation là quá trình auto-regressive, trong đó model sinh từng token một dựa trên context đã có [1]:

```
Context: "I prefer oat milk in my ___"
    ↓
Model processes context
    ↓
Outputs: Probability distribution over vocabulary
    ↓
Sampling strategy selects next token
    ↓
Token appended to context
    ↓
Process repeats
```

**Vị trí của sampling trong pipeline:**
> Sau khi model tính toán logits và áp dụng Softmax (với temperature), chúng ta có probability distribution P(token|context). **Decoding strategy** quyết định cách chọn token từ distribution này.

#### 1.1.2 Tầm Quan Trọng của Decoding Strategy

**Impact lên output quality:**
- **Determinism vs Randomness**: Ảnh hưởng repeatability và diversity
- **Creativity vs Accuracy**: Trade-off giữa novel outputs và factual correctness
- **Coherence vs Exploration**: Balance giữa staying on-topic và generating varied content

**Real-world consequences:**
```
Application          | Preferred Strategy | Reasoning
---------------------|-------------------|------------------
Chatbot conversation | Top-P (0.9)       | Natural, varied
Code generation      | Greedy/Low temp   | Correctness critical
Creative writing     | Top-P (0.95)      | Unexpected twists valued
Factual Q&A          | Greedy            | Accuracy paramount
Translation          | Beam search       | Multiple hypotheses
```

### 1.2 Phạm Vi Nghiên Cứu

#### 1.2.1 Phương Pháp Được Phân Tích

Bài viết này tập trung vào **bốn phương pháp chính**:

1. **Multinomial Sampling** (Pure Probabilistic)
   - Random selection theo exact probabilities
   - Baseline stochastic method

2. **Greedy Decoding** (Deterministic)
   - Always picks highest probability token
   - No randomness

3. **Top-K Sampling** (Constrained Stochastic)
   - Sample từ K highest-probability tokens
   - Fixed number of candidates

4. **Top-P / Nucleus Sampling** (Adaptive Stochastic)
   - Sample từ smallest set với cumulative prob ≥ P
   - Variable number of candidates

#### 1.2.2 Phương Pháp Khác (Không Chi Tiết Trong Bài)

**Briefly mentioned:**
- **Ancestral Sampling**: Variant của pure sampling
- **Beam Search**: Maintains multiple hypotheses [2]
- **Diverse Beam Search**: Encou18-RAGes diversity across beams [3]
- **Contrastive Decoding**: Uses model pairs [4]

**Note:**
> Tất cả probabilistic methods có similarities—core principles tương đồng, chỉ khác implementation details.

### 1.3 Cấu Trúc Bài Viết

**Organization:**
- **Section 2**: Mathematical foundations và formal definitions
- **Section 3**: Chi tiết từng phương pháp (Greedy, Top-K, Top-P)
- **Section 4**: So sánh và trade-offs
- **Section 5**: Implementation trong PyTorch
- **Section 6**: Applications và use case analysis
- **Section 7**: The hallucination problem
- **Section 8**: Best practices và recommendations

---

## 2. Nền Tảng Toán Học

### 2.1 Setup: Probability Distribution Over Tokens

#### 2.1.1 Model Output

Sau khi model xử lý context, output là probability distribution:

$$P(w_t | w_{1:t-1}) = \text{Softmax}(\mathbf{z}_t / T)$$

Trong đó:
- $w_t$ = next token to predict
- $w_{1:t-1}$ = context (previous tokens)
- $\mathbf{z}_t$ = logits vector [vocab_size]
- $T$ = temperature parameter

**Result:**
$$\mathbf{p} = [p_1, p_2, \ldots, p_V]$$

Trong đó:
- $V$ = vocabulary size (e.g., 100,000)
- $p_i \geq 0$ (non-negativity)
- $\sum_{i=1}^V p_i = 1$ (normalization)

#### 2.1.2 Example Distribution

**Context:** "I prefer oat milk in my ___"

**Top-5 probabilities (simplified):**
```
Token       | Probability
------------|------------
coffee      | 0.340
tea         | 0.285
cereal      | 0.195
smoothie    | 0.085
mouth       | 0.045
...         | ...
[others]    | 0.050
------------|------------
TOTAL       | 1.000
```

**Observation:**
- Top token (coffee): 34%
- Top-2: 62.5% cumulative
- Top-5: 95% cumulative
- Remaining ~99,995 tokens: 5%

### 2.2 Decoding Problem Definition

**Formal problem:**
> Cho probability distribution $\mathbf{p}$ over vocabulary $\mathcal{V}$, chọn next token $w_t \in \mathcal{V}$.

**Objectives (potentially conflicting):**
1. **Likelihood maximization**: Choose high-probability tokens
2. **Diversity**: Avoid repetitive outputs
3. **Coherence**: Maintain semantic consistency
4. **Exploration**: Enable surprising but valid continuations

**Challenge:**
> No single objective function satisfies tất cả goals simultaneously—hence multiple decoding strategies.

---

## 3. Phương Pháp Lấy Mẫu Chi Tiết

### 3.1 Multinomial Sampling (Baseline)

#### 3.1.1 Definition

**Pure probabilistic sampling:**
$$w_t \sim \text{Multinomial}(\mathbf{p})$$

**Meaning:**
- Mỗi token có probability $p_i$ được chọn
- Higher probability → higher chance, nhưng không guaranteed
- Any token có $p_i > 0$ có thể được chọn

**PyTorch implementation:**
```python
import torch

probs = torch.tensor([0.340, 0.285, 0.195, 0.085, 0.045])
next_token = torch.multinomial(probs, num_samples=1)
```

#### 3.1.2 Properties

**Advantages:**
- ✓ Respects full probability distribution
- ✓ Allows exploration của entire vocabulary
- ✓ Simple và theoretically principled
- ✓ No hyperparameters (trừ temperature)

**Disadvantages:**
- ✗ Can select very low-probability tokens
- ✗ May produce incoherent outputs
- ✗ High variance in output quality
- ✗ Risk of "going off the rails"

**Example behavior:**
```python
# Run 5 times với same context
for i in range(5):
    token = sample(probs)
    print(f"Sample {i+1}: {token}")

# Possible outputs:
# Sample 1: coffee   (34% chance)
# Sample 2: tea      (28.5% chance)
# Sample 3: coffee   (34% chance)
# Sample 4: smoothie (8.5% chance)
# Sample 5: mouth    (4.5% chance) - Low prob but possible!
```

#### 3.1.3 Statistical Analysis

**Expected frequency over N samples:**
$$\mathbb{E}[\text{count}(w_i)] = N \cdot p_i$$

**Example with N=1000:**
- coffee: ~340 times
- tea: ~285 times
- cereal: ~195 times
- smoothie: ~85 times
- mouth: ~45 times

**Variance:**
$$\text{Var}[\text{count}(w_i)] = N \cdot p_i \cdot (1 - p_i)$$

**Observation:**
> High variance cho low-probability tokens → unpredictable behavior.

### 3.2 Greedy Decoding

#### 3.2.1 Definition

**Deterministic selection:**
$$w_t = \arg\max_{w \in \mathcal{V}} P(w | w_{1:t-1})$$

**Algorithm:**
```
1. Compute probabilities p = Softmax(logits)
2. Find index of maximum probability
3. Return corresponding token
```

**PyTorch implementation:**
```python
def greedy_decode(logits):
    """
    Greedy decoding: always pick highest probability
    
    Args:
        logits: Model output logits [vocab_size]
    
    Returns:
        token_id: Index of highest-probability token
    """
    probs = torch.softmax(logits, dim=-1)
    token_id = torch.argmax(probs)
    return token_id
```

#### 3.2.2 Properties

**Advantages:**
- ✓ Deterministic: same input → same output
- ✓ Fast: no sampling overhead
- ✓ Predictable behavior
- ✓ Often selects reasonable tokens
- ✓ Good for factual/technical content

**Disadvantages:**
- ✗ No diversity: repetitive outputs
- ✗ Can get stuck in loops
- ✗ Ignores alternative valid continuations
- ✗ Sounds "robotic" in conversational contexts
- ✗ Potential for exposure bias [5]

**Example scenario:**

**Distribution 1 (Clear winner):**
```
coffee:   0.85
tea:      0.08
cereal:   0.04
smoothie: 0.02
mouth:    0.01
→ Greedy picks: coffee ✓ (reasonable)
```

**Distribution 2 (Competitive):**
```
coffee:   0.215
tea:      0.210
cereal:   0.205
smoothie: 0.200
mouth:    0.170
→ Greedy picks: coffee (ALWAYS!)
```

**Problem:**
> Trong Distribution 2, tất cả top-5 tokens gần như equally plausible, nhưng greedy **always** picks coffee, ignoring 78.5% of probability mass.

#### 3.2.3 Repetition Problem

**Degenerate behavior:**
```python
# Hypothetical example
context = "The cat sat on the"
generated = greedy_generate(model, context, max_len=20)

# Possible output:
"The cat sat on the mat. The cat sat on the mat. The cat sat..."
```

**Mechanism:**
1. Model generates high-prob token
2. That token appears in context
3. Model learns patterns → continues pattern
4. Loop reinforces itself

**Real example từ literature:**
> "I don't know, I don't know, I don't know, I don't know..." [6]

**Mitigation:**
- N-gram blocking (prevent repeating sequences)
- Repetition penalty (decrease prob of recent tokens)
- Switch to stochastic methods

### 3.3 Top-K Sampling

#### 3.3.1 Definition

**Constrained probabilistic sampling:**
1. Sort probabilities descending: $p_{(1)} \geq p_{(2)} \geq \cdots \geq p_{(V)}$
2. Keep only top-K tokens
3. Renormalize probabilities
4. Sample từ truncated distribution

**Mathematical formulation:**
$$\mathcal{V}_K = \{w_i : p_i \text{ is in top-K probabilities}\}$$

$$P_K(w) = \begin{cases}
\frac{p_w}{\sum_{w' \in \mathcal{V}_K} p_{w'}} & \text{if } w \in \mathcal{V}_K \\
0 & \text{otherwise}
\end{cases}$$

**PyTorch implementation:**
```python
def top_k_sampling(logits, k=50):
    """
    Top-K sampling
    
    Args:
        logits: Model logits [vocab_size]
        k: Number of top tokens to consider
    
    Returns:
        token_id: Sampled token
    """
    # Get top-K logits and indices
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # Softmax over top-K only
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    
    # Sample from top-K distribution
    sampled_index = torch.multinomial(top_k_probs, num_samples=1)
    
    # Map back to original vocabulary
    token_id = top_k_indices[sampled_index]
    
    return token_id
```

#### 3.3.2 Example với Different K Values

**Original distribution:**
```
Rank | Token    | Probability
-----|----------|------------
1    | coffee   | 0.340
2    | tea      | 0.285
3    | cereal   | 0.195
4    | smoothie | 0.085
5    | mouth    | 0.045
6    | pocket   | 0.040
...  | ...      | ...
```

**K = 3:**
```
Selected: {coffee, tea, cereal}
Renormalized probabilities:
  coffee:  0.340 / 0.820 = 0.415
  tea:     0.285 / 0.820 = 0.348
  cereal:  0.195 / 0.820 = 0.238

Excluded: {smoothie, mouth, pocket, ...}
```

**K = 2:**
```
Selected: {coffee, tea}
Renormalized:
  coffee: 0.340 / 0.625 = 0.544
  tea:    0.285 / 0.625 = 0.456

Note: 50-50 shot between coffee/tea, 
      despite original 34% vs 28.5%!
```

**K = 1:**
```
Selected: {coffee}
Probability: 1.0

Equivalent to greedy decoding!
```

#### 3.3.3 Properties Analysis

**Advantages:**
- ✓ Controls exploration: prevents very low-prob tokens
- ✓ Maintains diversity: multiple options available
- ✓ Simple hyperparameter (K)
- ✓ Computationally efficient (top-K is fast operation)

**Disadvantages:**
- ✗ **Fixed K problem**: Doesn't adapt to probability distribution shape
- ✗ Can exclude plausible tokens arbitrarily
- ✗ Can include implausible tokens if K too large
- ✗ Renormalization distorts original probabilities

**Critical Issue: Fixed K Ignores Distribution Shape**

**Scenario 1: Peaked distribution**
```
coffee: 0.92
tea:    0.03
cereal: 0.02
...

With K=10: Includes 9 nearly-zero-probability tokens!
Better: K=1 or K=2
```

**Scenario 2: Flat distribution**
```
coffee:   0.11
tea:      0.10
cereal:   0.10
smoothie: 0.09
...

With K=3: Excludes many reasonable options!
Better: K=20 or more
```

**Observation:**
> Optimal K varies dramatically based on model confidence. Fixed K is suboptimal.

### 3.4 Top-P (Nucleus) Sampling

#### 3.4.1 Definition

**Adaptive probabilistic sampling:**
1. Sort probabilities descending
2. Find **smallest set** với cumulative probability ≥ P
3. Sample từ that set

**Mathematical formulation:**
$$\mathcal{V}_P = \{w_{(1)}, w_{(2)}, \ldots, w_{(m)}\}$$

Trong đó $m$ là smallest index such that:
$$\sum_{i=1}^m p_{(i)} \geq P$$

**Sampling distribution:**
$$P_P(w) = \begin{cases}
\frac{p_w}{\sum_{w' \in \mathcal{V}_P} p_{w'}} & \text{if } w \in \mathcal{V}_P \\
0 & \text{otherwise}
\end{cases}$$

**PyTorch implementation:**
```python
def top_p_sampling(logits, p=0.9):
    """
    Top-P (Nucleus) sampling
    
    Args:
        logits: Model logits [vocab_size]
        p: Cumulative probability threshold (0 < p ≤ 1)
    
    Returns:
        token_id: Sampled token
    """
    # Sort probabilities descending
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff index
    # Keep tokens until cumulative prob exceeds p
    cutoff = (cumulative_probs > p).nonzero()[0].item()
    
    # Include tokens up to cutoff
    nucleus_probs = sorted_probs[:cutoff+1]
    nucleus_indices = sorted_indices[:cutoff+1]
    
    # Renormalize
    nucleus_probs = nucleus_probs / nucleus_probs.sum()
    
    # Sample
    sampled_index = torch.multinomial(nucleus_probs, num_samples=1)
    token_id = nucleus_indices[sampled_index]
    
    return token_id
```

#### 3.4.2 Example với Different P Values

**Distribution:**
```
Rank | Token    | Probability | Cumulative
-----|----------|-------------|------------
1    | coffee   | 0.340       | 0.340
2    | tea      | 0.285       | 0.625
3    | cereal   | 0.195       | 0.820
4    | smoothie | 0.085       | 0.905
5    | mouth    | 0.045       | 0.950
6    | pocket   | 0.040       | 0.990
```

**P = 0.98 (98%):**
```
Cumulative ≥ 0.98 at token 6 (pocket)
Nucleus: {coffee, tea, cereal, smoothie, mouth, pocket}
Size: 6 tokens

→ Sample from 6 tokens
```

**P = 0.90 (90%):**
```
Cumulative ≥ 0.90 at token 4 (smoothie)
Nucleus: {coffee, tea, cereal, smoothie}
Size: 4 tokens

→ Sample from 4 tokens
```

**P = 0.50 (50%):**
```
Cumulative ≥ 0.50 at token 2 (tea)
Nucleus: {coffee, tea}
Size: 2 tokens

→ Sample from 2 tokens
```

#### 3.4.3 Adaptive Behavior

**Key advantage: Nucleus size adapts to distribution shape**

**Scenario 1: High confidence (peaked distribution)**
```
Token     | Prob  | Cumulative
----------|-------|------------
the       | 0.92  | 0.92
a         | 0.03  | 0.95
...

With P=0.9:
  Nucleus = {the}  (size 1)
  → Acts like greedy! ✓

Reason: Model is confident, should be deterministic
```

**Scenario 2: Low confidence (flat distribution)**
```
Token     | Prob  | Cumulative
----------|-------|------------
coffee    | 0.11  | 0.11
tea       | 0.10  | 0.21
cereal    | 0.10  | 0.31
smoothie  | 0.09  | 0.40
...
(continuing to ~10th token to reach 0.9)

With P=0.9:
  Nucleus = {coffee, tea, ..., token_10} (size ~10)
  → Acts like top-K with adaptive K! ✓

Reason: Model uncertain, should explore options
```

**Comparison:**

| Confidence | Distribution | Top-K (K=5) | Top-P (P=0.9) |
|------------|--------------|-------------|---------------|
| High | Peaked | 5 tokens (overkill) | 1-2 tokens ✓ |
| Low | Flat | 5 tokens (too few) | 8-12 tokens ✓ |

**Conclusion:**
> Top-P automatically adjusts nucleus size based on model confidence—more principled than fixed K.

#### 3.4.4 Properties Analysis

**Advantages:**
- ✓ **Adaptive**: Nucleus size fits distribution shape
- ✓ Greedy-like khi model confident
- ✓ Exploratory khi model uncertain
- ✓ Principled: based on cumulative probability
- ✓ One hyperparameter (P) works across contexts

**Disadvantages:**
- ✗ Slightly more complex to implement
- ✗ Computational overhead (sorting, cumsum)
- ✗ Still truncates distribution (loses tail)
- ✗ Renormalization can distort probabilities

**Typical P values:**
```
P = 0.9:  Standard, balanced
P = 0.95: More exploratory
P = 0.8:  More conservative
P = 1.0:  Pure multinomial (no truncation)
```

---

## 4. So Sánh Phương Pháp

### 4.1 Bảng Tổng Hợp

| Method | Deterministic? | Parameters | Nucleus Size | Adapts to Distribution? |
|--------|----------------|------------|--------------|------------------------|
| **Multinomial** | No | Temperature | Full vocab | N/A |
| **Greedy** | Yes | None | 1 token | No |
| **Top-K** | No | K | Fixed K | No |
| **Top-P** | No | P | Variable | Yes ✓ |

### 4.2 Trade-offs Chi Tiết

#### 4.2.1 Diversity vs Quality

**Diversity spectrum:**
```
Low Diversity                                    High Diversity
(Repetitive)                                     (Unpredictable)
    |------------|------------|------------|------------|
  Greedy      Top-P(0.5)   Top-P(0.9)   Multinomial
              Top-K(5)     Top-K(50)
```

**Quality considerations:**

**High diversity (Multinomial, Top-K large, Top-P high):**
- ✓ Varied outputs
- ✓ Surprising continuations
- ✗ Can be incoherent
- ✗ May hallucinate

**Low diversity (Greedy, Top-K small, Top-P low):**
- ✓ Coherent outputs
- ✓ Factually grounded
- ✗ Repetitive
- ✗ Boring

#### 4.2.2 Computational Cost

**Complexity analysis:**

```python
# Assuming vocab_size = V, K << V

Method          | Time Complexity | Space | Notes
----------------|-----------------|-------|-------
Multinomial     | O(V)            | O(V)  | Full softmax
Greedy          | O(V)            | O(1)  | Simple argmax
Top-K           | O(V + K log K)  | O(K)  | Sorting/heap
Top-P           | O(V log V)      | O(V)  | Full sort + cumsum
```

**Practical speed:**
- **Fastest**: Greedy
- **Fast**: Top-K
- **Moderate**: Multinomial
- **Slower**: Top-P (due to sorting)

**Note:**
> Trong practice, với modern hardware và optimized implementations, differences thường negligible cho inference.

#### 4.2.3 Hyperparameter Sensitivity

**Greedy:**
- No hyperparameters
- Easy to use
- No tuning needed

**Top-K:**
```
K value    | Effect
-----------|---------------------------
K = 1      | Equivalent to greedy
K = 5-10   | Conservative exploration
K = 40-50  | Moderate diversity
K = 100+   | Approaching multinomial
```

**Top-P:**
```
P value    | Effect
-----------|---------------------------
P = 0.5    | Very conservative
P = 0.9    | Standard (recommended)
P = 0.95   | More exploratory
P = 0.99   | Very exploratory
P = 1.0    | Multinomial
```

**Sensitivity:**
- Top-K: **High** sensitivity (K=5 vs K=50 very different)
- Top-P: **Moderate** sensitivity (P=0.9 vs P=0.95 noticeable but manageable)

### 4.3 Interaction với Temperature

**Combined effects:**

Temperature scaling happens **before** sampling method:
$$\text{logits} \xrightarrow{/T} \text{scaled logits} \xrightarrow{\text{Softmax}} \text{probs} \xrightarrow{\text{Sampling}} \text{token}$$

**Examples:**

**Low temp (T=0.7) + Greedy:**
- Very deterministic
- Sharp distribution → same token always
- Use for: Factual Q&A

**Low temp (T=0.7) + Top-P(0.9):**
- Mostly deterministic
- Small nucleus
- Use for: Technical writing

**High temp (T=1.2) + Top-P(0.95):**
- Very exploratory
- Large nucleus
- Use for: Creative fiction

**High temp (T=1.5) + Multinomial:**
- Maximum diversity
- Risk of incoherence
- Use for: Experimental generation

**Recommendation:**
> Tune temperature và sampling method **together**, không independently. Optimal combinations depend on use case.

---

## 5. Implementation trong PyTorch

### 5.1 Complete Sampling Module

```python
"""
token_sampling.py
Comprehensive implementation of token sampling strategies
"""

import torch
import torch.nn.functional as F
from typing import Optional, Literal


class TokenSampler:
    """
    Unified interface for token sampling strategies
    """
    
    def __init__(
        self,
        method: Literal['greedy', 'multinomial', 'top_k', 'top_p'] = 'top_p',
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.9
    ):
        """
        Initialize sampler
        
        Args:
            method: Sampling method to use
            temperature: Temperature scaling (default 1.0)
            top_k: K value for top-k sampling
            top_p: P value for top-p sampling
        """
        self.method = method
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Validate parameters
        if method == 'top_k' and top_k is None:
            raise ValueError("top_k must be specified for top-k sampling")
        if method == 'top_p' and top_p is None:
            raise ValueError("top_p must be specified for top-p sampling")
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample next token
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
        
        Returns:
            next_tokens: Sampled token IDs [batch_size]
        """
        # Apply temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Route to appropriate method
        if self.method == 'greedy':
            return self._greedy(logits)
        elif self.method == 'multinomial':
            return self._multinomial(logits)
        elif self.method == 'top_k':
            return self._top_k(logits)
        elif self.method == 'top_p':
            return self._top_p(logits)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy decoding: argmax"""
        return torch.argmax(logits, dim=-1)
    
    def _multinomial(self, logits: torch.Tensor) -> torch.Tensor:
        """Pure multinomial sampling"""
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _top_k(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Top-K sampling
        
        Implementation:
        1. Get top-K logits
        2. Mask out others (-inf)
        3. Softmax + sample
        """
        # Get top-K values and indices
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Create mask for top-K
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, top_k_indices, top_k_logits)
        
        # Softmax over masked logits
        probs = F.softmax(mask, dim=-1)
        
        # Sample
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _top_p(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Top-P (Nucleus) sampling
        
        Implementation:
        1. Sort probabilities descending
        2. Compute cumulative sum
        3. Find nucleus (cumsum > p)
        4. Mask + sample
        """
        # Sort probabilities
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff: first position where cumsum > p
        # Keep tokens BEFORE cutoff (nucleus)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Create mask
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        
        # Set removed indices to -inf
        logits_masked = logits.clone()
        logits_masked[mask] = float('-inf')
        
        # Softmax + sample
        probs_masked = F.softmax(logits_masked, dim=-1)
        return torch.multinomial(probs_masked, num_samples=1).squeeze(-1)


# Example usage
if __name__ == "__main__":
    # Simulate model output
    batch_size = 4
    vocab_size = 50000
    logits = torch.randn(batch_size, vocab_size)
    
    print("Sampling Comparison\n" + "="*50)
    
    # Test different methods
    methods = {
        'Greedy': TokenSampler('greedy'),
        'Multinomial': TokenSampler('multinomial'),
        'Top-K (k=50)': TokenSampler('top_k', top_k=50),
        'Top-P (p=0.9)': TokenSampler('top_p', top_p=0.9),
    }
    
    for name, sampler in methods.items():
        tokens = sampler.sample(logits)
        print(f"{name:20s}: {tokens.tolist()}")
    
    print("\nTemperature Effects\n" + "="*50)
    
    # Test temperature
    for temp in [0.7, 1.0, 1.5]:
        sampler = TokenSampler('top_p', temperature=temp, top_p=0.9)
        tokens = sampler.sample(logits)
        print(f"T={temp:.1f} + Top-P:     {tokens.tolist()}")
```

### 5.2 Generation Loop

```python
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 50,
    sampler: Optional[TokenSampler] = None
) -> str:
    """
    Generate text using specified sampling strategy
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input text
        max_length: Maximum tokens to generate
        sampler: TokenSampler instance (default: top-p)
    
    Returns:
        generated_text: Complete generated text
    """
    # Default sampler
    if sampler is None:
        sampler = TokenSampler('top_p', temperature=1.0, top_p=0.9)
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generation loop
    for _ in range(max_length):
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # Last token logits
        
        # Sample next token
        next_token = sampler.sample(logits)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        
        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Decode
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


# Example usage
if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    prompt = "The future of artificial intelligence is"
    
    print("Comparison of Sampling Methods\n" + "="*70)
    
    # Compare methods
    configs = [
        ('Greedy', TokenSampler('greedy')),
        ('Top-K (k=50)', TokenSampler('top_k', temperature=1.0, top_k=50)),
        ('Top-P (p=0.9)', TokenSampler('top_p', temperature=1.0, top_p=0.9)),
        ('Top-P (p=0.95)', TokenSampler('top_p', temperature=1.0, top_p=0.95)),
    ]
    
    for name, sampler in configs:
        text = generate_text(model, tokenizer, prompt, max_length=30, sampler=sampler)
        print(f"\n{name}:")
        print(f"  {text}")
```

### 5.3 Evaluation Metrics

```python
def evaluate_diversity(texts: list[str]) -> dict:
    """
    Compute diversity metrics for generated texts
    
    Args:
        texts: List of generated texts
    
    Returns:
        metrics: Dictionary of diversity metrics
    """
    from collections import Counter
    import numpy as np
    
    # Tokenize
    all_tokens = []
    for text in texts:
        tokens = text.split()
        all_tokens.extend(tokens)
    
    # Unique tokens
    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    
    # Type-Token Ratio (TTR)
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
    
    # Entropy
    token_counts = Counter(all_tokens)
    probs = np.array(list(token_counts.values())) / total_tokens
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    # Self-BLEU (diversity within set)
    # Lower is more diverse
    # (Implementation omitted for brevity)
    
    return {
        'unique_tokens': unique_tokens,
        'total_tokens': total_tokens,
        'type_token_ratio': ttr,
        'entropy': entropy,
    }


# Benchmark different methods
def benchmark_sampling_methods(model, tokenizer, prompts: list[str], n_samples=10):
    """Compare sampling methods on diversity"""
    
    methods = {
        'Greedy': TokenSampler('greedy'),
        'Top-K(50)': TokenSampler('top_k', top_k=50),
        'Top-P(0.9)': TokenSampler('top_p', top_p=0.9),
        'Multinomial': TokenSampler('multinomial'),
    }
    
    results = {}
    
    for method_name, sampler in methods.items():
        texts = []
        for prompt in prompts:
            for _ in range(n_samples):
                text = generate_text(model, tokenizer, prompt, 
                                    max_length=30, sampler=sampler)
                texts.append(text)
        
        metrics = evaluate_diversity(texts)
        results[method_name] = metrics
    
    return results
```

---

## 6. Applications và Use Cases

### 6.1 Task-Specific Recommendations

#### 6.1.1 Factual Question Answering

**Requirements:**
- High accuracy
- Minimal hallucination
- Reproducibility

**Recommended configuration:**
```python
# Configuration 1: Pure greedy
sampler = TokenSampler('greedy')

# Configuration 2: Conservative top-p
sampler = TokenSampler('top_p', temperature=0.7, top_p=0.85)
```

**Examples:**
```
Q: "What is the capital of France?"
A: "Paris" (should be deterministic)

Q: "When was the Eiffel Tower built?"
A: "1889" (factual, no creativity needed)
```

**Why these settings:**
- Greedy/low-temp: Selects highest-probability (most likely correct) answer
- Small nucleus: Prevents low-probability hallucinations
- Deterministic: Same answer every time → reliable

#### 6.1.2 Code Generation

**Requirements:**
- Syntactic correctness
- Logical coherence
- Runnable code

**Recommended configuration:**
```python
# Conservative generation
sampler = TokenSampler('top_p', temperature=0.8, top_p=0.9)
```

**Example:**
```python
Prompt: "Write a Python function to reverse a string"

# Good (with conservative sampling):
def reverse_string(s):
    return s[::-1]

# Bad (with high-diversity sampling):
def reverse_string(s):
    # Uses quantum algorithms
    return qbit_reverse(s)  # Hallucinated!
```

**Why:**
- Code has strict syntax → need high probability tokens
- Small mistakes → syntax errors
- Some creativity ok (variable names, approaches) but within bounds

#### 6.1.3 Creative Writing

**Requirements:**
- Novelty
- Engaging plot developments
- Varied vocabulary

**Recommended configuration:**
```python
# Exploratory generation
sampler = TokenSampler('top_p', temperature=1.1, top_p=0.95)
```

**Example:**
```
Prompt: "Once upon a time in a magical forest"

With conservative (greedy):
"Once upon a time in a magical forest, there lived a young girl 
named Alice who loved to explore..."
(Predictable, common tropes)

With creative (top-p 0.95, T=1.1):
"Once upon a time in a magical forest, the trees whispered 
secrets in a language only the moonlight could translate..."
(More original, unexpected imagery)
```

**Why:**
- Unexpected twists valued
- Repetitive = boring
- Factual accuracy less critical

#### 6.1.4 Dialogue / Chatbots

**Requirements:**
- Natural conversation flow
- Some variability (not robotic)
- Coherence

**Recommended configuration:**
```python
# Balanced generation
sampler = TokenSampler('top_p', temperature=0.9, top_p=0.92)
```

**Example:**
```
User: "How was your day?"

Greedy (too robotic):
"My day was good. How was yours?"
(Every time same response)

Top-P balanced:
Response 1: "Pretty interesting! I learned about quantum computing."
Response 2: "It was great, thanks for asking! How about you?"
Response 3: "Not bad! Had some fascinating conversations."
(Varied but appropriate)
```

**Why:**
- Humans don't repeat exact phrases
- Need variety for natural feel
- But stay coherent and on-topic

#### 6.1.5 Translation

**Requirements:**
- Semantic accuracy
- Grammatical correctness
- Usually single best translation

**Recommended configuration:**
```python
# Often use beam search instead
# But if using sampling:
sampler = TokenSampler('greedy')
# Or very conservative top-p
sampler = TokenSampler('top_p', temperature=0.6, top_p=0.8)
```

**Why:**
- Translation typically has "correct" answer
- Creativity not desired
- Accuracy paramount

### 6.2 Configuration Matrix

| Application | Temperature | Method | Parameters | Rationale |
|-------------|-------------|--------|------------|-----------|
| Q&A | 0.7 | Greedy | - | Accuracy > diversity |
| Code | 0.8 | Top-P | p=0.9 | Correctness critical |
| Chat | 0.9-1.0 | Top-P | p=0.92 | Natural variation |
| Creative | 1.1-1.2 | Top-P | p=0.95 | Novelty valued |
| Translation | 0.6 | Greedy | - | Single correct answer |
| Summarization | 0.8 | Top-P | p=0.88 | Concise + accurate |

### 6.3 Real-World Examples

#### 6.3.1 ChatGPT Configuration

**Reported settings (approximate):**
- Method: Top-P
- P: ~0.95
- Temperature: User-adjustable (default ~1.0)

**Why:**
- Balances coherence và diversity
- Adapts to model confidence
- Generally produces natural outputs

#### 6.3.2 GitHub Copilot

**Likely settings:**
- Method: Top-P with low temperature
- Temperature: ~0.2-0.4
- P: ~0.8

**Why:**
- Code correctness critical
- Deterministic behavior preferred
- Still allows some variability cho different valid solutions

#### 6.3.3 AI Dungeon (Creative Game)

**Settings:**
- Method: Top-P
- Temperature: 1.0-1.5 (user adjustable)
- P: 0.9-0.98

**Why:**
- Creativity highly valued
- Unexpected plot twists = fun
- Some incoherence tolerable for entertainment

---

## 7. The Hallucination Problem

### 7.1 Định Nghĩa và Mechanism

#### 7.1.1 What is Hallucination?

**Definition:**
> LLM hallucination xảy ra khi model generates content that is **plausible-sounding but factually incorrect or entirely fabricated**.

**Examples:**
```
Query: "Tell me about the Battle of Hastings in 1776"
Hallucinated response: "The Battle of Hastings in 1776 was..."
Truth: Battle of Hastings was 1066, not 1776

Query: "What are the health benefits of quantum water?"
Hallucinated response: "Quantum water has been shown to..."
Truth: "Quantum water" is pseudoscience

Query: "Who invented the telephone in ancient Rome?"
Hallucinated response: "Marcus Telephonicus invented..."
Truth: Telephones didn't exist in ancient Rome
```

#### 7.1.2 Root Causes

**Cause 1: Training data artifacts**
- Model learns correlations, not facts
- Memorizes patterns without understanding
- "Sounds right" ≠ "is right"

**Cause 2: Probabilistic nature**
- Next-token prediction based on probability
- No explicit fact-checking mechanism
- High-probability ≠ true

**Cause 3: Sampling introduces randomness**
- Stochastic methods can select incorrect tokens
- Even low-probability wrong answers can be chosen
- Compounding errors through sequence

**Example cascade:**
```
Context: "The capital of France is"

Token 1: "Paris" (p=0.98) ✓ Correct
Token 2: "," (p=0.95) ✓ 
Token 3: "located" (p=0.87) ✓
Token 4: "on" (p=0.82) ✓
Token 5: "the" (p=0.91) ✓
Token 6: "moon" (p=0.003) ✗ Sampled despite low prob!

Result: "The capital of France is Paris, located on the moon"
```

### 7.2 Sampling Methods và Hallucination Risk

#### 7.2.1 Risk Analysis

**Hallucination risk spectrum:**
```
Low Risk                                        High Risk
(Factual)                                       (Creative but wrong)
    |------------|------------|------------|------------|
  Greedy      Top-P(0.8)   Top-P(0.95)   Multinomial
  T=0.7       T=0.9        T=1.2         T=1.5
```

**Method comparison:**

**Greedy:**
- ✓ Always picks most likely token
- ✓ If training data correct, output likely correct
- ✗ Can still hallucinate if top token is wrong
- ✗ May repeat hallucinated patterns

**Top-K / Top-P:**
- ✗ Can sample less likely (potentially wrong) tokens
- ✗ Small nucleus still has hallucination risk
- ✗ Large nucleus increases risk
- ✓ Can "recover" from model mistakes

**Multinomial:**
- ✗ Highest hallucination risk
- ✗ Even very low-prob tokens can be selected
- ✗ Compounds through sequence

#### 7.2.2 The Fundamental Dilemma

**Problem statement:**
> To generate interesting, creative, varied text, we need stochastic sampling. But stochasticity introduces risk of selecting incorrect tokens, leading to hallucinations.

**Trade-off:**
```
Deterministic (Greedy)
  ✓ Fewer hallucinations (usually)
  ✗ Boring, repetitive
  ✗ Can still hallucinate

Stochastic (Top-P, etc.)
  ✓ Diverse, interesting
  ✓ Natural-sounding
  ✗ More hallucinations
```

**No perfect solution:**
- Can't eliminate hallucinations without eliminating diversity
- Can't maximize diversity without risking hallucinations
- Different applications make different trade-offs

### 7.3 Mitigation Strategies

#### 7.3.1 Sampling-Level Mitigations

**Strategy 1: Conservative parameters**
```python
# For factual tasks
sampler = TokenSampler(
    'top_p',
    temperature=0.7,    # Lower temperature
    top_p=0.85          # Smaller nucleus
)
```

**Strategy 2: Factuality-aware sampling**
```python
# Penalize known-unreliable tokens
# (Requires external knowledge base)
def factual_sampling(logits, knowledge_base):
    probs = F.softmax(logits, dim=-1)
    
    # Penalize tokens known to be wrong in context
    for token_id in get_incorrect_tokens(knowledge_base, context):
        probs[token_id] *= 0.5  # Penalty
    
    # Renormalize and sample
    probs = probs / probs.sum()
    return torch.multinomial(probs, num_samples=1)
```

**Strategy 3: Constrained decoding**
- Force output to match template
- Require citations/sources
- Limit to verified knowledge base

#### 7.3.2 Model-Level Solutions

**Approach 1: Retrieval-Augmented Generation (RAG)**
- Retrieve relevant documents before generation
- Ground generation in retrieved facts
- Reduces hallucination significantly [7]

**Approach 2: Fact-checking modules**
- Separate model verifies factual claims
- Reject or flag hallucinations
- Post-hoc filtering

**Approach 3: Reinforcement Learning from Human Feedback (RLHF)**
- Train model to prefer factual outputs
- Penalize hallucinations during training
- Used in ChatGPT, Claude, etc. [8]

#### 7.3.3 User-Level Best Practices

**For users:**
1. **Verify critical facts** independently
2. **Use appropriate settings** (greedy cho facts)
3. **Ask for sources** when possible
4. **Cross-reference** important claims
5. **Understand limitations** của LLMs

**For developers:**
1. **Choose method** based on use case
2. **Test hallucination rate** on your data
3. **Implement safeguards** for high-risk applications
4. **Provide uncertainty estimates** when possible
5. **Document limitations** clearly

### 7.4 Current State và Future Directions

**Current challenges:**
- No perfect solution exists
- Fundamental to next-token prediction paradigm
- Trade-offs unavoidable

**Active research areas:**
- Uncertainty quantification [9]
- Fact verification during generation
- Constitutional AI [10]
- Iterative refinement methods

**Conclusion:**
> Hallucination remains **unresolved fundamental challenge**. Choice of sampling method affects risk but cannot eliminate it. Applications must balance creativity needs with factuality requirements.

---

## 8. Best Practices và Recommendations

### 8.1 Decision Framework

**Flowchart:**
```
START: Choose sampling method

1. Is factual accuracy critical?
   YES → Use Greedy or Top-P with T=0.7, p=0.85
   NO → Continue to 2

2. Is creativity/diversity important?
   YES → Continue to 3
   NO → Use Greedy or Top-K with small K

3. Does model confidence vary widely?
   YES → Use Top-P (adaptive nucleus)
   NO → Use Top-K (simpler)

4. Select hyperparameters:
   - Factual: T=0.7-0.8, p=0.85-0.9, K=5-20
   - Balanced: T=0.9-1.0, p=0.9-0.95, K=40-50
   - Creative: T=1.1-1.3, p=0.95-0.98, K=50+
```

### 8.2 Hyperparameter Tuning Guide

#### 8.2.1 Temperature

**Guidelines:**
- Start: 1.0 (neutral)
- Increase (1.1-1.5): More random, creative
- Decrease (0.6-0.9): More focused, deterministic
- Extreme low (<0.5): Nearly greedy
- Extreme high (>2.0): Nearly uniform (avoid)

**Tuning process:**
```python
# Test range
temperatures = [0.7, 0.9, 1.0, 1.2, 1.5]

for T in temperatures:
    sampler = TokenSampler('top_p', temperature=T, top_p=0.9)
    outputs = generate_samples(model, prompt, sampler, n=10)
    
    # Evaluate diversity
    diversity = evaluate_diversity(outputs)
    
    # Evaluate quality (human or automated)
    quality = evaluate_quality(outputs)
    
    print(f"T={T}: Diversity={diversity:.3f}, Quality={quality:.3f}")
```

#### 8.2.2 Top-K

**Guidelines:**
- Small K (1-10): Conservative, focused
- Medium K (20-50): Balanced
- Large K (100+): Exploratory
- K=1: Equivalent to greedy

**Selection criteria:**
- Vocab size matters: K=50 reasonable cho 50K vocab
- Task matters: Code generation → small K
- Model quality matters: Better models → can use larger K

#### 8.2.3 Top-P

**Guidelines:**
- Standard: 0.9 (90%)
- Conservative: 0.8-0.85
- Exploratory: 0.95-0.98
- No truncation: 1.0 (multinomial)

**Why 0.9 is common:**
- Balances diversity và quality
- Adapts to distribution shape
- Works across various tasks
- Empirically proven effective [11]

### 8.3 Common Mistakes to Avoid

#### 8.3.1 Mistake 1: Using Same Settings Everywhere

**Wrong:**
```python
# One-size-fits-all approach
sampler = TokenSampler('top_p', temperature=1.0, top_p=0.9)

# Use for ALL tasks
qa_output = generate(qa_model, sampler)        # ✗ Too random
code_output = generate(code_model, sampler)    # ✗ Too random
creative = generate(story_model, sampler)      # Maybe ok
```

**Right:**
```python
# Task-specific configurations
qa_sampler = TokenSampler('greedy')
code_sampler = TokenSampler('top_p', temperature=0.8, top_p=0.9)
creative_sampler = TokenSampler('top_p', temperature=1.1, top_p=0.95)
```

#### 8.3.2 Mistake 2: Extreme Hyperparameters

**Wrong:**
```python
# Temperature too low
sampler = TokenSampler('top_p', temperature=0.1, top_p=0.9)
# → Essentially greedy, defeats purpose of top-p

# Temperature too high
sampler = TokenSampler('top_p', temperature=3.0, top_p=0.9)
# → Nearly uniform, incoherent outputs
```

**Right:**
```python
# Reasonable ranges
temperatures = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
# Stay within this range for most applications
```

#### 8.3.3 Mistake 3: Ignoring Evaluation

**Wrong:**
```python
# Pick parameters arbitrarily
sampler = TokenSampler('top_k', top_k=42)  # Why 42? No reason
```

**Right:**
```python
# Systematic evaluation
def evaluate_config(config, test_prompts, n_samples=20):
    sampler = TokenSampler(**config)
    outputs = []
    
    for prompt in test_prompts:
        for _ in range(n_samples):
            output = generate(model, prompt, sampler)
            outputs.append(output)
    
    # Compute metrics
    diversity = compute_diversity(outputs)
    coherence = compute_coherence(outputs)
    quality = human_rating(outputs)  # or automated
    
    return {
        'config': config,
        'diversity': diversity,
        'coherence': coherence,
        'quality': quality,
    }

# Test multiple configurations
configs = [
    {'method': 'top_p', 'temperature': 0.9, 'top_p': 0.9},
    {'method': 'top_p', 'temperature': 1.0, 'top_p': 0.95},
    {'method': 'top_k', 'temperature': 1.0, 'top_k': 50},
]

results = [evaluate_config(cfg, test_prompts) for cfg in configs]
best_config = max(results, key=lambda x: x['quality'])
```

### 8.4 Testing và Validation

#### 8.4.1 Diversity Metrics

```python
def compute_diversity_metrics(texts):
    """
    Comprehensive diversity evaluation
    """
    metrics = {}
    
    # 1. Distinct-n (unigrams, bigrams)
    tokens = [text.split() for text in texts]
    all_tokens = [t for ts in tokens for t in ts]
    
    distinct_1 = len(set(all_tokens)) / len(all_tokens)
    
    bigrams = [tuple(ts[i:i+2]) for ts in tokens for i in range(len(ts)-1)]
    distinct_2 = len(set(bigrams)) / len(bigrams)
    
    metrics['distinct_1'] = distinct_1
    metrics['distinct_2'] = distinct_2
    
    # 2. Self-BLEU (lower = more diverse)
    # Compare each text against others
    # (Implementation omitted)
    
    # 3. Entropy
    from collections import Counter
    counts = Counter(all_tokens)
    probs = np.array(list(counts.values())) / len(all_tokens)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    metrics['entropy'] = entropy
    
    return metrics
```

#### 8.4.2 Quality Metrics

```python
def compute_quality_metrics(texts, references=None):
    """
    Evaluate generation quality
    """
    metrics = {}
    
    # 1. Perplexity (if model available)
    # Lower = more probable under model
    # (Implementation omitted)
    
    # 2. BLEU score (if references available)
    if references:
        # Compare generated to references
        # (Implementation omitted)
        pass
    
    # 3. Coherence (using separate model)
    # (Implementation omitted)
    
    # 4. Length statistics
    lengths = [len(text.split()) for text in texts]
    metrics['mean_length'] = np.mean(lengths)
    metrics['std_length'] = np.std(lengths)
    
    return metrics
```

### 8.5 Production Checklist

**Before deploying:**

- [ ] Choose method based on use case analysis
- [ ] Tune hyperparameters on validation set
- [ ] Evaluate diversity và quality metrics
- [ ] Test edge cases (very short/long prompts)
- [ ] Implement safeguards (max length, filtering)
- [ ] Monitor hallucination rate
- [ ] A/B test different configurations
- [ ] Document configuration choices
- [ ] Provide user controls where appropriate
- [ ] Plan for updates as models improve

---

## 9. Advanced Topics

### 9.1 Beam Search

**Brief overview:**

**Algorithm:**
1. Maintain K hypotheses (beams)
2. At each step, expand each beam
3. Keep top K by cumulative probability
4. Return highest-scoring complete sequence

**Advantages:**
- Finds higher-probability sequences than greedy
- Deterministic (same K → same output)
- Good for translation, summarization

**Disadvantages:**
- Computationally expensive (K times greedy)
- Can produce generic, repetitive outputs
- Requires careful length normalization

**When to use:**
- Tasks with "correct" outputs (translation)
- When diversity not important
- When computational cost acceptable

### 9.2 Contrastive Decoding

**Recent innovation:**

Use two models:
- **Expert model**: Large, capable
- **Amateur model**: Smaller, weaker

**Decoding:**
$$P_{\text{contrastive}}(w) \propto \frac{P_{\text{expert}}(w)}{P_{\text{amateur}}(w)^\alpha}$$

**Idea:** Amplify expert's advantages over amateur

**Benefits:**
- Reduces repetition
- Improves coherence
- Better than vanilla sampling

### 9.3 Classifier-Free Guidance

**Borrowed from diffusion models:**

Interpolate between:
- Unconditional generation
- Conditional generation

**Application to LLMs:**
- Control generation style
- Steer toward desired attributes
- Balance creativity và control

### 9.4 Speculative Decoding

**Speed optimization:**

**Idea:**
1. Draft multiple tokens với small fast model
2. Verify với large accurate model
3. Accept verified tokens
4. Speedup: ~2-3x

**Relevance to sampling:**
- Works with any sampling method
- Maintains distribution exactly
- Practical deployment technique

---

## 10. Kết Luận

### 10.1 Tóm Tắt Key Findings

**Về các phương pháp:**

**Greedy:**
- Simplest, fastest, deterministic
- Good cho factual tasks
- Risk: repetitive, boring

**Top-K:**
- Fixed-size nucleus
- Moderate complexity
- Risk: suboptimal K cho varying distributions

**Top-P:**
- Adaptive nucleus size
- Principled approach
- Current best practice cho most tasks

**Multinomial:**
- Maximum diversity
- Theoretical baseline
- Risk: incoherence, hallucinations

**Comparison:**
```
Method      | Complexity | Adaptability | Diversity | Use Case
------------|------------|--------------|-----------|----------
Greedy      | Lowest     | None         | None      | Factual
Top-K       | Low        | None         | Moderate  | General
Top-P       | Moderate   | High ✓       | Tunable   | Most tasks ✓
Multinomial | Low        | None         | Maximum   | Research
```

### 10.2 Core Insights

**Insight 1: No Universal Best Method**
> Optimal decoding strategy depends on task, model, và user preferences. One-size-fits-all không tồn tại.

**Insight 2: Trade-offs Are Fundamental**
> Cannot maximize diversity, quality, và factuality simultaneously. Different applications prioritize different objectives.

**Insight 3: Top-P Currently Dominant**
> Top-P (nucleus sampling) has emerged as preferred method cho most production LLMs due to adaptive behavior.

**Insight 4: Hallucination Remains Unsolved**
> Stochastic sampling enables creativity but introduces hallucination risk. This fundamental tension persists.

**Insight 5: Hyperparameters Matter**
> Temperature và nucleus size (K or P) critically affect output. Careful tuning essential for production quality.

### 10.3 Practical Guidelines

**For researchers:**
- Understand trade-offs between methods
- Evaluate on multiple metrics (diversity, quality, factuality)
- Consider task-specific requirements
- Report hyperparameters in papers

**For practitioners:**
- Start với Top-P (p=0.9, T=1.0)
- Tune based on application needs
- Monitor output quality continuously
- Implement safeguards for critical applications

**For users:**
- Understand what sampling controls do
- Adjust settings based on desired output
- Verify factual claims independently
- Appreciate limitations of current systems

### 10.4 Future Outlook

**Emerging trends:**

**Short-term (1-2 years):**
- Better default hyperparameters
- Automatic parameter selection
- Improved hallucination detection
- Faster sampling algorithms

**Medium-term (3-5 years):**
- Factuality-aware sampling
- Uncertainty quantification
- Retrieval-augmented generation mainstream
- Novel decoding paradigms

**Long-term (5+ years):**
- Fundamental advances beyond next-token prediction
- Reasoning-aware generation
- Verified factuality
- Human-AI collaborative decoding

### 10.5 Final Thoughts

Token sampling là **bridge between model probabilities và actual text generation**. Dù conceptually simple, nó có profound impact trên output quality, diversity, và factuality.

**Key message:**
> Choice of sampling method isn't afterthought—it's fundamental design decision that shapes user experience và determines what applications are feasible.

As LLMs continue advancing, decoding strategies sẽ evolve. Nhưng fundamental trade-offs—creativity vs accuracy, diversity vs coherence—sẽ persist. Understanding these trade-offs và choosing appropriate strategies remains essential skill cho anyone working với language models.

**Closing observation:**
The "unresolved" nature of hallucination problem mentioned ở đầu bài isn't failure—it's reflection của inherent complexity trong balancing competing objectives. Progress happens không through finding perfect solution, mà through better understanding trade-offs và developing tools to navigate them effectively.

---

## 11. Tài Liệu Tham Khảo

[1] Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019). "The Curious Case of Neural Text Degeneration." *International Conference on Learning Representations* (ICLR). https://arxiv.org/abs/1904.09751
   - **Seminal paper** introducing nucleus (top-p) sampling

[2] Freitag, M., & Al-Onaizan, Y. (2017). "Beam Search Strategies for Neural Machine Translation." *Proceedings of the First Workshop on Neural Machine Translation*, 56-60.

[3] Vijayakumar, A. K., et al. (2018). "Diverse Beam Search for Improved Description of Complex Scenes." *AAAI Conference on Artificial Intelligence*.

[4] Li, X. L., et al. (2022). "Contrastive Decoding: Open-ended Text Generation as Optimization." *arXiv preprint*. https://arxiv.org/abs/2210.15097

[5] Ranzato, M., et al. (2015). "Sequence Level Training with Recurrent Neural Networks." *International Conference on Learning Representations* (ICLR). https://arxiv.org/abs/1511.06732
   - Discusses exposure bias problem

[6] Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K., & Weston, J. (2020). "Neural Text Generation with Unlikelihood Training." *International Conference on Learning Representations* (ICLR). https://arxiv.org/abs/1908.04319
   - Addresses repetition problem

[7] Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing Systems* (NeurIPS), 33. https://arxiv.org/abs/2005.11401
   - RAG for reducing hallucinations

[8] Ouyang, L., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." *Advances in Neural Information Processing Systems* (NeurIPS), 35. https://arxiv.org/abs/2203.02155
   - RLHF methodology (ChatGPT)

[9] Kadavath, S., et al. (2022). "Language Models (Mostly) Know What They Know." *arXiv preprint*. https://arxiv.org/abs/2207.05221
   - Uncertainty quantification in LLMs

[10] Bai, Y., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." *arXiv preprint*. https://arxiv.org/abs/2212.08073
   - Anthropic's approach to alignment

[11] Fan, A., Lewis, M., & Dauphin, Y. (2018). "Hierarchical Neural Story Generation." *Proceedings of ACL*, 889-898. https://arxiv.org/abs/1805.04833
   - Early use of top-p sampling

[12] Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Technical Report*.
   - GPT-2, discusses generation strategies

[13] Keskar, N. S., et al. (2019). "CTRL: A Conditional Transformer Language Model for Controllable Generation." *arXiv preprint*. https://arxiv.org/abs/1909.05858

[14] Zhang, H., et al. (2021). "DYPLOC: Dynamic Planning of Content Using Mixed Language Models for Text Generation." *Proceedings of ACL-IJCNLP*, 6408-6423.

[15] Meister, C., et al. (2022). "Typical Decoding for Natural Language Generation." *arXiv preprint*. https://arxiv.org/abs/2202.00666
   - Alternative to top-p based on information theory

---

## Phụ Lục A: Complete Implementation

### A.1 Production-Ready Sampler

```python
"""
production_sampler.py
Production-ready token sampling implementation
"""

import torch
import torch.nn.functional as F
from typing import Optional, Literal, Dict, Any
from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """Configuration for token sampling"""
    method: Literal['greedy', 'multinomial', 'top_k', 'top_p', 'beam'] = 'top_p'
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.9
    
    # Advanced options
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # Safety
    min_length: int = 0
    max_length: int = 1024
    
    def validate(self):
        """Validate configuration"""
        assert 0.0 < self.temperature <= 2.0, "Temperature must be in (0, 2]"
        
        if self.method == 'top_k':
            assert self.top_k is not None and self.top_k > 0
        if self.method == 'top_p':
            assert self.top_p is not None and 0 < self.top_p <= 1.0
        
        assert self.repetition_penalty >= 1.0
        assert self.max_length > self.min_length


class ProductionSampler:
    """
    Production-ready token sampler with safeguards
    """
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.config.validate()
        self.generated_tokens = []
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample next token với all safeguards
        
        Args:
            logits: [batch, vocab_size]
            past_tokens: [batch, seq_len] for repetition penalty
        
        Returns:
            next_tokens: [batch]
        """
        # Apply temperature
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        
        # Apply repetition penalty
        if past_tokens is not None and self.config.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, past_tokens)
        
        # Enforce min length (suppress EOS)
        if len(self.generated_tokens) < self.config.min_length:
            logits[:, self.eos_token_id] = float('-inf')
        
        # Route to sampling method
        if self.config.method == 'greedy':
            return torch.argmax(logits, dim=-1)
        elif self.config.method == 'multinomial':
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        elif self.config.method == 'top_k':
            return self._top_k_sampling(logits)
        elif self.config.method == 'top_p':
            return self._top_p_sampling(logits)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        past_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize tokens that appear in context
        """
        penalty = self.config.repetition_penalty
        
        for batch_idx in range(logits.size(0)):
            for token_id in past_tokens[batch_idx].unique():
                # Penalize: divide if prob > 1, multiply if < 1
                if logits[batch_idx, token_id] < 0:
                    logits[batch_idx, token_id] *= penalty
                else:
                    logits[batch_idx, token_id] /= penalty
        
        return logits
    
    def _top_k_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """Top-K sampling implementation"""
        k = self.config.top_k
        
        # Get top-K
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # Mask others
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, top_k_indices, top_k_logits)
        
        # Sample
        probs = F.softmax(mask, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _top_p_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """Top-P (nucleus) sampling implementation"""
        p = self.config.top_p
        
        # Sort
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        
        # Cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumsum > p
        sorted_indices_to_remove = cumsum_probs > p
        
        # Keep at least 1 token
        sorted_indices_to_remove[..., 0] = False
        
        # Shift right (keep first token above threshold)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        
        # Create mask
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        
        # Apply mask
        logits_masked = logits.clone()
        logits_masked[mask] = float('-inf')
        
        # Sample
        probs = F.softmax(logits_masked, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


# Usage example
if __name__ == "__main__":
    # Create configuration
    config = SamplingConfig(
        method='top_p',
        temperature=0.9,
        top_p=0.92,
        repetition_penalty=1.2,
        min_length=10,
        max_length=100
    )
    
    # Initialize sampler
    sampler = ProductionSampler(config)
    
    # Simulate generation
    batch_size = 2
    vocab_size = 50000
    
    logits = torch.randn(batch_size, vocab_size)
    past_tokens = torch.randint(0, vocab_size, (batch_size, 20))
    
    next_tokens = sampler.sample(logits, past_tokens)
    print(f"Sampled tokens: {next_tokens}")
```

### A.2 Evaluation Suite

```python
"""
evaluation_suite.py
Comprehensive evaluation of sampling methods
"""

import numpy as np
from collections import Counter
from typing import List


class SamplingEvaluator:
    """Evaluate and compare sampling methods"""
    
    @staticmethod
    def diversity_metrics(texts: List[str]) -> Dict[str, float]:
        """
        Compute diversity metrics
        
        Returns:
            Dictionary với metrics
        """
        # Tokenize
        tokens_list = [text.split() for text in texts]
        all_tokens = [t for ts in tokens_list for t in ts]
        
        # Distinct-n
        distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
        
        bigrams = []
        for tokens in tokens_list:
            bigrams.extend([tuple(tokens[i:i+2]) for i in range(len(tokens)-1)])
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0
        
        # Entropy
        counts = Counter(all_tokens)
        probs = np.array(list(counts.values())) / len(all_tokens)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Length statistics
        lengths = [len(tokens) for tokens in tokens_list]
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'entropy': entropy,
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(set(all_tokens)),
        }
    
    @staticmethod
    def repetition_score(text: str, n: int = 4) -> float:
        """
        Measure repetition via repeated n-grams
        
        Returns:
            Fraction of n-grams that are repeated
        """
        tokens = text.split()
        if len(tokens) < n:
            return 0.0
        
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        counts = Counter(ngrams)
        
        repeated = sum(c > 1 for c in counts.values())
        total = len(counts)
        
        return repeated / total if total > 0 else 0.0
    
    @staticmethod
    def compare_methods(
        model,
        tokenizer,
        prompts: List[str],
        configs: Dict[str, SamplingConfig],
        n_samples: int = 20
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple sampling configurations
        
        Returns:
            Results dictionary
        """
        results = {}
        
        for name, config in configs.items():
            sampler = ProductionSampler(config)
            texts = []
            
            for prompt in prompts:
                for _ in range(n_samples):
                    text = generate_text(model, tokenizer, prompt, sampler)
                    texts.append(text)
            
            # Compute metrics
            diversity = SamplingEvaluator.diversity_metrics(texts)
            repetition = np.mean([
                SamplingEvaluator.repetition_score(t) for t in texts
            ])
            
            results[name] = {
                **diversity,
                'repetition_score': repetition,
            }
        
        return results


# Example usage
if __name__ == "__main__":
    configs = {
        'Greedy': SamplingConfig(method='greedy'),
        'Top-K(50)': SamplingConfig(method='top_k', top_k=50),
        'Top-P(0.9)': SamplingConfig(method='top_p', top_p=0.9),
        'Top-P(0.95)': SamplingConfig(method='top_p', top_p=0.95),
    }
    
    prompts = [
        "The future of AI is",
        "Once upon a time",
        "In my opinion, the best way to",
    ]
    
    # results = SamplingEvaluator.compare_methods(
    #     model, tokenizer, prompts, configs, n_samples=20
    # )
    
    # for name, metrics in results.items():
    #     print(f"\n{name}:")
    #     for metric, value in metrics.items():
    #         print(f"  {metric}: {value:.4f}")
```

---

## Phụ Lục B: Glossary

**Ancestral Sampling:** Variant of pure probabilistic sampling

**Beam Search:** Maintains multiple hypothesis sequences, selects best by score

**Decoding Strategy:** Method for selecting tokens from probability distribution

**Deterministic:** Same input always produces same output

**Diversity:** Variety in generated outputs

**Greedy Decoding:** Always selects highest-probability token

**Hallucination:** Generation of plausible-sounding but incorrect information

**Multinomial Sampling:** Pure probabilistic selection according to probabilities

**Nucleus:** Set of tokens considered for sampling trong top-p method

**Nucleus Sampling:** See Top-P Sampling

**Perplexity:** Measure of how well probability distribution predicts sample

**Repetition Penalty:** Technique to reduce repeated tokens/phrases

**Stochastic:** Involves randomness; different runs produce different outputs

**Temperature:** Parameter controlling sharpness of probability distribution

**Token:** Basic unit of text (word, subword, character)

**Top-K Sampling:** Sample randomly từ K highest-probability tokens

**Top-P Sampling:** Sample từ smallest set với cumulative probability ≥ P

**Trade-off:** Compromise between competing objectives

**Type-Token Ratio (TTR):** Ratio of unique tokens to total tokens

---

**Ghi chú kết thúc:**

Bài viết này cung cấp comprehensive analysis của token sampling methods trong text generation, từ mathematical foundations đến practical implementations, từ theoretical trade-offs đến real-world applications.

**Key message:**
> Decoding strategy choice isn't minor detail—it fundamentally shapes what LLMs can do và how they behave. Understanding methods, trade-offs, và appropriate use cases essential cho effective use của language models.

**Liên hệ và feedback:**
Academic discourse và contributions welcome. Sampling strategies continue evolving—staying current với research critical.

**Cập nhật:** 14/02/2026  
**Version:** 1.0  
**License:** Educational use với proper attribution

---

*Tài liệu này được tạo cho mục đích giáo dục và nghiên cứu trong lĩnh vực Natural Language Processing và Large Language Models.*
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
