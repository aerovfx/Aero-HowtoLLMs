
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [01 llm course](../index.md) > [lecturestanford](index.md)

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
# Chương 4 & 5: Mechanisms & Evaluation 🔧📊

> **CS229 Stanford** | Chương 4-5/5  
> **Autoregressive, Tokenization & Evaluation**

---

## CHƯƠNG 4: Cơ Chế Hoạt Động

### **1. Autoregressive Generation**

**Formula:**

$$
P(x₁, x₂, ..., xₙ) = ∏ᵢ P(xᵢ  \mid  x₁, ..., xᵢ₋₁)
$$

**Example:**
Input: "The cat"

$$
Step 1: P("sat"  \mid  "The cat") = 0.8 → Output: "sat" Step 2: P("on"  \mid  "The cat sat") = 0.9 → Output: "on" Step 3: P("the"  \mid  "The cat sat on") = 0.95 → Output: "the"
$$

...

Result: "The cat sat on the mat"

**Hạn chế:**
- Slow (sequential, not parallel)
- Can't "look ahead"
- Expensive at scale

---

### **2. Tokenization**

**Why?** Computers understand numbers, not words.

**BPE Algorithm:**
1. Start: ['h', 'e', 'l', 'l', 'o'] 
2. Merge frequent: 'l'+'l' → 'll'
3. Result: ['h', 'e', 'll', 'o']
4. Repeat...

**GPT-4 Tokenizer:**
- Vocab size: ~100K tokens
- Average: 1 token ≈ 0.75 words
- Handles 100+ languages

**Common Issues:**
```python
# Numbers get split weirdly
"327" → ["3", "27"] ❌  # Bad for math

# Indentation problems (old models)
"    def foo():" → ["  ", "  ", "def", " foo", "():"]

# Non-English struggles
"你好" (Chinese) → Multiple f18_ragments

---

## CHƯƠNG 5: Đánh Giá

### **1. Perplexity**

PPL = exp(-1/N ∑ log P(xᵢ  \mid  context))

$$
**Lower = Better** | Model | PPL (WikiText) | |-------|----------------| | LSTM (2017) | ~70 | | GPT-2 (2019) | ~18 | | **GPT-4 (2023)** | **~8** | --- ### **2. Benchmarks** #### **MMLU (Knowledge)** - 57 subjects - Multiple choice - GPT-4: **86.4%** (human expert ~90%) #### **HumanEval (Coding)** - 164 Python problems - GPT-4: **67.0% pass@1** #### **GSM8K (Math)** - Grade school math - GPT-4: **92.0%** --- ### **3. Human Evaluation** **Criteria:** 1. **Helpful:** Did it answer well? 2. **Honest:** No hallucinations? 3. **Harmless:** No toxic content? **Process:** Generate responses → Humans rate → Statistical analysis --- ### **4. Production Metrics** | Metric | Target | |--------|--------| | Latency | < 500ms | | Throughput | > 100 tok/s | | Cost | < 0.01/1K tokens | | Uptime | > 99.9% | --- ## 🎯 Summary: Full LLM Stack ┌─────────────────────────────┐ │  User Prompt                │ └──────────┬──────────────────┘ ↓
$$

Tokenization

$$
↓ ┌─────────────────────────────┐ │  Embedding Layer            │ │  Position + Token           │ └──────────┬──────────────────┘ ↓ ┌─────────────────────────────┐ │  Transformer Blocks × 120   │ │  ├─ Layer Norm              │ │  ├─ Self-Attention          │ │  ├─ MoE (8 experts, Top-2)  │ │  └─ Residual                │ └──────────┬──────────────────┘ ↓
$$

Final LN + Linear

$$
↓
$$

Softmax → Probs

$$
↓
$$

Sample Next Token

$$
↓
$$

Autoregressive Loop