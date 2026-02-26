
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../index.md) > [01 LLM Course](../../index.md) > [LectureStanford](../index.md)

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
# Chương 2: 5 Trụ Cột Của Việc Huấn Luyện LLMs 🏛️

> **Khóa học:** CS229 - Machine Learning (Stanford)  
> **Chương:** 2/5 - Training Foundations  
> **Mục tiêu:** Hiểu 5 yếu tố cốt lõi cần thiết để xây dựng LLM thành công

---

## 📚 Nội Dung Chương

1. [Tổng Quan 5 Trụ Cột](#tổng-quan-5-trụ-cột)
2. [Trụ Cột 1: Architecture](#trụ-cột-1-architecture-kiến-trúc)
3. [Trụ Cột 2: Training Loss & Algorithm](#trụ-cột-2-training-loss--algorithm)
4. [Trụ Cột 3: Data](#trụ-cột-3-data-dữ-liệu)
5. [Trụ Cột 4: Evaluation](#trụ-cột-4-evaluation-đánh-giá)
6. [Trụ Cột 5: Systems](#trụ-cột-5-systems-hệ-thống)
7. [Academia vs Industry](#academia-vs-industry)

---

## Tổng Quan 5 Trụ Cột

### **Mô Hình SUCCESS = f(Architecture, Loss, Data, Evaluation, Systems)**

```
        🏗️ Architecture
             ↓
        📉 Training Loss
             ↓
        📊 Data ←──────→ 📈 Evaluation
             ↓                ↑
        ⚙️ Systems ──────────┘
             ↓
        🎯 Production LLM
```

### **Ví Dụ Thực Tế: GPT-4**

| Trụ Cột | GPT-4 Implementation |
|---------|---------------------|
| **Architecture** | MoE Transformer (8 experts, Top-2) |
| **Loss** | Cross-entropy + PPO (RLHF) |
| **Data** | ~13T tokens (web, books, code) |
| **Evaluation** | MMLU, HumanEval, custom benchmarks |
| **Systems** | 10,000+ A100 GPUs, 100+ days |

---

## Trụ Cột 1: Architecture (Kiến Trúc)

### **Định Nghĩa**

**Architecture** = Thiết kế mạng neural, cấu trúc tính toán từ input → output.

### **Evolution of LLM Architectures**

```
2017: Transformer (Original)
  ↓
2018: GPT-1 (Decoder-only)
  ↓
2018: BERT (Encoder-only)
  ↓
2019: GPT-2 (Scaled decoder)
  ↓
2020: GPT-3 (Dense transformer, 175B)
  ↓
2021: Switch Transformer (MoE, 1.6T)
  ↓
2023: GPT-4 (MoE + Multimodal)
  ↓
2024: Gemini Ultra (Unified multimodal)
```

### **Key Architectural Components**

#### **A. Attention Mechanisms**

**1. Multi-Head Attention:**
```python
# Pseudo-code
def multi_head_attention(x, num_heads=8):
    # Split into multiple heads
    Q, K, V = split_heads(x, num_heads)
    
    # Scaled dot-product attention
    scores = (Q @ K.T) / sqrt(d_k)
    attn = softmax(scores)
    output = attn @ V
    
    # Concat and project
    return concat_heads(output)
```

**2. Grouped Query Attention (GQA):**
- Used in Llama 2
- Fewer K, V heads than Q heads
- Faster inference

**3. Multi-Query Attention (MQA):**
- Single K, V for all Q heads
- Maximum speed

#### **B. Position Encodings**

| Type | Formula | Used In |
|------|---------|---------|
| **Absolute** | sin/cos | Original Transformer |
| **Relative** | Learnable | T5 |
| **RoPE** | Rotary | Llama, GPT-NeoX |
| **ALiBi** | Attention bias | BLOOM |

**RoPE (Rotary Position Embedding):**
```python
def rope(x, positions):
    # Rotate pairs of dimensions
    freqs = 1.0 / (10000 ** (arange(0, d, 2) / d))
    angles = positions[:, None] * freqs[None, :]
    
    # Apply rotation
    cos, sin = cos(angles), sin(angles)
    x_rotated = rotate_half(x)
    return x * cos + x_rotated * sin
```

#### **C. Mixture of Experts (MoE)**

**Architecture:**
```
Input
  ↓
Gate/Router ──→ Gating scores [s₀, s₁, ..., s₇]
  ↓
Top-K (k=2) ──→ Select 2 highest scores
  ↓
┌────────┬────────┬────────┬────────┐
│ Expert0│ Expert1│ Expert2│ Expert3│  ← Only 2 are active
│ Expert4│ Expert5│ Expert6│ Expert7│
└────────┴────────┴────────┴────────┘
  ↓
Weighted sum = w₀·E₀(x) + w₁·E₁(x)
  ↓
Output
```

**Benefits:**
- ✅ Efficient: Only ~12.5% params active (2/16 experts)
- ✅ Specialized: Each expert learns different patterns
- ✅ Scalable: Easy to add more experts

** Challenges:**
- ❌ Load balancing: Some experts underutilized
- ❌ Training complexity: Needs auxiliary loss
- ❌ Serving: Higher memory requirements

### **GPT-4 Architecture Deep Dive**

**Specs (estimated):**
```python
{
    "model_type": "MoE Transformer",
    "num_layers": 120,
    "hidden_size": 18432,
    "num_attention_heads": 128,
    "head_dim": 144,
    "num_experts": 8,
    "experts_active": 2,
    "vocab_size": 100277,
    "context_length": 32768,  # up to 128K
    "total_params": "1.76T",
    "active_params": "~220B per token"
}
```

**Visualization trong llm_viz:**
- Expert grid: 2×4 layout
- Router visualization
- Color coding (active=green, inactive=gray)
- Top-K selection animation

---

## Trụ Cột 2: Training Loss & Algorithm

### **Training Loss**

**Primary: Cross-Entropy Loss**

```python
def cross_entropy_loss(logits, targets):
    """
    logits: [batch, seq_len, vocab_size]
    targets: [batch, seq_len]
    """
    # Softmax to get probabilities
    probs = softmax(logits, dim=-1)
    
    # Negative log likelihood
    loss = -log(probs[range(len(targets)), targets])
    
    return loss.mean()
```

**Formula:**
```
L = -∑ᵢ log P(xᵢ | x₁, ..., xᵢ₋₁)
```

**Objective:** Maximize likelihood of correct next token

### **Optimization Algorithms**

#### **A. Adam (GPT-2, GPT-3)**

```python
# Adam parameters
lr = 6e-4  # learning rate
beta1 = 0.9
beta2 = 0.95
epsilon = 1e-8
weight_decay = 0.1

# Update rule
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad**2
update = lr * m / (sqrt(v) + epsilon)
params -= update
```

#### **B. AdamW (Modern LLMs)**

- Decoupled weight decay
- Better generalization
- **Used in:** GPT-4, Llama, Gemini

#### **C. Adafactor (T5, PaLM)**

- Memory-efficient
- Factorized second moments
- Good for huge models

### **Learning Rate Schedule**

**Cosine Decay with Warmup:**
```
Warmup (0-2000 steps):
  lr = base_lr * (step / warmup_steps)

Cosine Decay:
  lr = min_lr + 0.5 * (max_lr - min_lr) * 
       (1 + cos(π * (step - warmup) / total_steps))
```

**GPT-3 Schedule:**
- Warmup: 375M tokens
- Peak LR: 6e-4
- Decay to: 6e-5
- Total: 300B tokens

### **Gradient Clipping**

```python
# Prevent gradient explosion
max_grad_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

### **Mixed Precision Training**

**BF16 (Brain Float 16):**
```python
from torch.cuda.amp import autocast

with autocast(dtype=torch.bfloat16):
    logits = model(inputs)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- 2× faster training
- 2× less memory
- No loss in accuracy

---

## Trụ Cột 3: Data (Dữ Liệu)

### **"Data is the New Oil" 🛢️**

**Tầm quan trọng:**
> "Having 10× more data is often better than having a 10× better algorithm."  
> — Andrew Ng

### **Data Sources for LLMs**

| Source | Size | Quality | Examples |
|--------|------|---------|----------|
| **Web Crawl** | ~50T tokens | Low-Medium | Common Crawl |
| **Books** | ~100B tokens | High | Books3, Gutenberg |
| **Code** | ~500B tokens | High | GitHub, StackOverflow |
| **Wikipedia** | ~6B tokens | Very High | Wikipedia dumps |
| **Papers** | ~200B tokens | Very High | arXiv, PubMed |
| **Conversations** | ~10B tokens | Variable | Reddit, forums |

**GPT-3 Training Data:**
```
Common Crawl (filtered): 410B tokens (60%)
WebText2: 19B tokens (22%)
Books1: 12B tokens (8%)
Books2: 55B tokens (8%)
Wikipedia: 3B tokens (3%)
```

### **Data Preprocessing Pipeline**

```
Raw Data
  ↓
1. Deduplication
  ├── Exact match removal
  ├── Near-duplicate detection (MinHash)
  └── URL deduplication
  ↓
2. Quality Filtering
  ├── Language detection
  ├── Perplexity filtering
  ├── Toxicity filtering
  └── PII removal
  ↓
3. Balancing
  ├── Domain distribution
  ├── Language distribution
  └── Temporal distribution
  ↓
4. Tokenization
  └── BPE/SentencePiece
  ↓
Clean Training Data
```

### **Data Quality Metrics**

**Perplexity-based filtering:**
```python
# Train small model on high-quality data
ref_model = train_tiny_gpt(wikipedia + books)

# Filter web data
for doc in web_crawl:
    perplexity = ref_model.perplexity(doc)
    if perplexity < threshold:  # e.g., 1000
        keep(doc)
```

### **Synthetic Data**

**Use cases:**
1. **Math:** Generate problems + solutions
2. **Code:** Create coding challenges
3. **Reasoning:** Chain-of-thought examples

**Example (GPT-4):**
```python
# Generate math problems
prompt = "Generate 100 algebra word problems with step-by-step solutions"
synthetic_data = gpt4.generate(prompt)

# Filter for quality
high_quality = filter_by_correctness(synthetic_data)
```

### **Data Privacy & Ethics**

**Challenge:**
- Personal information in training data
- Copyright issues (books, code)
- Bias amplification

**Solutions:**
- PII removal
- Licensing compliance
- Bias audits
- Opt-out mechanisms

---

## (Continued in next message due to length...)
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
