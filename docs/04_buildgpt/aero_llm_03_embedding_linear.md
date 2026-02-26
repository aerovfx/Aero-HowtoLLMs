
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
# Phân Tích Kỹ Thuật: So Sánh `nn.Embedding` và `nn.Linear` trong PyTorch

## Tóm tắt

Bài viết này trình bày phân tích chi tiết về hai lớp PyTorch được sử dụng phổ biến trong mô hình ngôn ngữ: `nn.Embedding` và `nn.Linear`. Mặc dù hai lớp này phục vụ các mục đích khác nhau trong kiến trúc mô hình (embedding cho token representations và linear cho unembedding transformations), nghiên cứu chứng minh rằng về bản chất, chúng là các wrapper khác nhau xung quanh cùng một cấu trúc dữ liệu cơ bản. Phân tích làm rõ sự khác biệt về cú pháp, phương thức khởi tạo, và cơ chế truy cập dữ liệu giữa hai lớp này.

---

## 1. Giới Thiệu

### 1.1 Bối Cảnh

Trong kiến trúc mô hình ngôn ngữ transformer, hai phép biến đổi quan trọng được thực hiện:
- **Embedding**: Chuyển đổi token indices thành dense vectors
- **Unembedding**: Ánh xạ hidden states trở lại vocabulary space

PyTorch cung cấp hai lớp riêng biệt cho các phép toán này: `nn.Embedding` và `nn.Linear`, tạo ra sự nhầm lẫn về mối quan hệ và sự khác biệt giữa chúng.

### 1.2 Mục Tiêu Nghiên Cứu

Nghiên cứu này nhằm:
1. Làm rõ bản chất cơ bản giống nhau của hai lớp
2. Phân tích các khác biệt về implementation và usage
3. Chứng minh tính tương đương về mặt toán học
4. Cung cấp hướng dẫn thực tiễn cho việc sử dụng

---

## 2. Cơ Sở Lý Thuyết

### 2.1 Wrapper Classes trong PyTorch

**Định nghĩa:** Wrapper class là một lớp bao bọc xung quanh một object cơ bản hơn, cung cấp interface và functionality bổ sung.

**Nguyên lý cốt lõi:**
> Cả `nn.Embedding` và `nn.Linear` đều là wrapper classes xung quanh `nn.Parameter`, một tensor có khả năng tính gradient và tích hợp vào computational graph.

### 2.2 Vai Trò trong Kiến Trúc LLM

Token IDs → [nn.Embedding] → Dense Vectors → [Transformer Blocks] 
          → Hidden States → [nn.Linear] → Logits → Probabilities

**Chức năng:**
- **nn.Embedding**: Token lookup operation
- **nn.Linear**: Matrix multiplication với transpose

---

## 3. Phân Tích Kỹ Thuật Chi Tiết

### 3.1 Khai Báo và Khởi Tạo

#### 3.1.1 Định Nghĩa Cú Pháp

**nn.Embedding:**
```python

$$

$$

e = nn.Embedding(num_embeddings=5000, embedding_dim=70)

$$

$$

# Cú pháp: (vocab_size, embed_dim)
# Thứ tự: INPUT → OUTPUT

**nn.Linear:**
```python

$$

$$

l = nn.Linear(in_features=70, out_features=5000)

$$

$$

# Cú pháp: (embed_dim, vocab_size)
# Thứ tự: INPUT → OUTPUT (nhưng đảo ngược so với Embedding)

**Quan sát:**
> Thứ tự tham số bị đảo ngược giữa hai lớp, tạo ra nguồn gây nhầm lẫn lớn. Tuy nhiên, kích thước thực tế của weight matrix cơ bản là giống nhau.

#### 3.1.2 Kích Thước Weight Matrix

**Verification:**
```python
print(e.weight.shape)  # torch.Size([5000, 70])
print(l.weight.shape)  # torch.Size([5000, 70])

**Kết luận:** Cả hai đều lưu trữ ma trận có kích thước `[vocab_size, embedding_dim]`.

**Giải thích lý do đảo ngược:**

1. **nn.Embedding**: 
   - Mỗi token ID mapping đến một row trong matrix
   - Row i chứa embedding vector cho token i
   - Thứ tự `(vocab, embed)` phù hợp với semantic của "lookup table"

2. **nn.Linear**:
   - Thực hiện phép toán: `y = xW^T + b`
   - Matrix được transpose trong quá trình tính toán
   - Thứ tự `(in, out)` phù hợp với convention của linear algebra

---

### 3.2 Phân Tích Attributes và Methods

#### 3.2.1 Số Lượng Attributes

**Thống kê:**
```python
len(dir(e))  # > 100 attributes
len(dir(l))  # > 100 attributes
# Nhưng không hoàn toàn giống nhau

#### 3.2.2 Unique Attributes

**Chỉ có trong nn.Embedding:**
- `_fill_padding_idx_with_zero`
- `max_norm`
- `norm_type`
- `scale_grad_by_freq`
- `sparse`
- `padding_idx`

**Chỉ có trong nn.Linear:**
- `in_features`
- `out_features`
- `bias` (optional parameter)

**Phân tích:**
- Embedding có các features đặc biệt cho text processing (padding, sparsity)
- Linear có bias term (optional), embedding không có

---

### 3.3 Cơ Chế Truy Cập Dữ Liệu (Indexing)

#### 3.3.1 Direct Indexing với nn.Embedding

**Cách 1: Implicit indexing**
```python
# Lấy embedding vector cho token index 14

$$

$$

vector = e(torch.tensor([14]))  # Shape: [1, 70]

$$

$$

**Đặc điểm:**
- Cú pháp đơn giản, trực quan
- PyTorch tự động xử lý indexing
- Trả về embedding vector trực tiếp

#### 3.3.2 Indexing với nn.Linear

**Cách 1: Direct weight access (FAILS)**
```python

$$

$$

vector = l(14)  # TypeError: forward() missing required argument

$$

$$

**Lý do:** `nn.Linear` không support direct integer indexing.

**Cách 2: Manual weight indexing (WORKS)**
```python

$$

$$

vector = l.weight[14]  # Shape: [70]

$$

$$

**Cách 3: One-hot encoding emulation (MATHEMATICALLY EQUIVALENT)**
```python
# Tạo one-hot vector

$$

$$

one_hot = torch.zeros(5000)

$$

$$

$$
one_hot[14] = 1.0
$$

$$
# Matrix multiplication
$$

$$
vector = one_hot @ l.weight  # Shape: [70]
$$

$$
**Giải thích toán học:** Phương pháp one-hot emulation mô phỏng chính xác cách `nn.Embedding` hoạt động:
$$

$$
\mathbf{v} = \mathbf{e}_i^T \mathbf{W}
$$

$$
Trong đó: - \mathbf{e}_i = one-hot vector với 1 ở vị trí i - \mathbf{W} = weight matrix [vocab_size × embed_dim] - \mathbf{v} = embedding vector kết quả #### 3.3.3 Bảng So Sánh Phương Pháp Indexing | Phương pháp | nn.Embedding | nn.Linear | Hiệu quả | Use case | |-------------|--------------|-----------|----------|----------| | Direct call `(index)` | ✓ | ✗ | Cao | Standard embedding lookup | | `.weight[index]` | ✓ | ✓ | Cao | Manual weight access | | One-hot × weight | ✓ | ✓ | Thấp | Educational, debugging | --- ### 3.4 Phân Phối Khởi Tạo (Initialization Distribution) #### 3.4.1 Default Initialization **nn.Embedding (Normal Distribution):** ```python
$$

$$
e = nn.Embedding(5000, 70)
$$

$$
# Default: Normal(μ=0, σ=1) **Đặc điểm phân phối:** - Mean ≈ 0 - Std ≈ 1.0 - Gaussian/Normal distribution - Symmetric bell curve **nn.Linear (Uniform Distribution):** ```python
$$

$$
l = nn.Linear(70, 5000)
$$

$$
# Default: Uniform(-k, k) where k = sqrt(1/in_features) **Đặc điểm phân phối:** - Kaiming Uniform initialization - Bounds: \pm \sqrt{\frac{1}{\text{in\_features}}} - Ví dụ: với `in_features=70`, bounds ≈ ±0.119 **Quan sát thực nghiệm:** ```python import matplotlib.pyplot as plt
$$

plt.hist(e.weight.flatten().detach().numpy(), bins=50)

$$
# Hình dạng: Bell curve (Gaussian)
$$

plt.hist(l.weight.flatten().detach().numpy(), bins=50)

$$
# Hình dạng: Flat-top (Uniform) #### 3.4.2 Custom Initialization **Mục tiêu:** Làm cho `nn.Linear` có phân phối giống `nn.Embedding` **Implementation:** ```python
$$

l2 = nn.Linear(70, 5000)

$$

$$

torch.nn.init.normal_(l2.weight, mean=0.0, std=1.0)

$$
# Verification print(l2.weight.mean())  # ≈ 0.0 print(l2.weight.std())   # ≈ 1.0 **Kết quả:** - Phân phối của `l2.weight` bây giờ match với `e.weight` - Chứng minh tính linh hoạt của initialization #### 3.4.3 Kaiming Initialization Analysis **Công thức Kaiming Uniform:**
$$

$$
\text{bound} = \sqrt{\frac{6}{(1 + a^2) \times \text{fan\_in}}}
$$

$$
Với ReLU a=0:
$$

$$
\text{bound} = \sqrt{\frac{6}{\text{fan\_in}}}
$$

$$
**Expected Statistics:** ```python import math
$$

$$
k = math.sqrt(1/70)  # k ≈ 0.1195
$$

$$
# Uniform distribution [-k, k] # Expected mean: 0 # Expected std: k/sqrt(3) ≈ 0.069 **Empirical verification:** ```python print(f"Theoretical std: {k/math.sqrt(3):.4f}") print(f"Actual std: {l.weight.std():.4f}") # Output shows close match --- ## 4. Phân Tích Source Code ### 4.1 Underlying Implementation #### 4.1.1 nn.Embedding Source Code **Trích xuất từ PyTorch source:** ```python class Embedding(Module): def __init__(self, num_embeddings, embedding_dim, ...): super(Embedding, self).__init__()
$$

$$
self.num_embeddings = num_embeddings
$$

$$

$$

self.embedding_dim = embedding_dim

$$

$$

        
        # KEY LINE: Creates weight matrix

$$

$$

self.weight = Parameter(

$$

$$

            torch.empty((num_embeddings, embedding_dim))
        )
        self.reset_parameters()

#### 4.1.2 nn.Linear Source Code

**Trích xuất từ PyTorch source:**
```python
class Linear(Module):

$$

$$

def __init__(self, in_features, out_features, bias=True):

$$

$$

        super(Linear, self).__init__()

$$

$$

self.in_features = in_features

$$

$$

$$
self.out_features = out_features
$$

$$
# KEY LINE: Creates weight matrix
$$

$$
self.weight = Parameter(
$$

$$
torch.empty((out_features, in_features)) ) if bias:
$$

$$
self.bias = Parameter(torch.empty(out_features))
$$

$$
else: self.register_parameter('bias', None) #### 4.1.3 Common Core: nn.Parameter **Cả hai đều sử dụng:** ```python nn.Parameter(tensor) **Định nghĩa `nn.Parameter`:** - Subclass của `torch.Tensor` - Automatically registered as model parameter - Included in `model.parameters()` - Tracked for gradient computation - Part of computational graph **Kết luận quan trọng:** > Sự khác biệt giữa `nn.Embedding` và `nn.Linear` chỉ là "syntactic sugar" - chúng đều dựa trên cùng một underlying mechanism (`nn.Parameter`) và chỉ khác nhau về interface và một số features đặc biệt. --- ## 5. Computational Equivalence ### 5.1 Forward Pass Comparison #### 5.1.1 nn.Embedding Forward **Pseudocode:** ```python def embedding_forward(input_ids, weight): # input_ids: [batch_size] hoặc [batch_size, seq_len] # weight: [vocab_size, embed_dim]
$$

$$
output = weight[input_ids]  # Advanced indexing
$$

$$
return output **Ví dụ:** ```python
$$

$$
input_ids = torch.tensor([14, 27, 103])  # 3 tokens
$$

$$

$$

output = e(input_ids)  # Shape: [3, 70]

$$

$$

# Equivalent to: weight[[14, 27, 103], :]

#### 5.1.2 nn.Linear Forward (for unembedding)

**Pseudocode:**
```python

$$

$$

def linear_forward(input, weight, bias=None):

$$

$$

    # input: [batch_size, in_features]
    # weight: [out_features, in_features]
    # output: [batch_size, out_features]
    
$$

$$

output = input @ weight.T  # Matrix multiply with transpose

$$

$$

    if bias is not None:

$$
output += bias
$$

    return output

**Ví dụ:**
```python

$$

$$

hidden = torch.randn(3, 70)  # 3 samples, 70 dims

$$

$$

$$
logits = l(hidden)  # Shape: [3, 5000]
$$

$$
# Equivalent to: hidden @ l.weight.T + l.bias ### 5.2 Mathematical Operations #### 5.2.1 Embedding as Matrix Multiplication **Embedding operation có thể được viết lại:** ```python # Standard embedding
$$

$$
output = e(torch.tensor([14]))
$$

$$
# Equivalent one-hot multiplication
$$

$$
one_hot = F.one_hot(torch.tensor([14]), num_classes=5000).float()
$$

$$

$$

output_equiv = one_hot @ e.weight

$$

$$

assert torch.allclose(output, output_equiv)

**Complexity analysis:**
- Direct indexing: O(1) lookup
- One-hot multiplication: O(vocab_size × embed_dim)
- **Direct indexing là tối ưu hơn rất nhiều**

#### 5.2.2 Linear as Reverse Embedding

**Conceptually:**
Embedding:    Token ID → Dense Vector
             [discrete] → [continuous]

Linear:       Dense Vector → Logits over Vocab
             [continuous] → [discrete probabilities]

**Trong unembedding context:**
- Input: Hidden state [embed_dim]
- Weight: [vocab_size, embed_dim]
- Output: Logits [vocab_size]
- Operation: Dot product với mỗi vocab entry

---

## 6. Practical Implications

### 6.1 Memory Efficiency

**Cả hai lớp:**
- Store same-sized weight matrix: `vocab_size × embed_dim`
- Memory footprint: Identical
- Example: 50k vocab × 768 dim × 4 bytes = ~153 MB

### 6.2 Computational Efficiency

**nn.Embedding:**
```python
# Batch lookup: Very efficient

$$

$$

input_ids = torch.randint(0, 5000, (32, 512))  # [batch, seq]

$$

$$

$$
output = e(input_ids)  # [32, 512, 70]
$$

$$
# Operation: Simple indexing, O(batch × seq) **nn.Linear:** ```python # Batch matrix multiplication
$$

$$
hidden = torch.randn(32, 512, 70)  # [batch, seq, hidden]
$$

$$

$$

logits = l(hidden)  # [32, 512, 5000]

$$

$$

# Operation: GEMM, O(batch × seq × hidden × vocab)

**Performance consideration:**
- Embedding lookup: Extremely fast
- Linear transformation: Depends on matrix sizes
- Bottleneck in LLMs: Usually the unembedding step

### 6.3 Gradient Flow

**Cả hai support:**
- Automatic differentiation
- Backpropagation through weights
- Gradient accumulation
- Optimizer updates

**Embedding-specific:**
```python
# Sparse gradients: Only update accessed embeddings

$$

$$

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

$$

$$

# Only rows corresponding to input tokens get gradient updates

**Linear:**
```python
# Dense gradients: All weights potentially updated
# Full matrix receives gradients every forward pass

---

## 7. Best Practices và Recommendations

### 7.1 Khi Nào Dùng nn.Embedding

**Use cases:**
1. Token-to-vector mapping (standard embedding layer)
2. Lookup tables cho discrete entities
3. Khi cần padding_idx functionality
4. Khi cần sparse gradients
5. Positional embeddings

**Ví dụ:**
```python

$$

$$

token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

$$

$$

$$
position_embedding = nn.Embedding(max_seq_len, hidden_dim)
$$

$$
### 7.2 Khi Nào Dùng nn.Linear **Use cases:** 1. Unembedding layer (hidden → logits) 2. Feed-forward layers trong transformer 3. Projection layers (Q, K, V trong attention) 4. Khi cần bias term 5. Bất kỳ dense transformation nào **Ví dụ:** ```python
$$

$$
unembedding = nn.Linear(hidden_dim, vocab_size, bias=False)
$$

$$

$$

ffn = nn.Linear(hidden_dim, ffn_dim)

$$

$$

$$
query_proj = nn.Linear(hidden_dim, head_dim)
$$

$$
### 7.3 Weight Tying (Liên kết Trọng Số) **Advanced technique:** ```python class TransformerLM(nn.Module): def __init__(self, vocab_size, hidden_dim): super().__init__()
$$

$$
self.embedding = nn.Embedding(vocab_size, hidden_dim)
$$

$$

$$

self.unembedding = nn.Linear(hidden_dim, vocab_size, bias=False)

$$

$$

        
        # WEIGHT TYING: Share weights

$$

$$

self.unembedding.weight = self.embedding.weight

$$

$$

**Lợi ích:**
- Giảm 50% số parameters
- Regularization effect
- Embedding và unembedding học cùng representation
- Được sử dụng trong nhiều LLMs (GPT-2, BERT variants)

**Lưu ý:** Weight tying chứng minh rõ ràng rằng embedding và linear weights là interchangeable.

---

## 8. Common Pitfalls và Troubleshooting

### 8.1 Dimension Ordering Confusion

**Problem:**
```python
# WRONG

$$

$$

e = nn.Embedding(70, 5000)  # Swapped!

$$

$$

$$
l = nn.Linear(5000, 70)      # Swapped!
$$

$$
# CORRECT
$$

$$
e = nn.Embedding(5000, 70)   # (vocab, embed)
$$

$$

$$

l = nn.Linear(70, 5000)      # (in, out)

$$

$$

**Solution:** Always double-check parameter order and verify với `.weight.shape`.

### 8.2 Indexing Errors

**Problem:**
```python

$$

$$

l = nn.Linear(70, 5000)

$$

$$

$$
vector = l(14)  # TypeError!
$$

$$
**Solution:** ```python # Option 1: Direct weight access
$$

$$
vector = l.weight[14]
$$

$$
# Option 2: Use as intended
$$

$$
hidden = torch.randn(1, 70)
$$

$$

$$

logits = l(hidden)

$$

$$

### 8.3 Initialization Mismatch

**Problem:**
```python

$$

$$

e = nn.Embedding(5000, 70)  # Normal distribution

$$

$$

$$
l = nn.Linear(70, 5000)     # Uniform distribution
$$

$$
# Different initializations may cause training issues **Solution:** ```python # Standardize initialization
$$

torch.nn.init.normal_(l.weight, mean=0.0, std=1.0)

$$
# Or use Xavier/Kaiming consistently --- ## 9. Kết Luận ### 9.1 Tóm Tắt Findings **Điểm giống nhau:** 1. Cả hai đều wrapper xung quanh `nn.Parameter` 2. Weight matrix có cùng kích thước 3. Có thể emulate functionality của nhau 4. Support automatic differentiation 5. Part of computational graph **Điểm khác nhau:** | Aspect | nn.Embedding | nn.Linear | |--------|--------------|-----------| | Parameter order | (vocab, embed) | (in, out) | | Direct indexing | ✓ | ✗ | | Bias term | ✗ | ✓ (optional) | | Default init | Normal(0,1) | Kaiming Uniform | | Primary use | Token lookup | Matrix transformation | | Sparse gradients | ✓ (possible) | ✗ | | Padding support | ✓ | ✗ | ### 9.2 Practical Wisdom **Key Insight:** > Sự khác biệt giữa `nn.Embedding` và `nn.Linear` không phải về toán học hay kiến trúc cơ bản, mà về **convenience và optimization cho specific use cases**. Hiểu rõ điều này giúp developer sử dụng đúng tool cho đúng task. ### 9.3 Educational Value Việc phân tích sâu về hai lớp này minh họa một nguyên lý quan trọng trong deep learning frameworks: **"Under the hood simplicity with surface-level convenience"** PyTorch (và các frameworks khác) cung cấp multiple interfaces cho cùng một underlying operation, optimized cho different contexts và use patterns. --- ## 10. Directions for Further Study ### 10.1 Advanced Topics 1. **Sparse embeddings**: Hiệu quả memory với large vocabularies 2. **Quantization**: Giảm precision để tăng speed 3. **Custom initialization schemes**: Xavier, He, orthogonal 4. **Gradient clipping**: Stabilize training với embeddings 5. **Embedding regularization**: L2 penalty, dropout ### 10.2 Related Concepts - **Weight tying** trong language models - **Positional embeddings** (learned vs sinusoidal) - **Subword tokenization** impact lên embedding size - **Low-rank factorization** of embedding matrices - **Contextual embeddings** (BERT-style) vs static --- ## Tài Liệu Tham Khảo 1. Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS. 2. He, K., et al. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." ICCV. 3. Glorot, X., & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." AISTATS. 4. Press, O., & Wolf, L. (2017). "Using the Output Embedding to Improve Language Models." EACL. --- ## Phụ Lục: Complete Code Examples ### A.1 Basic Setup ```python import torch import torch.nn as nn
$$

$$
vocab_size = 5000
$$

$$

$$

embed_dim = 70

$$

$$

$$
e = nn.Embedding(vocab_size, embed_dim)
$$

$$

$$

l = nn.Linear(embed_dim, vocab_size)

$$

$$

### A.2 Equivalence Demonstration
```python
# Method 1: Embedding

$$
idx = 14
$$

$$
emb_output = e(torch.tensor([idx]))
$$

$$
# Method 2: Linear with one-hot
$$

$$
one_hot = torch.zeros(vocab_size)
$$

$$

$$

one_hot[idx] = 1.0

$$

$$

$$
lin_output = one_hot @ l.weight
$$

$$
# Method 3: Direct indexing
$$

$$
direct = l.weight[idx]
$$

$$
# All should be equivalent (except shape) ### A.3 Weight Tying Example ```python class TiedModel(nn.Module): def __init__(self, vocab_size, embed_dim): super().__init__()
$$

$$
self.embed = nn.Embedding(vocab_size, embed_dim)
$$

$$

$$

self.linear = nn.Linear(embed_dim, vocab_size, bias=False)

$$

$$

$$
self.linear.weight = self.embed.weight  # Tie weights
$$

$$
def forward(self, input_ids):
$$

$$
embedded = self.embed(input_ids)
$$

$$

$$

logits = self.linear(embedded)

$$

$$

        return logits

---

**Từ khóa:** PyTorch, nn.Embedding, nn.Linear, Weight Matrix, Token Embedding, Unembedding, Parameter Initialization, Kaiming Initialization, Weight Tying, Computational Graph, Automatic Differentiation, Deep Learning Framework
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Mở rộng Kiến trúc GPT: Position Embedding, Layer Normalization, Weight Tying và Temperature Scaling](aero_llm_010_posion_embedding.md) | [Xem bài viết →](aero_llm_010_posion_embedding.md) |
| [Biểu diễn Tính Nhân Quả Thời Gian trong Cơ Chế Attention bằng Đại Số Tuyến Tính](aero_llm_011_temporal_causality_via_linear_algebra_theory_.md) | [Xem bài viết →](aero_llm_011_temporal_causality_via_linear_algebra_theory_.md) |
| [Cơ Chế Trung Bình Hóa Quá Khứ và Loại Bỏ Tương Lai trong Mô Hình Ngôn Ngữ Nhân Quả](aero_llm_012_averaging_the_past_while_ignoring_the_future.md) | [Xem bài viết →](aero_llm_012_averaging_the_past_while_ignoring_the_future.md) |
| [Thuật Toán Attention trong Mô Hình Transformer: Cơ Sở Lý Thuyết, Cơ Chế Hoạt Động và Hàm Ý Ứng Dụng](aero_llm_013_the_attention_algorithm_theory_.md) | [Xem bài viết →](aero_llm_013_the_attention_algorithm_theory_.md) |
| [Phân Tích và Triển Khai Cơ Chế Attention: So Sánh Cài Đặt Thủ Công và PyTorch Tối Ưu](aero_llm_014_codechallenge_code_attention.md) | [Xem bài viết →](aero_llm_014_codechallenge_code_attention.md) |
| [Phân Tích Kiến Trúc Mô Hình Ngôn Ngữ với Một Attention Head: Lý Thuyết, Triển Khai và Đánh Giá](aero_llm_015_model.md) | [Xem bài viết →](aero_llm_015_model.md) |
| [Phân Tích Cấu Trúc Transformer Block: Lý Thuyết, Cơ Chế Biểu Diễn và Vai Trò Trong Mô Hình Ngôn Ngữ](aero_llm_016_the_transformer_block_theory_.md) | [Xem bài viết →](aero_llm_016_the_transformer_block_theory_.md) |
| [Cài Đặt Transformer Block Bằng PyTorch: Phân Tích Kiến Trúc, Luồng Dữ Liệu và Tối Ưu Hóa](aero_llm_017_the_transformer_block_code_.md) | [Xem bài viết →](aero_llm_017_the_transformer_block_code_.md) |
| [Mô Hình Nhiều Transformer Blocks Trong Mạng Ngôn Ngữ: Kiến Trúc, Phân Cấp Biểu Diễn và Khả Năng Mở Rộng](aero_llm_018_model_4_multiple_transformer_blocks_.md) | [Xem bài viết →](aero_llm_018_model_4_multiple_transformer_blocks_.md) |
| [aero llm 019 copy 10](aero_llm_019_copy_10.md) | [Xem bài viết →](aero_llm_019_copy_10.md) |
| [aero llm 019 copy 11](aero_llm_019_copy_11.md) | [Xem bài viết →](aero_llm_019_copy_11.md) |
| [aero llm 019 copy 12](aero_llm_019_copy_12.md) | [Xem bài viết →](aero_llm_019_copy_12.md) |
| [aero llm 019 copy 13](aero_llm_019_copy_13.md) | [Xem bài viết →](aero_llm_019_copy_13.md) |
| [aero llm 019 copy 9](aero_llm_019_copy_9.md) | [Xem bài viết →](aero_llm_019_copy_9.md) |
| [Multi-Head Attention: Cơ Sở Lý Thuyết và Triển Khai Thực Tiễn](aero_llm_019_multihead_attention_theory_and_implementation.md) | [Xem bài viết →](aero_llm_019_multihead_attention_theory_and_implementation.md) |
| [aero llm 01 intro](aero_llm_01_intro.md) | [Xem bài viết →](aero_llm_01_intro.md) |
| [Tối Ưu Hóa Huấn Luyện Mô Hình Học Sâu Bằng GPU: Nguyên Lý và Thực Hành](aero_llm_020_working_on_the_gpu.md) | [Xem bài viết →](aero_llm_020_working_on_the_gpu.md) |
| [Triển Khai Mô Hình GPT-2 Hoàn Chỉnh Trên GPU: Kiến Trúc, Tối Ưu Hóa và Đánh Giá Hiệu Năng](aero_llm_021_mo_hinh_gpt_2_hoan_chinh_tren_gpu.md) | [Xem bài viết →](aero_llm_021_mo_hinh_gpt_2_hoan_chinh_tren_gpu.md) |
| [Đánh Giá Hiệu Năng GPT-2 Trên CPU và GPU: Thực Nghiệm Thời Gian Khởi Tạo, Suy Luận và Huấn Luyện](aero_llm_022_anh_gia_hieu_nang_gpt_2_tren_cpu_va_gpu.md) | [Xem bài viết →](aero_llm_022_anh_gia_hieu_nang_gpt_2_tren_cpu_va_gpu.md) |
| [Khảo Sát Mô Hình GPT-2 Tiền Huấn Luyện của OpenAI: Kiến Trúc, Tham Số và Cơ Chế Sinh Văn Bản](aero_llm_023_inspecting_openai_s_gpt2.md) | [Xem bài viết →](aero_llm_023_inspecting_openai_s_gpt2.md) |
| [Kiến Trúc Transformer và Triển Khai GPT-2 trên GPU: Phân Tích Toán Học và Hiệu Năng Tính Toán](aero_llm_024_summarizing_gpt_using_equations.md) | [Xem bài viết →](aero_llm_024_summarizing_gpt_using_equations.md) |
| [Trực Quan Hóa Kiến Trúc GPT Thông Qua nano-GPT: Tiếp Cận Trực Quan trong Nghiên Cứu Mô Hình Ngôn Ngữ](aero_llm_025_visualizing_nano_gpt.md) | [Xem bài viết →](aero_llm_025_visualizing_nano_gpt.md) |
| [Phân Tích Số Lượng Tham Số Trong Mô Hình GPT-2: Phương Pháp Định Lượng và Ý Nghĩa Kiến Trúc](aero_llm_026_codechallenge_how_many_parameters_part_1_.md) | [Xem bài viết →](aero_llm_026_codechallenge_how_many_parameters_part_1_.md) |
| [Phân Bố Tham Số Trong GPT-2: So Sánh Attention, MLP và Layer Normalization](aero_llm_027_codechallenge_how_many_parameters_part_2_.md) | [Xem bài viết →](aero_llm_027_codechallenge_how_many_parameters_part_2_.md) |
| [📘 Phân Tích Kiến Trúc GPT-2: Từ Cơ Chế Multi-Head Attention Đến Hiệu Năng Tính Toán Trên GPU](aero_llm_028_codechallenge_gpt2_trained_weights_distributions.md) | [Xem bài viết →](aero_llm_028_codechallenge_gpt2_trained_weights_distributions.md) |
| [🧠 Phân Tích Nhân Quả Trong GPT-2: Vai Trò Của Ma Trận Query Thông Qua Can Thiệp Tham Số](aero_llm_029_codechallenge_do_we_really_need_q.md) | [Xem bài viết →](aero_llm_029_codechallenge_do_we_really_need_q.md) |
| [Phân Tích Kiến Trúc và Cơ Chế Hoạt Động của Mô Hình Ngôn Ngữ Transformer Cơ Bản](aero_llm_02_transformer.md) | [Xem bài viết →](aero_llm_02_transformer.md) |
| 📌 **[Phân Tích Kỹ Thuật: So Sánh `nn.Embedding` và `nn.Linear` trong PyTorch](aero_llm_03_embedding_linear.md)** | [Xem bài viết →](aero_llm_03_embedding_linear.md) |
| [Phân Tích So Sánh Hàm Kích Hoạt GELU và ReLU trong Mô Hình Ngôn Ngữ Lớn: Góc Nhìn Lý Thuyết và Thực Nghiệm](aero_llm_04_gelu_vs_relu_academic_analysis.md) | [Xem bài viết →](aero_llm_04_gelu_vs_relu_academic_analysis.md) |
| [Hàm Softmax và Tham Số Temperature trong Mô Hình Ngôn Ngữ Lớn: Phân Tích Toán Học và Thực Nghiệm](aero_llm_05_softmax_temperature_academic_analysis.md) | [Xem bài viết →](aero_llm_05_softmax_temperature_academic_analysis.md) |
| [Phân Tích `torch.multinomial`: Lấy Mẫu Xác Suất trong Sinh Văn Bản với PyTorch](aero_llm_06_torch_multinomial_academic_analysis.md) | [Xem bài viết →](aero_llm_06_torch_multinomial_academic_analysis.md) |
| [Phương Pháp Lấy Mẫu Token trong Sinh Văn Bản: Phân Tích So Sánh Greedy, Top-K, Top-P và Multinomial Sampling](aero_llm_07_token_sampling_methods.md) | [Xem bài viết →](aero_llm_07_token_sampling_methods.md) |
| [Phân Tích Hành Vi Của Hàm Softmax Trong Mô Hình Học Sâu: Ảnh Hưởng Của Lặp, Phạm Vi Số Học Và Nhiệt Độ](aero_llm_08_ham_softbank.md) | [Xem bài viết →](aero_llm_08_ham_softbank.md) |
| [Phân Tích Layer Normalization Trong Học Sâu: Cơ Sở Lý Thuyết, Ổn Định Số Học Và Ứng Dụng Thực Tiễn](aero_llm_09_layer_normalization.md) | [Xem bài viết →](aero_llm_09_layer_normalization.md) |
| [kien truc mo hinh ngon ngu lon](kien_truc_mo_hinh_ngon_ngu_lon.md) | [Xem bài viết →](kien_truc_mo_hinh_ngon_ngu_lon.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
