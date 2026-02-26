
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
Dưới đây là **bài viết khoa học bằng tiếng Việt**, trình bày theo định dạng **Markdown (MD)**, dựa trên nội dung tài liệu *“Code Attention Manually and in PyTorch”*, có phân tích học thuật và trích dẫn nguồn phù hợp.

---

# Phân Tích và Triển Khai Cơ Chế Attention: So Sánh Cài Đặt Thủ Công và PyTorch Tối Ưu

## Tóm tắt (Abstract)

Cơ chế attention là thành phần cốt lõi của kiến trúc Transformer và các mô hình ngôn ngữ lớn (LLM). Việc triển khai attention có thể được thực hiện thủ công dựa trên công thức toán học hoặc sử dụng các hàm tối ưu hóa sẵn trong thư viện deep learning. Bài báo này phân tích quá trình hiện thực hóa attention từ công thức lý thuyết sang mã Python, so sánh giữa cài đặt thủ công và hàm `scaled_dot_product_attention` trong PyTorch, đồng thời đánh giá hiệu năng trên CPU và GPU. Kết quả cho thấy các triển khai được tối ưu và biên dịch mang lại cải thiện đáng kể về tốc độ và độ ổn định số.

---

## 1. Giới thiệu (Introduction)

Attention là cơ chế cho phép mô hình học cách tập trung vào các phần quan trọng của chuỗi đầu vào. Trong Transformer, attention được sử dụng để tính toán mối quan hệ giữa các token dựa trên vector truy vấn (Query), khóa (Key) và giá trị (Value).

Tài liệu hướng dẫn lập trình attention cung cấp một bài thực hành nhằm:

- Chuyển đổi công thức attention sang mã Python,
- Cài đặt thủ công bằng PyTorch,
- So sánh với hàm tối ưu hóa có sẵn,
- Đánh giá hiệu năng thực nghiệm. 

Mục tiêu của nghiên cứu này là phân tích quá trình trên dưới góc nhìn học thuật và hệ thống.

---

## 2. Cơ sở lý thuyết (Theoretical Background)

### 2.1. Công thức Attention

Scaled Dot-Product Attention được định nghĩa:

\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V

Trong đó:

- $Q \in $\mathbb${R}^{T \times d}$: Query,
- $K \in $\mathbb${R}^{T \times d}$: Key,
- $V \in $\mathbb${R}^{T \times d}$: Value,
- $d_k$: số chiều của vector Key.

Công thức này cho phép mô hình tính toán mức độ liên quan giữa các token. 

---

### 2.2. Causal Mask trong Attention

Đối với mô hình tự hồi quy, attention cần tuân thủ ràng buộc nhân quả:

$$
j > i \Rightarrow \text{masked}
$$

Causal mask được áp dụng để ngăn mô hình truy cập token tương lai, đảm bảo tính hợp lệ khi sinh chuỗi.

---

### 2.3. Vai trò của Chuẩn hóa

Hệ số $\frac{1}{\sqrt{$d_k$}}$ được sử dụng nhằm:

- Giảm độ lớn của tích vô hướng,
- Tránh hiện tượng gradient quá lớn,
- Cải thiện độ ổn định của softmax.

Đây là yếu tố quan trọng trong huấn luyện mô hình sâu. 

---

## 3. Phương pháp (Methodology)

### 3.1. Thiết lập dữ liệu mô phỏng

Thay vì sử dụng dữ liệu thực, nghiên cứu mô phỏng các tensor ngẫu nhiên với tham số:

- Batch size: 4,
- Context length: 8,
- Vocabulary size: 40,
- Embedding dimension: 10.

Token được sinh ngẫu nhiên và ánh xạ sang embedding thông qua ma trận học được. 

---

### 3.2. Sinh Q, K, V bằng Linear Layer

Ba ma trận Q, K, V được xây dựng bằng các lớp tuyến tính:

Q = XW_Q,\quad K = XW_K,\quad V = XW_V

$$
với W_Q, W_K, W_V \in \mathbb{R}^{d \times d}.
$$

Cách tiếp cận này phản ánh đúng kiến trúc Transformer chuẩn. 

---

### 3.3. Cài đặt Attention Thủ Công

Các bước triển khai thủ công gồm:

1. Tính $QK^T$,
2. Chuẩn hóa theo $\sqrt{d}$,
3. Áp dụng causal mask,
4. Softmax theo hàng,
5. Nhân với V.

Việc xử lý phép transpose cần tránh tác động đến chiều batch. 

---

### 3.4. Sử dụng Hàm PyTorch Tối Ưu

PyTorch cung cấp hàm:

torch.nn.functional.scaled_dot_product_attention

Hàm này tích hợp:

- Masking,
- Softmax ổn định số,
- Kernel CUDA tối ưu.

Kết quả đầu ra tương đương với cài đặt thủ công. 

---

### 3.5. Đánh giá Hiệu năng

Hai phương pháp được so sánh bằng cách:

- Lặp 50.000 lần trên CPU,
- Lặp 200 lần trên GPU,
- Đo thời gian thực thi.

Ngoài ra, thử nghiệm sử dụng JIT compiler để biên dịch hàm attention. 

---

## 4. Kết quả (Results)

### 4.1. So sánh Độ chính xác

Kết quả cho thấy:

- Đầu ra của hai phương pháp gần như trùng khớp,
- Sai khác ở mức $10^{-8}$–$10^{-9}$,
- `torch.allclose` xác nhận tương đương.

Sai khác nhỏ xuất phát từ sai số làm tròn số học. 

---

### 4.2. Hiệu năng trên CPU

Kết quả thực nghiệm cho thấy:

| Phương pháp | Thời gian (50k vòng) |
|------------|----------------------|
| Thủ công | ~7.0 s |
| PyTorch | ~6.5 s |

Phiên bản tối ưu nhanh hơn do kernel được fuse. 

---

### 4.3. Hiệu năng trên GPU

Với ma trận lớn:

| Phương pháp | Thời gian |
|-------------|-----------|
| Thủ công | ~4.5 s |
| PyTorch | ~9.0 s |
| Compiled | ~0.05 s |

Sau khi JIT compile, hiệu năng cải thiện hơn 80 lần. 

---

### 4.4. Hiệu ứng Warm-up

Lần chạy đầu tiên trên GPU thường chậm hơn do:

- Khởi tạo kernel,
- Load library,
- Memory allocation.

Do đó, các lần chạy sau phản ánh chính xác hơn hiệu năng thực. 

---

## 5. Thảo luận (Discussion)

### 5.1. Ý nghĩa của Triển khai Thủ Công

Cài đặt thủ công giúp:

- Hiểu sâu công thức toán học,
- Phát hiện lỗi transpose và broadcasting,
- Nắm rõ vai trò của mask và softmax.

Đây là bước quan trọng trong đào tạo kỹ sư AI. 

---

### 5.2. Lợi thế của Hàm Tối Ưu

Hàm PyTorch cung cấp:

- Tính ổn định số cao,
- Tối ưu GPU,
- Hỗ trợ mixed precision,
- Dễ tích hợp.

Trong môi trường production, đây là lựa chọn ưu tiên.

---

### 5.3. Vai trò của Biên dịch (Compilation)

JIT compiler cho phép:

- Fuse kernel,
- Giảm overhead Python,
- Tối ưu pipeline.

Điều này minh họa vai trò của compiler trong hệ thống LLM hiện đại. 

---

### 5.4. Góc nhìn Hệ thống

Attention là phép toán được lặp lại hàng tỷ lần. Do đó:

- Tối ưu từng micro-second mang lại lợi ích lớn,
- Việc lựa chọn kernel ảnh hưởng trực tiếp đến chi phí vận hành.

---

## 6. Hạn chế (Limitations)

Nghiên cứu tồn tại một số hạn chế:

1. Chỉ xét single-head attention,
2. Chưa đánh giá multi-GPU,
3. Chưa so sánh với FlashAttention kernel,
4. Chưa phân tích backward pass.

Các hạn chế này cần được nghiên cứu thêm.

---

## 7. Ứng dụng Thực tiễn (Practical Implications)

Kết quả có thể áp dụng cho:

- Thiết kế LLM inference engine,
- Tối ưu training pipeline,
- Debug attention kernel,
- Phát triển framework AI.

Việc hiểu rõ cả hai cách triển khai giúp cải thiện độ tin cậy hệ thống.

---

## 8. Kết luận (Conclusion)

Bài báo đã phân tích quá trình triển khai attention từ công thức toán học đến mã PyTorch, so sánh giữa cài đặt thủ công và hàm tối ưu. Kết quả cho thấy:

- Hai phương pháp cho kết quả tương đương,
- Phiên bản tối ưu vượt trội về tốc độ,
- Biên dịch đóng vai trò quyết định trong hiệu năng.

Nghiên cứu khẳng định rằng việc tối ưu attention là nền tảng cho sự phát triển của các mô hình ngôn ngữ quy mô lớn.

---

## Tài liệu tham khảo (References)

[1] Code Challenge: Code Attention Manually and in PyTorch, 2024. 

[2] Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.

[3] Dao, T. et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention. NeurIPS.

[4] Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.

---

Nếu bạn muốn, mình có thể tiếp tục mở rộng cho bạn sang:

✅ Multi-Head Attention + Benchmark
✅ Backward Pass Optimization
✅ FlashAttention vs SDPA
✅ Full LLM Attention Engine Design

Bạn muốn đi tiếp theo hướng **hệ thống, kernel, hay research paper submission**?
Dưới đây là **bài viết khoa học bằng tiếng Việt** về **Multi-Head Attention và Benchmark hiệu năng**, trình bày theo chuẩn học thuật và định dạng **Markdown (MD)**.

---

# Multi-Head Attention và Đánh Giá Hiệu Năng: Phân Tích Kiến Trúc và Benchmark Thực Nghiệm

## Tóm tắt (Abstract)

Multi-Head Attention (MHA) là thành phần trung tâm trong kiến trúc Transformer và các mô hình ngôn ngữ lớn. Cơ chế này cho phép mô hình học song song nhiều không gian biểu diễn khác nhau. Bài báo này trình bày cơ sở lý thuyết của MHA, phương pháp triển khai bằng PyTorch, và đánh giá hiệu năng thông qua các benchmark trên CPU và GPU. Kết quả cho thấy việc tối ưu kernel và sử dụng hàm attention tích hợp giúp cải thiện đáng kể tốc độ huấn luyện và suy luận.

---

## 1. Giới thiệu (Introduction)

Trong kiến trúc Transformer, Single-Head Attention chỉ cho phép mô hình học một dạng quan hệ giữa các token. Điều này hạn chế khả năng biểu diễn ngữ nghĩa phức tạp.

Multi-Head Attention mở rộng cơ chế này bằng cách:

- Chia embedding thành nhiều không gian con,
- Áp dụng attention song song,
- Kết hợp kết quả để tăng năng lực biểu diễn.

MHA là nền tảng cho các mô hình như BERT, GPT và LLaMA.

---

## 2. Cơ sở lý thuyết (Theoretical Background)

### 2.1. Công thức Multi-Head Attention

Multi-Head Attention được định nghĩa:

\text{MHA}(Q,K,V) = \text{Concat}(h_1,\dots,h_H)W_O

với:

h_i = \text{Attention}(QW_i^Q,KW_i^K,VW_i^V)

Trong đó:

- $H$: số head,
- $W_i^Q, $W_i$^K, $W_i$^V$: ma trận chiếu,
- $W_$O(: ma trận đầu ra.

Mỗi head học một không gian biểu diễn riêng biệt.

---

### 2.2. Phân rã Không gian Đặc trưng

Với embedding dimension )$d$:

d_{head} = \frac{d}{H}

Mỗi head xử lý tensor kích thước:

$$
(T, d_{head})
$$

Cách chia này giúp:

- Giảm chi phí tính toán mỗi head,
- Tăng khả năng học quan hệ đa chiều.

---

### 2.3. Causal Multi-Head Attention

Trong mô hình tự hồi quy, mỗi head đều áp dụng causal mask:

M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}

Mask này đảm bảo không rò rỉ thông tin tương lai.

---

## 3. Phương pháp (Methodology)

### 3.1. Môi trường Thực nghiệm

- Framework: PyTorch
- Phần cứng:
  - CPU: x86-64
  - GPU: NVIDIA CUDA
- Precision: FP32 / FP16
- Context length: 128–1024
- Heads: 4, 8, 16

---

### 3.2. Kiến trúc Mô hình

Mô hình thử nghiệm gồm:

1. Embedding layer
2. Multi-Head Attention
3. Feedforward
4. LayerNorm

Cấu trúc tương đương một block Transformer tiêu chuẩn.

---

### 3.3. Pseudocode Multi-Head Attention

Input: X ∈ R^(B×T×d)
Output: Y ∈ R^(B×T×d)

for each head i in H:

Qi = X · WQi

$$
Ki = X · WKi
$$

Vi = X · WVi

$$
Ai = softmax(Qi Ki^T / sqrt(dh) + Mask)
$$

Hi = Ai · Vi

$$
H = concat(H1,...,HH)
$$

Y = H · WO

````

---

### 3.4. PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        

assert d_model % n_heads == 0

$$
self.d_model = d_model
$$

self.n_heads = n_heads

$$
self.d_head = d_model // n_heads
$$

self.qkv = nn.Linear(d_model, 3 * d_model)

$$
self.out = nn.Linear(d_model, d_model)
$$

def forward(self, x, causal=True):

$$
B, T, D = x.shape
$$

qkv = self.qkvx

$$
qkv = qkv.view(B, T, 3, self.n_heads, self.d_head)
$$

qkv = qkv.permute(2, 0, 3, 1, 4)

$$
q, k, v = qkv[0], qkv[1], qkv[2]
$$

scores = torch.matmul(q, k.transpose(-2, -1))

$$
scores /= self.d_head ** 0.5 if causal: mask = torch.tril(torch.ones(T, T, device=x.device))
$$

scores = scores.masked_fill(mask == 0, -1e9)

$$
attn = F.softmax(scores, dim=-1)
$$

out = torch.matmul(attn, v)

$$
out = out.transpose(1, 2).contiguous()
$$

out = out.view(B, T, D)

        
        return self.out(out)
````

---

## 4. Thiết kế Benchmark (Benchmark Design)

### 4.1. Cấu hình Đánh giá

Các biến số:

| Tham số    | Giá trị        |
| ---------- | -------------- |
| Batch size | 1, 8, 32       |
| Seq length | 128, 512, 1024 |
| Heads      | 4, 8, 16       |
| Precision  | FP32, FP16     |

---

### 4.2. Quy trình Đo lường

1. Warm-up 50 vòng
2. Chạy 500–1000 vòng
3. Đo trung bình thời gian
4. Đồng bộ CUDA
5. Loại bỏ outlier

---

### 4.3. Mã Benchmark

```python
import time

$$
def benchmark(model, x, runs=500):
$$

    torch.cuda.synchronize()
    
    for _ in range(50):

$$
_ = modelx start = time.time() for _ in range(runs):
$$

_ = modelx