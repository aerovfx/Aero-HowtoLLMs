# Tại sao văn bản cần được đánh số?

## Phân tích khoa học về cấu trúc, tính toán và tối ưu hóa xử lý ngôn ngữ

---

## Tóm tắt

Trong các hệ thống xử lý ngôn ngữ tự nhiên (NLP) hiện đại, việc **đánh số văn bản (token numbering / positional indexing)** đóng vai trò nền tảng trong biểu diễn chuỗi, tính toán attention và tối ưu hóa mô hình Transformer. Bài viết này phân tích vai trò của đánh số văn bản dưới góc độ toán học, khoa học nhận thức và kiến trúc mô hình ngôn ngữ lớn. Phân tích dựa trên kiến trúc Transformer của Vaswani et al. (2017) và các mô hình GPT do OpenAI phát triển.

---

# 1. Giới thiệu

Con người hiểu văn bản theo trình tự tuyến tính. Máy tính cũng cần một cơ chế tương tự để:

* Phân biệt vị trí token
* Xác định quan hệ phụ thuộc
* Tính toán attention

Nếu không đánh số hoặc mã hóa vị trí, chuỗi:

> “Tôi yêu AI”

và

> “AI yêu tôi”

sẽ có cùng tập token nhưng ý nghĩa hoàn toàn khác.

Vấn đề này dẫn đến nhu cầu **positional encoding** trong các mô hình Transformer.

---

# 2. Biểu diễn chuỗi dưới dạng toán học

Giả sử một câu gồm ( T ) token:

[
x = (x_1, x_2, ..., x_T)
]

Mỗi token được ánh xạ thành vector embedding:

[
e_i = E(x_i)
]

Nếu không có đánh số vị trí, ta chỉ có:

[
X = (e_1, e_2, ..., e_T)
]

Nhưng self-attention thuần túy là **bất biến hoán vị (permutation invariant)**.

Điều này có nghĩa:

[
\text{Attention}(X) = \text{Attention}(PX)
]

với ( P ) là ma trận hoán vị.

Do đó, mô hình không phân biệt thứ tự.

---

# 3. Positional Encoding

## 3.1. Mã hóa vị trí sin-cos

Transformer nguyên bản sử dụng:

[
PE(pos, 2i) = \sin \left( \frac{pos}{10000^{2i/d}} \right)
]

[
PE(pos, 2i+1) = \cos \left( \frac{pos}{10000^{2i/d}} \right)
]

Trong đó:

* (pos): vị trí token
* (i): chỉ số chiều embedding
* (d): kích thước embedding

Vector đầu vào:

[
z_i = e_i + PE(i)
]

---

## 3.2. Positional Embedding học được

Trong GPT:

[
z_i = e_i + p_i
]

với (p_i) là tham số học được.

Điều này cho phép mô hình tối ưu trực tiếp biểu diễn vị trí.

---

# 4. Vai trò của đánh số trong Self-Attention

Attention được tính:

[
\text{Attention}(Q,K,V)
=======================

\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
]

Trong đó:

[
Q = ZW_Q, \quad
K = ZW_K
]

Nếu (Z) không chứa thông tin vị trí:

[
QK^T
]

chỉ phản ánh nội dung, không phản ánh thứ tự.

Khi có positional encoding:

[
Z = E + P
]

attention có thể học:

* Quan hệ xa
* Phụ thuộc cú pháp
* Quan hệ nguyên nhân – kết quả

---

# 5. Đánh số văn bản trong huấn luyện mô hình ngôn ngữ

Mô hình GPT tối ưu:

[
P(x) = \prod_{t=1}^{T} P(x_t | x_{<t})
]

Điều kiện (x_{<t}) phụ thuộc trực tiếp vào thứ tự.

Causal masking:

[
M_{ij} =
\begin{cases}
0 & j \le i \
-\infty & j > i
\end{cases}
]

Ma trận attention thực tế:

[
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}} + M
\right)
]

Đánh số vị trí cho phép xác định chính xác token nào thuộc (x_{<t}).

---

# 6. Đánh số và tối ưu hóa tính toán

Self-attention có độ phức tạp:

[
\mathcal{O}(T^2 d)
]

Khi tăng chiều dài văn bản (T):

[
\text{Compute} \propto T^2
]

Việc đánh số giúp:

* Quản lý cửa sổ ngữ cảnh
* Chia chunk
* Triển khai sliding window

---

# 7. Ảnh hưởng trong Reinforcement Learning from Human Feedback

Trong RLHF, chuỗi gồm:

[
x = [\text{Prompt}; \text{Response}]
]

Đánh số cho phép:

* Phân biệt phần prompt và response
* Mask loss chính xác

Loss:

[
\mathcal{L} = -\sum_{t \in R} \log P(x_t | x_{<t})
]

Nếu không đánh số rõ ràng, mô hình không biết đâu là phần cần tối ưu.

---

# 8. Góc nhìn lý thuyết thông tin

Entropy của chuỗi:

[
H(X) = - \sum_x P(x)\log P(x)
]

Thứ tự ảnh hưởng trực tiếp đến entropy.

Ví dụ:

* Chuỗi có cấu trúc → entropy thấp
* Chuỗi ngẫu nhiên → entropy cao

Đánh số giúp mô hình ước lượng xác suất chính xác hơn.

---

# 9. So sánh các phương pháp encoding vị trí

| Phương pháp   | Công thức       | Ưu điểm           | Nhược điểm         |
| ------------- | --------------- | ----------------- | ------------------ |
| Sin-Cos       | Hàm lượng giác  | Không cần học     | Cứng               |
| Learned       | Vector học được | Linh hoạt         | Giới hạn chiều dài |
| Rotary (RoPE) | Phép quay phức  | Tổng quát hóa tốt | Phức tạp           |
| ALiBi         | Bias tuyến tính | Dài ngữ cảnh tốt  | Giảm linh hoạt     |

---

# 10. Thảo luận

Đánh số văn bản không chỉ là vấn đề kỹ thuật mà là:

* Điều kiện cần cho mô hình hiểu ngữ nghĩa
* Cơ sở cho attention hoạt động
* Yếu tố then chốt trong huấn luyện LLM

Nếu bỏ positional encoding:

[
\text{Transformer} \to \text{Bag-of-Words Model}
]

---

# 11. Kết luận

Việc đánh số văn bản là nền tảng của:

1. Mô hình hóa chuỗi
2. Self-attention
3. Causal masking
4. Huấn luyện autoregressive

Về mặt toán học, positional encoding đưa thêm thông tin vị trí vào không gian embedding, phá vỡ tính bất biến hoán vị và cho phép mô hình học cấu trúc ngôn ngữ.

---

# Tài liệu tham khảo

1. Vaswani, A. et al. (2017). *Attention Is All You Need*.
2. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*.
3. Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
4. Press, O. et al. (2021). *Train Short, Test Long: Attention with Linear Biases*.

---