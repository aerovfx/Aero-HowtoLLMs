# Phân tích và chuyển đổi văn bản thành chuỗi token được đánh số: Cơ sở toán học và ứng dụng trong mô hình ngôn ngữ lớn

---

## Tóm tắt

Quá trình phân tích (parsing) văn bản thành các token được đánh số là bước nền tảng trong huấn luyện và suy luận của các mô hình ngôn ngữ lớn (LLMs). Bài viết này trình bày cơ sở lý thuyết của tokenization, đánh số vị trí (positional indexing), và vai trò của chúng trong kiến trúc Transformer. Phân tích dựa trên các công trình nền tảng như Transformer của Vaswani et al. (2017) và các mô hình GPT do OpenAI phát triển. Các công thức toán học minh họa quá trình ánh xạ văn bản sang không gian vector và cách mô hình xử lý chuỗi có thứ tự.

---

# 1. Giới thiệu

Máy tính không xử lý trực tiếp “từ” hay “câu” như con người, mà xử lý **chuỗi số**.

Do đó, văn bản phải được:

1. Tách thành token
2. Ánh xạ thành chỉ số (ID)
3. Chuyển thành vector embedding
4. Đánh số theo vị trí trong chuỗi

Ví dụ:

> "AI is powerful"

Sau tokenization có thể trở thành:

[
["AI", " is", " powerful"]
]

Và được ánh xạ thành:

[
[50256, 318, 3665]
]

---

# 2. Tokenization: Cơ sở toán học

Giả sử tập từ vựng (V) có kích thước:

[
|V| = N
]

Hàm tokenization:

[
T: \mathcal{X} \to V^T
]

với:

* ( \mathcal{X} ): không gian văn bản
* (V^T): chuỗi các token ID

Nếu chuỗi văn bản là (x), ta có:

[
T(x) = (t_1, t_2, ..., t_T)
]

Mỗi (t_i \in {1,2,...,N})

---

# 3. Byte Pair Encoding (BPE)

GPT sử dụng BPE để xử lý từ hiếm.

Giả sử ban đầu ta có tập ký tự (C).
Thuật toán lặp:

1. Tìm cặp ký tự xuất hiện nhiều nhất
2. Gộp thành token mới
3. Thêm vào từ vựng

Quá trình tối ưu hóa nhằm giảm entropy:

[
H(X) = -\sum_x P(x)\log P(x)
]

BPE giúp:

* Giảm độ dài chuỗi (T)
* Tăng hiệu quả tính toán

---

# 4. Đánh số token (Positional Indexing)

Sau tokenization:

[
(t_1, t_2, ..., t_T)
]

Ta cần biểu diễn thứ tự:

[
i = 1,2,...,T
]

Nếu không có chỉ số vị trí, mô hình Transformer sẽ bất biến hoán vị.

---

## 4.1. Biểu diễn embedding

Mỗi token ID được ánh xạ:

[
e_i = E(t_i)
]

Vector đầu vào cuối cùng:

[
z_i = e_i + p_i
]

Trong đó:

* (p_i): vector vị trí

---

# 5. Self-Attention và vai trò của thứ tự

Attention được định nghĩa:

[
\text{Attention}(Q,K,V)
=======================

\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
]

Nếu không có positional encoding:

[
\text{Attention}(PX) = P\text{Attention}(X)
]

→ Không phân biệt thứ tự.

Khi thêm (p_i):

[
Z = E + P
]

ma trận attention phản ánh quan hệ phụ thuộc có hướng.

---

# 6. Causal Masking

Trong mô hình tự hồi quy (GPT):

[
P(x) = \prod_{t=1}^{T} P(x_t | x_{<t})
]

Mask:

[
M_{ij} =
\begin{cases}
0 & j \le i \
-\infty & j > i
\end{cases}
]

Ma trận attention thực tế:

[
A = \text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}} + M
\right)
]

Đánh số token cho phép xác định chính xác vị trí (i).

---

# 7. Độ phức tạp tính toán

Self-attention:

[
\mathcal{O}(T^2 d)
]

Nếu chiều dài chuỗi tăng gấp đôi:

[
\text{Compute} \approx 4\times
]

Do đó việc tokenization hiệu quả giúp:

* Giảm (T)
* Giảm chi phí huấn luyện

---

# 8. Ví dụ minh họa

Giả sử câu:

> "Machine learning is amazing"

Tokenization:

[
[1543, 4673, 318, 4996]
]

Embedding:

[
E \in \mathbb{R}^{|V| \times d}
]

Đầu vào:

[
Z \in \mathbb{R}^{T \times d}
]

Qua attention:

[
Z' = \text{Transformer}(Z)
]

---

# 9. Liên hệ với Reinforcement Learning from Human Feedback

Trong RLHF:

[
x = [\text{Prompt}; \text{Response}]
]

Đánh số cho phép:

* Phân biệt đoạn cần tối ưu
* Mask loss chính xác

Loss:

[
\mathcal{L} = - \sum_{t \in R} \log P(x_t | x_{<t})
]

---

# 10. Thảo luận

Quá trình parsing text to numbered tokens là:

* Bước đầu tiên của NLP pipeline
* Điều kiện cần cho Transformer hoạt động
* Yếu tố quyết định hiệu suất tính toán

Nếu bỏ bước này:

[
\text{Model} \to \text{Không thể huấn luyện}
]

---

# 11. Kết luận

Chuyển đổi văn bản thành chuỗi token được đánh số là:

1. Nền tảng của mô hình ngôn ngữ
2. Cơ sở cho self-attention
3. Điều kiện để thực hiện causal modeling

Toán học cho thấy thứ tự là thành phần thiết yếu trong biểu diễn ngôn ngữ.

---

# Tài liệu tham khảo

1. Vaswani, A. et al. (2017). *Attention Is All You Need*.
2. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*.
3. Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*.
4. Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.

---