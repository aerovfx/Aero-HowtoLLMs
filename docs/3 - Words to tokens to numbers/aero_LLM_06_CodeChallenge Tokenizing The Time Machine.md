# Phân tích quy trình Tokenization qua ví dụ *The Time Machine*: Cơ sở thuật toán và mô hình hóa toán học

---

## Tóm tắt

Bài viết này trình bày một phân tích khoa học về quy trình tokenization thông qua ví dụ văn bản *The Time Machine* của H. G. Wells. Nội dung tập trung vào cách tiền xử lý văn bản, xây dựng từ vựng, ánh xạ token sang chỉ số và chuyển đổi sang không gian vector phục vụ huấn luyện mô hình Transformer. Bài viết bổ sung các nền tảng lý thuyết từ kiến trúc Attention Is All You Need và các mô hình GPT do OpenAI phát triển, kèm theo các công thức toán học minh họa cho quá trình rời rạc hóa và biểu diễn liên tục.

---

# 1. Giới thiệu

Trong các mô hình ngôn ngữ hiện đại, dữ liệu văn bản phải được chuyển đổi từ dạng ký tự sang dạng token trước khi đưa vào mạng nơ-ron.

Cho văn bản đầu vào:

[
X = (c_1, c_2, \dots, c_n), \quad c_i \in \Sigma
]

Tokenization thực hiện ánh xạ:

[
\tau : \Sigma^* \rightarrow V^*
]

trong đó:

* (\Sigma): tập ký tự
* (V): từ vựng token
* (V^*): chuỗi token

Ví dụ với *The Time Machine*:

```
The Time Machine by H. G. Wells
```

Sau xử lý có thể thành:

[
[\text{"the"}, \text{"time"}, \text{"machine"}, \text{"by"}, \text{"h"}, \text{"g"}, \text{"wells"}]
]

---

# 2. Tiền xử lý văn bản

## 2.1 Chuẩn hóa chữ thường

[
f_{lower}(x) = \text{lower}(x)
]

Giúp giảm kích thước từ vựng:

[
|V_{raw}| > |V_{normalized}|
]

---

## 2.2 Loại bỏ ký tự đặc biệt

Hàm lọc:

[
f_{clean}(x) = x \setminus { \text{punctuation} }
]

Mục tiêu:

* Giảm nhiễu
* Chuẩn hóa cấu trúc

---

# 3. Tokenization mức từ (Word-level Tokenization)

Sau khi tách theo khoảng trắng:

[
X = (w_1, w_2, \dots, w_T)
]

Số lượng token:

[
T \leq n
]

Tần suất xuất hiện của từ (w):

[
f(w) = \sum_{i=1}^{T} \mathbf{1}(w_i = w)
]

---

# 4. Xây dựng từ vựng (Vocabulary Construction)

Tập từ vựng:

[
V = { w \mid f(w) \geq \delta }
]

với (\delta) là ngưỡng tối thiểu.

Kích thước từ vựng:

[
|V| = M
]

Ánh xạ:

[
w \rightarrow id(w) \in {0, 1, \dots, M-1}
]

---

# 5. Biểu diễn One-Hot

Token (w_i) được biểu diễn:

[
x_i \in \mathbb{R}^{M}
]

với:

[
x_{ij} =
\begin{cases}
1 & \text{nếu } j = id(w_i) \
0 & \text{ngược lại}
\end{cases}
]

Nhược điểm:

* Kích thước lớn
* Không phản ánh ngữ nghĩa

---

# 6. Embedding Vector

Embedding matrix:

[
E \in \mathbb{R}^{M \times d}
]

Vector embedding:

[
e_i = E^T x_i
]

Do đó:

[
e_i \in \mathbb{R}^{d}
]

Khoảng cách cosine:

[
\cos(e_i, e_j) =
\frac{e_i \cdot e_j}{|e_i||e_j|}
]

Giúp đo mức độ tương đồng ngữ nghĩa.

---

# 7. Mô hình hóa xác suất ngôn ngữ

Theo mô hình tự hồi quy:

[
P(X) = \prod_{t=1}^{T} P(w_t \mid w_{<t})
]

Mạng Transformer tính:

[
Z = \text{Transformer}(e_1, \dots, e_T)
]

Logits:

[
z_t = W_{out} h_t
]

Softmax:

[
P(w_t = j \mid w_{<t}) =
\frac{\exp(z_{tj})}
{\sum_{k=1}^{M} \exp(z_{tk})}
]

---

# 8. Độ phức tạp tính toán

Self-attention:

[
\mathcal{O}(T^2 d)
]

Nếu văn bản dài như *The Time Machine* (~30,000 từ), chi phí tăng theo bình phương độ dài chuỗi.

Do đó, tokenization tối ưu giúp:

* Giảm (T)
* Giảm bộ nhớ
* Tăng tốc huấn luyện

---

# 9. Phân tích thống kê văn bản

Entropy của tập từ:

[
H(W) = - \sum_{w \in V} P(w) \log P(w)
]

Với:

[
P(w) = \frac{f(w)}{T}
]

Nếu phân bố Zipf:

[
f(w_r) \propto \frac{1}{r}
]

trong đó (r) là thứ hạng tần suất.

Điều này cho thấy:

* Số ít từ xuất hiện rất nhiều
* Nhiều từ hiếm xuất hiện

---

# 10. So sánh với Subword Tokenization

Word-level tokenization có nhược điểm:

[
P(\text{OOV}) > 0
]

Giải pháp: Byte Pair Encoding (BPE).

Tập hợp phân rã:

[
w = s_1 s_2 \dots s_k
]

với (s_i \in V_{subword})

Đảm bảo:

[
\forall w, \exists \text{ decomposition}
]

---

# 11. Thảo luận

Tokenization là quá trình:

[
\text{Text} \rightarrow \text{Discrete Representation} \rightarrow \text{Continuous Geometry}
]

Về mặt toán học:

* Là ánh xạ từ chuỗi ký tự sang không gian vector
* Là bước nén thông tin
* Ảnh hưởng trực tiếp đến phân phối xác suất

---

# 12. Kết luận

Thông qua ví dụ *The Time Machine*, ta thấy:

1. Tokenization quyết định cấu trúc dữ liệu đầu vào
2. Vocabulary ảnh hưởng đến kích thước embedding
3. Biểu diễn vector quyết định khả năng học ngữ nghĩa
4. Độ dài chuỗi ảnh hưởng đến độ phức tạp Transformer

Toàn bộ quá trình có thể được mô hình hóa:

[
\Sigma^* \xrightarrow{\tau} V^* \xrightarrow{E} \mathbb{R}^{T \times d}
]

đóng vai trò nền tảng cho mọi mô hình Transformer hiện đại.

---

# Tài liệu tham khảo

1. Attention Is All You Need
2. Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*.
3. Shannon, C. (1948). *A Mathematical Theory of Communication*.
4. Jurafsky, D., Martin, J. (2023). *Speech and Language Processing*.
5. Manning, C., Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*.

