# So sánh Tokenization mức ký tự, từ và subword: Phân tích lý thuyết và mô hình toán học

---

## Tóm tắt

Bài báo này phân tích và so sánh ba chiến lược tokenization phổ biến trong xử lý ngôn ngữ tự nhiên: **character-level**, **word-level** và **subword-level**. Dựa trên nền tảng kiến trúc Attention Is All You Need và các mô hình ngôn ngữ lớn như GPT-4 do OpenAI phát triển, bài viết mô hình hóa toán học sự khác biệt giữa các phương pháp, phân tích entropy, độ phức tạp tính toán và tác động đến self-attention.

---

# 1. Giới thiệu

Tokenization là quá trình ánh xạ:

[
\tau: \Sigma^* \rightarrow V^*
]

trong đó:

* (\Sigma): tập ký tự
* (V): tập token
* (\Sigma^*): chuỗi ký tự
* (V^*): chuỗi token

Ba chiến lược chính:

1. Character-level
2. Word-level
3. Subword-level (BPE, Unigram LM)

Mỗi phương pháp tạo ra độ dài chuỗi (T) và kích thước từ vựng (|V|) khác nhau.

---

# 2. Tokenization mức ký tự (Character-Level)

## 2.1 Định nghĩa

Mỗi token là một ký tự:

[
V = \Sigma
]

Chuỗi:

[
X = (c_1, c_2, \dots, c_n)
]

Số token:

[
T = n
]

---

## 2.2 Ưu điểm

* Không có OOV:

[
\forall x \in \Sigma^*, \tau(x) \text{ luôn tồn tại}
]

* Kích thước từ vựng nhỏ:

[
|V| \approx 100 - 500
]

---

## 2.3 Nhược điểm

Self-attention có độ phức tạp:

[
\mathcal{O}(T^2 d)
]

Vì (T = n) lớn → chi phí tăng mạnh.

Ví dụ: văn bản 1000 ký tự

[
T_{char} = 1000
]

Chi phí attention:

[
\propto 1000^2 = 10^6
]

---

# 3. Tokenization mức từ (Word-Level)

## 3.1 Định nghĩa

Chuỗi:

[
X = (w_1, w_2, \dots, w_m)
]

với:

[
m < n
]

Tập từ vựng:

[
V = { w }
]

---

## 3.2 Đặc điểm thống kê

Phân bố tần suất từ tuân theo định luật Zipf:

[
f(w_r) \propto \frac{1}{r}
]

trong đó (r) là thứ hạng.

Entropy:

[
H(W) = -\sum_{w} P(w)\log P(w)
]

---

## 3.3 Nhược điểm

Xác suất OOV:

[
P(\text{OOV}) = 1 - \sum_{w \in V} P(w)
]

Vì từ vựng hữu hạn.

Kích thước từ vựng lớn:

[
|V| \approx 30,000 - 200,000
]

Embedding matrix:

[
E \in \mathbb{R}^{|V| \times d}
]

→ tiêu tốn bộ nhớ.

---

# 4. Tokenization mức Subword

Subword kết hợp ưu điểm của hai phương pháp trên.

## 4.1 Byte Pair Encoding (BPE)

BPE lặp lại:

[
(a^*, b^*) = \arg\max_{a,b} f(a,b)
]

Cập nhật từ vựng:

[
V_{k+1} = V_k \cup {ab}
]

---

## 4.2 Unigram Language Model

Tối ưu:

[
\max_{\theta} \prod_i \sum_{z \in \mathcal{Z}(x_i)} P(z|\theta)
]

Trong đó:

* (z): một phân tách hợp lệ
* (\mathcal{Z}(x_i)): tập các phân tách

---

## 4.3 Độ dài chuỗi trung bình

Giả sử:

* Character-level: (T_c = n)
* Word-level: (T_w = m)
* Subword-level: (T_s)

Thông thường:

[
m < T_s < n
]

Do đó:

[
T_s^2 < T_c^2
]

và

[
|V_s| < |V_w|
]

---

# 5. So sánh độ phức tạp

| Phương pháp | Độ dài (T) | Từ vựng (|V|) | OOV | Chi phí attention |
|-------------|--------------|-----------------|------|-------------------|
| Character | Lớn | Nhỏ | Không | Rất cao |
| Word | Nhỏ | Rất lớn | Có | Thấp |
| Subword | Trung bình | Trung bình | Không | Trung bình |

Self-attention:

[
\text{Cost} = \mathcal{O}(T^2 d)
]

Embedding memory:

[
\mathcal{O}(|V| d)
]

Subword tối ưu cân bằng hai yếu tố.

---

# 6. Phân tích thông tin

Theo định lý Shannon:

[
H(X) = -\sum_x P(x)\log P(x)
]

Chiều dài mã tối ưu:

[
L \approx \frac{H(X)}{\log |V|}
]

Subword giúp:

* Giảm chiều dài chuỗi
* Giảm entropy điều kiện

---

# 7. Ảnh hưởng đến Transformer

Mô hình Transformer tính:

[
Z = \text{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
]

Vì attention phụ thuộc (T):

* Character-level → khó mở rộng
* Word-level → vấn đề OOV
* Subword → cân bằng tối ưu

Các mô hình GPT hiện đại sử dụng biến thể byte-level BPE.

---

# 8. Thảo luận thực nghiệm

Trong thực tế:

* Character-level phù hợp cho dữ liệu nhiễu
* Word-level phù hợp cho corpora nhỏ
* Subword-level phù hợp cho LLM quy mô lớn

Giả sử chuỗi 1000 ký tự:

[
T_c = 1000
]
[
T_s \approx 250 - 400
]
[
T_w \approx 150 - 250
]

Chi phí attention giảm theo bình phương độ dài.

---

# 9. Kết luận

Tokenization có thể xem là bài toán tối ưu đa mục tiêu:

[
\min_{V} \left( \alpha T^2 + \beta |V| \right)
]

Trong đó:

* (T): độ dài chuỗi
* (|V|): kích thước từ vựng
* (\alpha, \beta): trọng số chi phí

Subword tokenization là nghiệm cân bằng gần tối ưu trong thực tế.

---

# Tài liệu tham khảo

1. Attention Is All You Need
2. Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*.
3. Kudo, T. (2018). *Subword Regularization*.
4. Shannon, C. (1948). *A Mathematical Theory of Communication*.
5. Jurafsky, D., Martin, J. (2023). *Speech and Language Processing*.

