
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [02 words to tokens to numbers](index.md)

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
Tokenization trong BERT: Phân tích Cơ chế WordPiece và Mô hình Toán học

⸻

Tóm tắt

Bài viết này phân tích cơ chế tokenization trong BERT dựa trên tài liệu đính kèm, tập trung vào thuật toán WordPiece, nền tảng của quá trình phân tách subword trong mô hình Google phát triển – BERT. Chúng tôi trình bày cơ sở toán học của WordPiece, so sánh với Byte Pair Encoding (BPE), đồng thời phân tích ảnh hưởng của tokenization đến embedding, attention và hàm mất mát trong huấn luyện.

⸻

1. Giới thiệu

BERT (Bidirectional Encoder Representations from Transformers) là mô hình Transformer encoder hai chiều được giới thiệu năm 2018.

Chuỗi đầu vào:

S = (w_1, w_2, ..., w_n)

Được ánh xạ thành chuỗi token:

T = (t_1, t_2, ..., t_m)

Với:

m \ge n

Do một từ có thể bị tách thành nhiều subword.

⸻

2. Thuật toán WordPiece

2.1 Nguyên lý cơ bản

WordPiece bắt đầu từ tập ký tự cơ sở và lặp lại quá trình:
	•	Chọn cặp subword có xác suất cao nhất
	•	Gộp lại thành một token mới

Khác với BPE (chọn theo tần suất), WordPiece tối ưu theo xác suất tối đa hóa likelihood.

⸻

2.2 Hàm Mục tiêu

Giả sử tập dữ liệu huấn luyện D.

WordPiece tối đa hóa:

$$
\mathcal{L} = \sum_{w \in D} \log P(w)
$$

Trong đó một từ w được phân rã thành:

w = (t_1, t_2, ..., t_k)

Xác suất:

$P(w)$ = $\prod$_{i=1}^{k} $P($t_i$)$

Thuật toán chọn phép gộp làm tăng likelihood nhiều nhất.

⸻

2.3 Quy tắc Tiền tố “##”

Ví dụ:

playing → play + ##ing

Ký hiệu “##” cho biết token không ở đầu từ.

Điều này giúp mô hình phân biệt:
	•	“play” (từ độc lập)
	•	“##play” (không hợp lệ)

⸻

3. Mô hình Toán học của Tokenization

3.1 Phân bố Subword

Gọi:
	•	V: tập từ vựng
	•	|V|: kích thước (≈ 30k với BERT-base)

Phân bố:

$P(t)$ = \frac{\text{count}$t$}{$\sum$_{t' \in V} \text{count}(t')}

Entropy:

H = - \sum_{t \in V} P(t)\log P(t)

⸻

3.2 Độ dài Trung bình Chuỗi Token

Nếu văn bản có n từ và trung bình mỗi từ tách thành \alpha subword:

$$
m = \alpha n
$$

Self-attention trong Transformer encoder:

$O(m^2)$

Do đó:

$O((\alpha n)$^2)

Tokenization ảnh hưởng trực tiếp đến chi phí tính toán.

⸻

4. So sánh WordPiece và BPE

Đặc điểm	WordPiece	BPE
Tiêu chí gộp	Tối đa hóa likelihood	Tần suất
Mô hình xác suất	Có	Không trực tiếp
Ứng dụng	BERT	GPT
Tối ưu	Theo corpus	Theo tần suất thuần

⸻

5. Biểu diễn Embedding

Mỗi token được ánh xạ:

$$
E: V \rightarrow \mathbb{R}^d
$$

Chuỗi token tạo thành ma trận:

$$
X \in \mathbb{R}^{m \times d}
$$

BERT cộng thêm:
	•	Positional embedding
	•	Segment embedding

Tổng embedding:

$$
E_{\text{total}} = E_{\text{token}} + E_{\text{position}} + E_{\text{segment}}
$$

⸻

6. Masked Language Modeling (MLM)

BERT huấn luyện bằng cách che một số token:

$P($t_i$ \mid  T_{\setminus i})$

Loss:

$$
\mathcal{L}_{MLM} = - \sum_{i \in M} \log P(t_i \mid  T_{\setminus i})
$$

Trong đó M là tập token bị mask.

Tokenization ảnh hưởng trực tiếp đến:
	•	Số token bị mask
	•	Độ khó của nhiệm vụ dự đoán

⸻

7. Phân tích Lý thuyết Thông tin

Tokenization tối ưu hóa sự cân bằng giữa:
	•	Vocabulary size |V|
	•	Độ dài chuỗi m

Bài toán tối ưu:

\min_{V} \left( \mathbb{E}[m] + \lambda |V| \right)

Với:
	•	\lambda: hệ số điều chỉnh

$$
•	\mathbb{E}[m]: số token trung bình
$$

⸻

8. Tính Khái quát hóa (Generalization)

WordPiece cho phép xử lý từ hiếm:

Ví dụ:

unbelievable → un + ##believ + ##able

Do đó:

\forall w \notin V_{word}, \exists \text{decomposition in } V_{subword}

Giảm vấn đề OOV (Out-of-Vocabulary).

⸻

9. Hạn chế
	1.	Phụ thuộc corpus huấn luyện
	2.	Có thể tách không tự nhiên về mặt ngôn ngữ
	3.	Tăng độ dài chuỗi trong ngôn ngữ có cấu trúc phức tạp

⸻

10. Kết luận

Tokenization trong BERT dựa trên WordPiece có thể được mô hình hóa:

\max \sum_{w \in D} \log \prod_{i=1}^{k} P(t_i)

Ảnh hưởng trực tiếp đến:

$$
m = \alpha n \text{Attention Cost} = O(m^2) H = - \sum P(t)\log P(t)
$$

