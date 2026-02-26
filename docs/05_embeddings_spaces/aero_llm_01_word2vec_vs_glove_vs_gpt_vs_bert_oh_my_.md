
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [05 embeddings spaces](index.md)

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
So sánh Word2Vec, GloVe, GPT và BERT:

Từ Biểu diễn Phân bố Tuyến tính đến Transformer Tự Hồi quy và Hai Chiều

⸻

Tóm tắt

Bài viết này tổng hợp và phân tích nội dung từ tài liệu đính kèm “Word2Vec vs. GloVe vs. GPT vs. BERT… oh my!”, đồng thời mở rộng với các nguồn học thuật nền tảng nhằm làm rõ sự tiến hóa của mô hình biểu diễn ngôn ngữ: từ embedding tĩnh (static embeddings) như Word2Vec và GloVe đến mô hình ngữ cảnh hóa (contextual embeddings) như GPT và BERT.

Chúng tôi phân tích:
	•	Mô hình toán học nền tảng
	•	Hàm mục tiêu huấn luyện
	•	Cấu trúc xác suất
	•	Tính chất tuyến tính của embedding
	•	Attention và self-attention
	•	So sánh định lượng về độ phức tạp tính toán

Các công thức toán học được trình bày nhằm minh họa rõ sự khác biệt bản chất giữa các thế hệ mô hình.

⸻

1. Giới thiệu

Biểu diễn từ (word representation) là bài toán trung tâm trong xử lý ngôn ngữ tự nhiên (NLP).

Ta xét một tập từ vựng:

V = \{w_1, w_2, \dots, w_{|V|}\}

Mục tiêu là xây dựng ánh xạ:

$$
E: V \rightarrow \mathbb{R}^d
$$

Trong đó d là số chiều embedding.

Lịch sử phát triển có thể chia thành hai giai đoạn chính:
	1.	Embedding tĩnh (Static embeddings)
	•	Word2Vec
	•	GloVe
	2.	Embedding ngữ cảnh hóa (Contextual embeddings)
	•	GPT
	•	BERT

⸻

2. Word2Vec: Mô hình dựa trên Ngữ cảnh Cục bộ

2.1 Tổng quan

Word2Vec (Mikolov et al., 2013) dựa trên giả thuyết phân bố:

$P(w \mid context)$

Hai biến thể chính:
	•	CBOW (Continuous Bag of Words)
	•	Skip-gram

⸻

2.2 Mô hình Skip-gram

Giả sử chuỗi từ:

$w_1$, $w_2$, \dots, $w_T$

Hàm mục tiêu:

\max \sum_{t=1}^{T} \sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} \mid w_t)

Với:

$P($w_$O( \mid )$w_I$)$ = \frac{\exp$v_{$w_$O(}^\top v_{)$w_I$}$}{$\sum$_{w \in V} \exp$v_w^\top v_{$w_I$}$}

Do chi phí tính toán lớn, sử dụng negative sampling:

$\log$ \sigma$v_{$w_$O(}^\top v_{)$w_I$}$ + $\sum$_{i=1}^{k} $\mathbb${E}_{$w_i$ \sim P_n$w$} $\log$ \sigma$-v_{$w_i$}^\top v_{$w_I$}$

⸻

2.3 Tính chất Tuyến tính

Một tính chất nổi tiếng:

\text{king} - \text{man} + \text{woman} \approx \text{queen}

Toán học:

v_{king} - v_{man} + v_{woman} \approx v_{queen}

Điều này cho thấy embedding học được cấu trúc tuyến tính.

⸻

3. GloVe: Ma trận Đồng xuất hiện Toàn cục

3.1 Tổng quan

GloVe (Pennington et al., 2014) dựa trên ma trận đồng xuất hiện:

X_{ij} = \text{số lần } w_j \text{ xuất hiện trong ngữ cảnh của } w_i

⸻

3.2 Hàm mục tiêu

J = \sum_{i,j} fX_{ij} \leftw_i^\top \tilde{w}_j + b_i + b_j - \log X_{ij} \right^2

Trong đó:

$$
fx =
$$

\begin{cases}
$x/x_{max}$^\alpha & x < x_{max} \\
1 & \text{otherwise}
\end{cases}

Khác với Word2Vec, GloVe khai thác thống kê toàn cục.

⸻

4. GPT: Transformer Tự Hồi quy

4.1 Cấu trúc tổng quan

GPT (Radford et al.) dựa trên kiến trúc Transformer từ bài báo của Ashish Vaswani et al. (2017).

Mô hình xác suất:

$P($w_1$,\dots,$w_T$)$ = $\prod$_{t=1}^{T} $P($w_t$ \mid w_{\lt t})$

⸻

4.2 Self-Attention

Với:

Q = XW_Q,\quad K = XW_K,\quad V = XW_V

Attention:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left\frac{QK^\top}{\sqrt{d_k}} \rightV Độ phức tạp: O(n^2 d) ⸻ 4.3 Hàm mất mát Cross-entropy:
$$

\mathcal${L} = - $\sum$_{t=1}^{T} $\log$ $P($w_t$ \mid w_{\lt t})

$$
GPT sinh văn bản theo hướng trái → phải (autoregressive). ⸻ 5. BERT: Transformer Hai chiều 5.1 Kiến trúc BERT (Devlin et al., 2018) sử dụng: •	Masked Language Modeling (MLM) •	Next Sentence Prediction (NSP) ⸻ 5.2 Masked Language Model Chọn tập vị trí M:
$$

\mathcal${L}_{MLM} = - $\sum$_{t \in M} $\log$ $P($w_t$ \mid w_{\setminus M})

$$
Khác GPT: •	GPT: dự đoán tương lai •	BERT: dùng cả trái và phải ⸻ 5.3 Biểu diễn Ngữ cảnh hóa Embedding giờ là hàm của toàn bộ câu: e_t = fw_1,\dots,w_T, t Không còn là ánh xạ cố định. ⸻ 6. So sánh Toán học Mô hình	Xác suất	Phạm vi ngữ cảnh	Embedding Word2Vec	P(w_O\mid w_I)	Cục bộ	Tĩnh
$$

GloVe	\log X_{ij}	Toàn cục	Tĩnh

$$
GPT	P(w_t \mid w_{\lt t})	Trái	Ngữ cảnh BERT	P(w_t\mid w_{\setminus M})	Hai chiều	Ngữ cảnh ⸻ 7. Phân tích Entropy Entropy chuỗi: H = - \sum P(w_1,\dots,w_T)\log P(w_1,\dots,w_T) GPT mô hình hóa trực tiếp: H = - \sum_{t} \log P(w_t \mid w_{\lt t}) Perplexity:
$$

\text{PPL} = 2^H

$$
Word2Vec/GloVe không tối ưu trực tiếp perplexity. ⸻ 8. So sánh Độ phức tạp Word2Vec: O(T c d) GloVe: O(|X|) Transformer: O(n^2 d) Trong đó n là độ dài chuỗi. ⸻ 9. Tiến hóa Mô hình Quá trình phát triển: 1.	Vector tĩnh (Word2Vec, GloVe) 2.	Transformer một chiều (GPT) 3.	Transformer hai chiều (BERT) Bước chuyển quan trọng nhất là self-attention. ⸻ 10. Kết luận Từ Word2Vec đến GPT và BERT cho thấy sự chuyển dịch: •	Từ mô hình cục bộ → mô hình toàn chuỗi •	Từ embedding tĩnh → embedding ngữ cảnh •	Từ ma trận đồng xuất hiện → mô hình xác suất chuỗi Toán học chuyển từ:
$$

v_w \in \mathbb{R}^d