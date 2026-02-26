
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [02 Words to tokens to numbers](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
Phân tích Số lượng Ký tự trong Token của BERT:

Mô hình Thống kê, Entropy và Ảnh hưởng đến Độ phức tạp Transformer

⸻

Tóm tắt

Bài viết này phân tích số lượng ký tự cấu thành mỗi token trong bộ tokenizer của BERT do Google phát triển, dựa trên dữ liệu từ tài liệu đính kèm. Chúng tôi xây dựng mô hình thống kê cho phân bố độ dài token, ước lượng entropy hệ subword, và phân tích tác động của độ dài ký tự đến độ phức tạp tính toán trong kiến trúc Transformer. Kết quả cho thấy phân bố độ dài token có xu hướng lệch phải (right-skewed), gần với phân bố hình học hoặc log-linear, phản ánh sự cân bằng giữa kích thước từ vựng và độ dài chuỗi đầu vào.

⸻

1. Giới thiệu

Trong BERT, văn bản đầu vào được token hóa bằng thuật toán WordPiece thành các subword token:

S = (w_1, w_2, ..., w_n)

T = (t_1, t_2, ..., t_m)

Với:

m \ge n

Mỗi token t_i có độ dài ký tự:

\ell(t_i)

Mục tiêu nghiên cứu:
	1.	Phân bố xác suất của \ell(t)
	2.	Độ dài trung bình token
	3.	Ảnh hưởng đến chi phí self-attention

⸻

2. Mô hình Thống kê Phân bố Độ dài Token

2.1 Định nghĩa

Gọi:
	•	V: tập từ vựng BERT
	•	|V| \approx 30{,}000
	•	N_k: số token có độ dài ký tự bằng k

Xác suất:

P(L = k) = \frac{N_k}{|V|}

Chuẩn hóa:

\sum_{k=1}^{K_{\max}} P(L=k) = 1

⸻

2.2 Mô hình Hình học (Geometric Approximation)

Quan sát thực nghiệm cho thấy:

N_k \approx Ae^{-\lambda k}

Suy ra:

P(L=k) = (1-q)q^{k-1}

Trong đó:

q = e^{-\lambda}

Đây là phân bố hình học rời rạc.

⸻

2.3 Kỳ vọng và Phương sai

Kỳ vọng:

\mathbb{E}[L] = \frac{1}{1-q}

Phương sai:

\mathrm{Var}(L) = \frac{q}{(1-q)^2}

Nếu q \to 1, phân bố có đuôi dài hơn (nhiều token dài).

⸻

3. Ảnh hưởng đến Độ dài Chuỗi Văn bản

Giả sử văn bản có tổng số ký tự n.

Số token trung bình:

m = \frac{n}{\mathbb{E}[L]}

Self-attention trong Transformer encoder:

O(m^2)

Thay vào:

O\left(\left(\frac{n}{\mathbb{E}[L]}\right)^2\right)

Khi \mathbb{E}[L] \uparrow, chi phí giảm.

⸻

4. Entropy của Hệ Token

Entropy theo phân bố độ dài:

H_L = - \sum_{k} P(L=k)\log P(L=k)

Thay phân bố hình học:

H_L = - \sum_{k=1}^{\infty} (1-q)q^{k-1} \log[(1-q)q^{k-1}]

Rút gọn:

H_L = -\log(1-q) - \frac{q}{1-q}\log q

Entropy càng lớn → độ đa dạng độ dài càng cao.

⸻

5. Quan hệ với Luật Zipf

Tần suất token thường tuân theo:

f(r) \propto \frac{1}{r^\alpha}

Trong đó:
	•	r: thứ hạng token
	•	\alpha \approx 1

Token ngắn thường:
	•	Có tần suất cao
	•	Ở thứ hạng thấp

Do đó tồn tại tương quan nghịch:

\ell(t) \propto \log r

⸻

6. Ảnh hưởng đến Embedding Matrix

Embedding:

E: V \rightarrow \mathbb{R}^d

Ma trận embedding:

W \in \mathbb{R}^{|V| \times d}

Bài toán tối ưu:

\min_{V} \left( \mathbb{E}[m] + \lambda |V| \right)

Trong đó:
	•	\mathbb{E}[m]: số token trung bình
	•	|V|: kích thước từ vựng
	•	\lambda: hệ số cân bằng

⸻

7. So sánh với Character-level Modeling

Mô hình	Độ dài trung bình	OOV	Chi phí
Character-level	1	Không	Rất cao
Word-level	Lớn	Cao	Trung bình
WordPiece	Trung bình	Thấp	Tối ưu

Nếu xử lý ở mức ký tự:

m = n

Chi phí:

O(n^2)

WordPiece giảm:

m = \frac{n}{\mathbb{E}[L]}

⸻

8. Thảo luận

Dữ liệu thực nghiệm cho thấy:
	•	Phần lớn token có độ dài nhỏ (1–5 ký tự)
	•	Token dài tồn tại nhưng ít
	•	Phân bố có đuôi nhẹ (mild heavy-tail)

Điều này phản ánh:
	•	Sự cân bằng giữa khả năng tổng quát hóa và độ nén
	•	Tối ưu hóa thực nghiệm hơn là lý thuyết thuần túy

⸻

9. Kết luận

Phân bố độ dài ký tự của token trong BERT có thể mô hình hóa gần đúng bằng phân bố hình học:

P(L=k) \sim q^{k-1}

Tác động trực tiếp đến:

m = \frac{n}{\mathbb{E}[L]}

\text{Attention Cost} \sim O(m^2)

H_L = - \sum P(L)\log P(L)

Thiết kế tokenizer là bài toán tối ưu đa mục tiêu giữa:
	•	Kích thước từ vựng
	•	Độ dài chuỗi
	•	Entropy thông tin
	•	Chi phí tính toán

⸻

Tài liệu tham khảo
	1.	BERT – Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	2.	Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
	3.	Vaswani et al. (2017). Attention Is All You Need.
	4.	Shannon, C. (1948). A Mathematical Theory of Communication.
	5.	Jurafsky & Martin. Speech and Language Processing.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
