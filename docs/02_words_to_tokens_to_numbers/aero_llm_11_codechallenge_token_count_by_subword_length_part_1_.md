
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
Phân tích Phân bố Độ dài Subword và Số lượng Token trong Bộ Tokenizer của GPT-4

(Dựa trên tài liệu đính kèm và mở rộng học thuật)

⸻

Tóm tắt

Bài viết này phân tích mối quan hệ giữa độ dài subword và số lượng token trong hệ tokenizer của GPT-4, dựa trên dữ liệu thực nghiệm từ tài liệu đính kèm. Thông qua mô hình hóa toán học và thống kê xác suất, chúng tôi làm rõ cách phân bố token ảnh hưởng đến hiệu năng mô hình Transformer. Bài viết mở rộng nền tảng lý thuyết của Byte Pair Encoding (BPE) và thảo luận tác động đến độ phức tạp tính toán trong kiến trúc Attention của OpenAI.

⸻

1. Giới thiệu

Trong các mô hình ngôn ngữ lớn (LLMs), tokenization quyết định cách văn bản được phân mảnh thành các đơn vị xử lý. Với GPT-4, tokenizer hoạt động ở byte-level BPE, nghĩa là mọi chuỗi Unicode được mã hóa thành các chuỗi byte trước khi thực hiện hợp nhất subword.

Giả sử một chuỗi văn bản đầu vào:

$$

$$

S = (c_1, c_2, ..., c_n)

$$

$$

Tokenizer ánh xạ thành chuỗi token:

$$

$$

T = (t_1, t_2, ..., t_m), \quad m \le n

$$

$$

Mỗi token có độ dài subword \ell((t_i).

$$
⸻ 2. Mô hình Toán học của Phân bố Độ dài Subword 2.1 Định nghĩa Gọi: •	V: tập từ vựng token •	|V|: kích thước từ vựng
$$

•	)\ell((t): độ dài ký tự (hoặc byte) của token t

$$
Phân bố xác suất theo độ dài:
$$

)P(L = k) = \frac{\mid \{t \in V : \ell((t) = k\}|}{|V|}

$$
⸻ 2.2 Kỳ vọng độ dài token Độ dài trung bình của token: ) \mathbb{E}[L] = \sum_{k=1}^{\infty} k \cdot P(L = k)
$$

Nếu phân bố lệch phải (right-skewed), phần lớn token sẽ có độ dài nhỏ (1–4 byte), nhưng tồn tại một số token dài hơn đại diện cho cụm từ phổ biến.

⸻

2.3 Hàm phân bố tích lũy

$$

$$

Fk = P(L \le k)

$$

$$

Giúp đánh giá tỷ lệ token ngắn chiếm bao nhiêu phần trăm trong toàn bộ từ vựng.

⸻

3. Phân tích Thực nghiệm từ Tài liệu

Dựa trên dữ liệu đính kèm:
	•	Token 1–3 ký tự chiếm tỷ lệ cao nhất.
	•	Token dài (>10 ký tự) rất hiếm.
	•	Phân bố gần giống hàm mũ giảm dần.

Ta có thể xấp xỉ:

$P(L = k)$ $\approx$ Ce^{-\lambda k}

Trong đó:
	•	C: hằng số chuẩn hóa
	•	\lambda > 0: hệ số suy giảm

Chuẩn hóa:

$$
\sum_{k=1}^{\infty} Ce^{-\lambda k} = 1
$$

$$
C = 1 - e^{-\lambda}
$$

⸻

4. Ảnh hưởng đến Độ Phức tạp Attention

Trong kiến trúc Transformer của OpenAI, self-attention có độ phức tạp:

$O(m^2)$

Trong đó m là số token sau khi token hóa.

$$
Nếu độ dài trung bình token là \mathbb{E}[L], thì:
$$

$$
m \approx \frac{n}{\mathbb{E}[L]}
$$

$$
Do đó chi phí tính toán:
$$

$O($\le$)$ft($\le$ft(\frac{n}{$\mathbb${E}[L]}\right)^2\right)

$$
Tokenizer tối ưu sẽ:
$$

•	Tăng \mathbb{E}[L]

$$
•	Giảm m •	Giảm chi phí attention ⸻ 5. Mối quan hệ với Entropy Thông tin Entropy của phân bố token:
$$

$$
HT = - \sum_{t \in V} P(t)\log P(t)
$$

$$
Nếu token ngắn quá nhiều: •	Entropy cao •	Chuỗi dài •	Attention tốn tài nguyên Nếu token quá dài: •	Vocabulary lớn •	Khó tổng quát hóa Do đó BPE tối ưu cân bằng giữa hai yếu tố này. ⸻ 6. Mô hình Zipf và Phân bố Tần suất Tần suất token thường tuân theo luật Zipf:
$$

fr \propto \frac{1}{r^\alpha}

$$
Trong đó: •	r: thứ hạng token
$$

$$
•	\alpha \approx 1
$$

$$
Kết hợp Zipf và phân bố độ dài: •	Token phổ biến thường ngắn •	Token hiếm thường dài ⸻ 7. So sánh với Các Phương pháp Khác Phương pháp	Phân bố độ dài	Tính ổn định	Chi phí Word-level	Không đồng đều	OOV cao	Trung bình
$$

Character-level	L = 1	Ổn định	Rất cao

$$
BPE	Phân bố mũ	Cân bằng	Tối ưu Unigram LM	Xác suất	Linh hoạt	Cao ⸻ 8. Hệ quả Đối với Huấn luyện Loss function:
$$

$\mathcal${L} = - $\sum$_{i=1}^{m} $\log$ P($t_i$  \mid  t_{\lt i})

$$
Vì m phụ thuộc tokenizer nên: •	Tokenizer ảnh hưởng trực tiếp đến giá trị loss •	Ảnh hưởng tốc độ hội tụ •	Ảnh hưởng khả năng tổng quát hóa ⸻ 9. Thảo luận Kết quả cho thấy: •	Phân bố độ dài token có dạng suy giảm hàm mũ •	Độ dài trung bình là tham số then chốt •	Tokenizer quyết định cấu trúc không gian xác suất đầu vào Trong tương lai, adaptive tokenization có thể tối ưu theo ngữ cảnh thay vì cố định từ vựng. ⸻ 10. Kết luận Phân tích cho thấy:
$$

m \sim \frac{n}{\mathbb{E}[L]}