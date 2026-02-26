
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
Biến thể Từ vựng trong Tokenizer của Claude:

Phân tích Hình thức, Phân bố Xác suất và Ảnh hưởng đến Biểu diễn Ngữ nghĩa

⸻

Tóm tắt

Dựa trên tài liệu đính kèm “Word Variations in Claude Tokenizer”, bài viết này phân tích cách tokenizer của mô hình Claude xử lý các biến thể từ vựng (word variations) như tiền tố, hậu tố, chữ hoa–thường và hình thái học. Chúng tôi xây dựng mô hình toán học cho phân rã subword, phân tích phân bố xác suất token, và đánh giá ảnh hưởng đến entropy, tỷ lệ nén và chi phí self-attention trong Transformer. Bài viết cũng so sánh với tokenizer của BERT và các phương pháp dựa trên BPE.

⸻

1. Giới thiệu

Tokenizer là hàm ánh xạ:

\mathcal{T}: \Sigma^* \rightarrow V^*

Trong đó:
	•	\Sigma: bảng ký tự
	•	V: tập token
	•	V^*: chuỗi token

Một từ có nhiều biến thể hình thái:

w_k = r + s_k

với:
	•	r: gốc từ (root)
	•	s_k: hậu tố (suffix)

Tokenizer subword sẽ phân rã:

\mathcal{T}(w_k) = (r, s_k)

Thay vì xem mỗi biến thể là một token độc lập.

⸻

2. Mô hình Toán học của Biến thể Từ

Giả sử một tập biến thể:

W = \{w_1, w_2, \dots, w_K\}

Trong đó:

w_k = r + s_k

Nếu xác suất xuất hiện:

P(w_k)

thì xác suất của root:

P(r) = \sum_{k=1}^{K} P(w_k)

Tokenizer hiệu quả sẽ học:

P(r) \gg P(w_k)

⸻

3. Entropy Trước và Sau Phân rã

3.1 Entropy ở mức từ

H_W = -\sum_{k=1}^{K} P(w_k)\log P(w_k)

⸻

3.2 Entropy ở mức subword

Giả sử tách thành root và suffix:

H_{sub} = -P(r)\log P(r) - \sum_{k} P(s_k)\log P(s_k)

Vì:

P(r) = \sum_k P(w_k)

nên:

H_{sub} \le H_W

(giảm entropy nhờ gom tần suất về root chung).

⸻

4. Compression Ratio và Độ dài Chuỗi

Giả sử:
	•	Văn bản có n ký tự
	•	Sau tokenization có m token

Compression ratio:

R = \frac{n}{m}

Nếu tokenizer tái sử dụng root cho nhiều biến thể:

m \downarrow \Rightarrow R \uparrow

Chi phí attention:

O(m^2)

Thay:

O\left(\frac{n^2}{R^2}\right)

⸻

5. Phân bố Zipf trong Biến thể Từ

Theo George Kingsley Zipf:

f(r) \propto \frac{1}{r^\alpha}

Root thường có thứ hạng thấp (tần suất cao).
Suffix có phân bố đuôi dài.

Phân rã subword làm thay đổi hệ số:

\alpha_{sub} \neq \alpha_{word}

⸻

6. Mô hình Xác suất Hình thái

Giả sử xác suất sinh từ:

P(w_k) = P(r)P(s_k \mid r)

Log-likelihood:

\log P(w_k) = \log P(r) + \log P(s_k \mid r)

Tokenizer subword xấp xỉ phân tích hình thái này.

⸻

7. So sánh với Tokenizer của BERT

Tokenizer WordPiece trong BERT tối ưu:

\arg\max_{s_1,\dots,s_m} \prod_i P(s_i)

Trong khi các tokenizer hiện đại (như Claude) tối ưu theo tần suất byte hoặc subword linh hoạt hơn.

⸻

8. Ảnh hưởng đến Embedding

Embedding:

E: V \rightarrow \mathbb{R}^d

Nếu các biến thể chia sẻ root:

e(w_k) \approx e(r) + e(s_k)

Sai số:

\delta_k = \| e(w_k) - (e(r)+e(s_k)) \|_2

Tối ưu hóa:

\min \sum_k \delta_k^2

Điều này cải thiện khả năng tổng quát hóa.

⸻

9. Ảnh hưởng đến Huấn luyện

Gradient của token hiếm:

\nabla L(w_k)

Nếu chia thành root và suffix:

\nabla L(r) = \sum_k \nabla L(w_k)

→ Tăng ổn định gradient.

⸻

10. Phân tích Đa ngôn ngữ

Trong ngôn ngữ chắp dính:

|s_k| \uparrow

Tokenizer phải cân bằng giữa:
	•	Giữ nguyên toàn bộ từ
	•	Chia thành nhiều subword

Tối ưu hóa đa mục tiêu:

\min \left( \frac{n^2}{R^2} + \lambda |V| \right)

⸻

11. Thảo luận

Biến thể từ vựng tạo ra:
	•	Đuôi dài trong phân bố token
	•	Tăng entropy nếu không tách

Tokenizer subword hiệu quả:
	1.	Gom tần suất vào root
	2.	Giảm entropy
	3.	Tăng compression ratio
	4.	Ổn định huấn luyện

Các hệ do Anthropic, OpenAI và Google phát triển đều áp dụng nguyên tắc này.

⸻

12. Kết luận

Phân rã biến thể từ có thể được mô hình hóa:

P(w_k) = P(r)P(s_k \mid r)

Entropy giảm khi:

H_{sub} \le H_W

Compression ratio:

R = \frac{n}{m}

Chi phí attention:

O\left(\frac{n^2}{R^2}\right)

Tokenizer hiện đại tận dụng cấu trúc hình thái để:
	•	Nén thông tin
	•	Giảm độ dài chuỗi
	•	Tăng tính tổng quát hóa

⸻

Tài liệu tham khảo
	1.	Zipf, G. K. (1935). The Psycho-Biology of Language.
	2.	Shannon, C. (1948). A Mathematical Theory of Communication.
	3.	Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	4.	Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
	5.	Vaswani et al. (2017). Attention Is All You Need.
	6.	Kudo & Richardson (2018). SentencePiece.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
