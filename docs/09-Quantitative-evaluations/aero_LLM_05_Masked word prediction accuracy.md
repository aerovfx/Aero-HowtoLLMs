
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [09 Quantitative evaluations](../index.md)

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
Đánh giá độ chính xác trong bài toán dự đoán từ bị che (Masked Word Prediction Accuracy)

Phân tích định lượng, công thức toán học và thách thức thực nghiệm

⸻

Tóm tắt

Bài báo này phân tích cơ chế đánh giá độ chính xác dự đoán từ bị che (Masked Word Prediction Accuracy) trong các mô hình ngôn ngữ hiện đại như OpenAI GPT, Google BERT và các biến thể Transformer. Dựa trên tài liệu đính kèm, chúng tôi mở rộng bằng cách bổ sung các nguồn học thuật như Jacob Devlin et al. (2018), Ashish Vaswani et al. (2017) và Tomas Mikolov et al. (2013).

Bài viết trình bày:
	•	Cơ chế Masked Language Modeling (MLM)
	•	Công thức toán học của accuracy, cross-entropy, perplexity
	•	So sánh accuracy và perplexity
	•	Thách thức khi đánh giá định lượng

⸻

1. Giới thiệu

Masked Language Modeling (MLM) là một phương pháp huấn luyện trong đó một số token trong câu được thay thế bằng ký hiệu [MASK], và mô hình phải dự đoán token gốc.

Ví dụ:

“The cat sits on the [MASK].”
→ Đáp án đúng: “mat”

Phương pháp này được phổ biến rộng rãi trong kiến trúc Transformer của Google thông qua mô hình BERT (2018).

⸻

2. Cơ sở toán học của Masked Word Prediction

Giả sử:
	•	Câu đầu vào:
X = (x_1, x_2, ..., x_n)
	•	Tập chỉ số các token bị che:
M \subset \{1, 2, ..., n\}

Mô hình học xác suất có điều kiện:

P(x_i \mid X_{\setminus M})

Trong đó X_{\setminus M} là chuỗi đầu vào đã thay các vị trí trong M bằng [MASK].

⸻

3. Định nghĩa Accuracy trong MLM

3.1 Accuracy đơn giản

Nếu có N token bị che:

Accuracy = \frac{1}{N} \sum_{i \in M} \mathbf{1}(\hat{x}_i = x_i)

Trong đó:
	•	\hat{x}_i = \arg\max P(x_i \mid X_{\setminus M})
	•	\mathbf{1}(\cdot) là hàm chỉ báo

⸻

3.2 Top-k Accuracy

Trong thực tế, ta thường sử dụng Top-k accuracy:

Top\text{-}k = \frac{1}{N} \sum_{i \in M} \mathbf{1}(x_i \in \text{Top-}k(\hat{P}_i))

Điều này đặc biệt quan trọng khi:
	•	Từ vựng lớn (30k–100k tokens)
	•	Nhiều từ có xác suất gần nhau

⸻

4. Liên hệ với Cross-Entropy và Perplexity

4.1 Cross-Entropy Loss

\mathcal{L} = - \frac{1}{N} \sum_{i \in M} \log P(x_i \mid X_{\setminus M})

Cross-entropy đo mức “bất ngờ” của mô hình trước dữ liệu thực.

⸻

4.2 Perplexity

Perplexity được định nghĩa:

PP = e^{\mathcal{L}}

Hoặc:

PP = \exp\left(- \frac{1}{N} \sum_{i=1}^{N} \log P(x_i) \right)

Perplexity càng thấp → mô hình càng tốt.

⸻

5. So sánh Accuracy và Perplexity

Tiêu chí	Accuracy	Perplexity
Dễ hiểu	✔	✖
Nhạy với xác suất	✖	✔
Phù hợp cho MLM	✔	✔
Phản ánh độ tự tin	✖	✔

Ví dụ:

Giả sử mô hình dự đoán đúng nhưng với xác suất thấp:

P(x_i) = 0.51

→ Accuracy = 100%
→ Cross-entropy cao
→ Perplexity cao

Do đó accuracy không phản ánh đầy đủ chất lượng mô hình.

⸻

6. Thách thức khi đánh giá Masked Accuracy

6.1 Tokenization

Các mô hình như BERT sử dụng WordPiece:

Ví dụ:

playing → play + ##ing

Điều này tạo ra vấn đề:
	•	Dự đoán đúng 1 phần của từ?
	•	Tính accuracy theo token hay theo word?

⸻

6.2 Vocabulary Bias

Nếu từ vựng lớn:

P_{\text{random}} = \frac{1}{|V|}

Với |V| = 50,000:

P_{\text{random}} = 0.00002

Accuracy cao hơn mức này nhiều lần mới có ý nghĩa thống kê.

⸻

6.3 Distribution Shift

Nếu tập test khác domain train:

D_{train} \neq D_{test}

Accuracy có thể giảm mạnh dù mô hình vẫn tốt về mặt xác suất tổng thể.

⸻

7. Mối liên hệ với Transformer

Kiến trúc Transformer do Ashish Vaswani et al. đề xuất có cơ chế self-attention:

Attention(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

MLM tận dụng self-attention hai chiều để dự đoán token bị che.

⸻

8. Phân tích thống kê độ tin cậy của Accuracy

Nếu số token bị che là N, accuracy ước lượng là:

\hat{p} = \frac{k}{N}

Sai số chuẩn:

SE = \sqrt{\frac{\hat{p}(1-\hat{p})}{N}}

Khoảng tin cậy 95%:

\hat{p} \pm 1.96 \cdot SE

Điều này quan trọng khi so sánh hai mô hình.

⸻

9. So sánh MLM và Next Token Prediction

MLM (BERT) vs Autoregressive (GPT):

P(x_1,...,x_n) = \prod_{t=1}^{n} P(x_t \mid x_{<t})

Khác với:

P(x_i \mid X_{\setminus M})

Do đó accuracy trong MLM không tương đương trực tiếp với perplexity trong GPT.

⸻

10. Kết luận

Masked Word Prediction Accuracy:
	•	Dễ hiểu
	•	Phù hợp cho đánh giá nội bộ
	•	Không phản ánh đầy đủ phân phối xác suất

Nên kết hợp:
	•	Accuracy
	•	Cross-entropy
	•	Perplexity
	•	Calibration metrics

⸻

Tài liệu tham khảo
	1.	Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	2.	Vaswani, A. et al. (2017). Attention is All You Need.
	3.	Mikolov, T. et al. (2013). Efficient Estimation of Word Representations.
	4.	Jurafsky & Martin. Speech and Language Processing.
	5.	Goodfellow et al. (2016). Deep Learning.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
