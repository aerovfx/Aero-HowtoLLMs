Dưới đây là bài viết khoa học được xây dựng dựa trên nội dung tài liệu “Code Challenge: Perplexing Perplexities”, kết hợp mở rộng lý thuyết từ các công trình nền tảng về mô hình ngôn ngữ và lý thuyết thông tin, trình bày dưới dạng Markdown và có bổ sung các công thức toán học minh họa.

⸻

Perplexing Perplexities: Phân Tích Sâu về Độ Rối trong Đánh Giá Mô Hình Ngôn Ngữ

Tóm tắt

Perplexity là chỉ số chuẩn trong đánh giá mô hình ngôn ngữ xác suất. Tuy nhiên, cách tính và diễn giải perplexity thường gây hiểu nhầm, đặc biệt khi thay đổi tokenization, độ dài chuỗi hoặc miền dữ liệu. Bài viết này phân tích sâu bản chất toán học của perplexity, các trường hợp “nghịch lý” (perplexing cases), và giới hạn khi áp dụng trong các mô hình ngôn ngữ hiện đại. Nền tảng lý thuyết dựa trên công trình của Claude Shannon, Christopher D. Manning và Yoshua Bengio.

⸻

1. Cơ sở Toán học của Perplexity

1.1 Mô hình ngôn ngữ xác suất

Với chuỗi từ:

w_1, w_2, \dots, w_T

Xác suất toàn chuỗi:

P(w_1^T)
=
\prod_{t=1}^{T}
P(w_t | w_1^{t-1})

Log-likelihood trung bình:

\ell
=
\frac{1}{T}
\sum_{t=1}^{T}
\log P(w_t | w_1^{t-1})

⸻

1.2 Entropy và Cross-Entropy

Entropy:

H(p)
=
-
\sum_x
p(x)\log p(x)

Cross-entropy thực nghiệm:

\hat{H}
=
-
\frac{1}{T}
\sum_{t=1}^{T}
\log P(w_t | context)

⸻

1.3 Định nghĩa Perplexity

PP
=
\exp(\hat{H})
=
\exp
\left(
-
\frac{1}{T}
\sum_{t=1}^{T}
\log P(w_t | context)
\right)

Hoặc:

PP = e^{-\ell}

⸻

2. Những Trường Hợp “Perplexing”

2.1 Perplexity thấp nhưng văn bản kém tự nhiên

Perplexity đo:

P(data | model)

Không đo:
	•	Tính sáng tạo
	•	Tính logic
	•	Tính đúng sự thật

Một mô hình có thể tối ưu likelihood nhưng sinh văn bản lặp lại.

⸻

2.2 Phụ thuộc Tokenization

Giả sử cùng một câu:
	•	Tokenization A → T_A token
	•	Tokenization B → T_B token

Vì:

PP =
\exp
\left(
\frac{\mathcal{L}}{T}
\right)

Nếu T thay đổi → PP thay đổi.

Do đó:

PP_A \neq PP_B

Ngay cả khi mô hình tương đương về xác suất chuỗi.

⸻

2.3 Ảnh hưởng Độ Dài Chuỗi

Với chuỗi rất dài:

\hat{H}
\rightarrow
H

Theo luật số lớn.

Với chuỗi ngắn:

Var(\hat{H})
=
\frac{\sigma^2}{T}

Perplexity không ổn định khi T nhỏ.

⸻

3. Phân tích Thống kê

3.1 Sai số chuẩn

Nếu entropy ước lượng:

SE(H)
=
\frac{\sigma}{\sqrt{T}}

Khoảng tin cậy 95%:

\hat{H}
\pm
1.96 \cdot SE(H)

Từ đó:

PP_{CI}
=
\exp(\hat{H} \pm 1.96 SE)

⸻

3.2 Liên hệ với KL-Divergence

H(p,q)
=
H(p)
+
D_{KL}(p||q)

Perplexity:

PP
=
\exp(H(p) + D_{KL}(p||q))

Tối thiểu khi:

q = p

⸻

4. Phân tích Các Tình Huống Code Challenge

Từ bài thực hành:

Trường hợp 1: Dự đoán đều

Nếu:

P(w) = \frac{1}{V}

Thì:

H = \log V

PP = V

→ Perplexity bằng kích thước từ vựng.

⸻

Trường hợp 2: Dự đoán hoàn hảo

Nếu:

P(w_t) = 1

H = 0

PP = 1

⸻

Trường hợp 3: Sai hoàn toàn

Nếu:

P(w_t) \rightarrow 0

H \rightarrow \infty

PP \rightarrow \infty

⸻

5. Perplexity và Softmax

Trong mô hình neural:

z_t = W h_t

P(w_t | context)
=
\frac{\exp(z_{t,w})}
{\sum_j \exp(z_{t,j})}

Cross-entropy loss:

\mathcal{L}
=
-
\sum_t
\log P(w_t)

Perplexity:

PP
=
\exp
\left(
\frac{\mathcal{L}}{T}
\right)

⸻

6. Perplexity trong LLMs Hiện đại

Trong các mô hình lớn:
	•	Instruction tuning
	•	RLHF
	•	Fine-tuning theo nhiệm vụ

Có thể xảy ra:

PP_{instruction}
>
PP_{base}

Nhưng chất lượng hội thoại tốt hơn.

Điều này cho thấy perplexity không đo được alignment với người dùng.

⸻

7. Phân tích Giới hạn Lý thuyết

Perplexity tối ưu hóa:

\min_\theta
D_{KL}(p||q_\theta)

Không tối ưu hóa:
	•	Utility
	•	Human preference
	•	Task-specific reward

Theo nguyên lý Goodhart:

Khi một chỉ số trở thành mục tiêu tối ưu, nó có thể mất đi ý nghĩa ban đầu.

⸻

8. Kết luận

Perplexity là:

PP = e^{H}

Một thước đo chặt chẽ dựa trên lý thuyết thông tin.

Nó hữu ích để:
	•	So sánh mô hình xác suất
	•	Theo dõi quá trình huấn luyện
	•	Phát hiện overfitting

Tuy nhiên:
	•	Phụ thuộc tokenization
	•	Không đo ngữ nghĩa sâu
	•	Không phản ánh alignment

Do đó, perplexity nên được dùng như chỉ số cơ sở, kết hợp với đánh giá định tính và task-specific metrics để đánh giá toàn diện mô hình ngôn ngữ.

⸻

Tài liệu tham khảo
	1.	Shannon, C. E. (1948). A Mathematical Theory of Communication.
	2.	Manning, C. D., & Schütze, H. (1999). Foundations of Statistical Natural Language Processing.
	3.	Bengio, Y. et al. (2003). A Neural Probabilistic Language Model.
	4.	Jurafsky, D., & Martin, J. H. (Speech and Language Processing).
	5.	Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
