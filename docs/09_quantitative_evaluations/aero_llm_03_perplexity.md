
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [09 quantitative evaluations](index.md)

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
# Perplexity trong Mô Hình Ngôn Ngữ: Cơ Sở Toán Học, Diễn Giải và Giới Hạn

Tóm tắt

Perplexity (độ rối) là thước đo chuẩn để đánh giá mô hình ngôn ngữ xác suất. Nó phản ánh mức độ “bất ngờ” trung bình của mô hình khi dự đoán một chuỗi từ. Bài viết này phân tích nền tảng toán học của perplexity, mối liên hệ với entropy và cross-entropy, cũng như các hạn chế khi sử dụng perplexity để đánh giá mô hình ngôn ngữ hiện đại. Nội dung mở rộng từ các công trình của Claude Shannon, Christopher D. Manning và Yoshua Bengio.

⸻

1. Giới thiệu

Mô hình ngôn ngữ xác suất ước lượng:

P($w_1$, $w_2$, ..., $w_T$)

Theo quy tắc chuỗi:

P($w_1$^T)

$$
= \prod_{t=1}^{T}
$$

P($w_t$ | $w_1$^{t-1})

Mục tiêu là tối đa hóa xác suất chuỗi văn bản trong tập kiểm tra.

⸻

2. Entropy và Cross-Entropy

2.1 Entropy (Shannon, 1948)

Entropy của phân phối p$x$:

H$p$

$$
=
$$

-
$\sum$_x

$$
px\log px
$$

Đơn vị: bits (log base 2) hoặc nats (log base e).

⸻

2.2 Cross-Entropy

Nếu mô hình ước lượng phân phối q$x$:

H(p, q)

$$
=
$$

-
$\sum$_x

$$
px\log qx
$$

Trong thực nghiệm, ta dùng ước lượng:

\hat{H}

$$
=
$$

-
\frac{1}{T}

$$
\sum_{t=1}^{T}
$$

$\log$ P_\theta($w_t$ | $w_1$^{t-1})

⸻

3. Định nghĩa Perplexity

Perplexity được định nghĩa là:

$$
PP =
$$

\exp$\hat{H}$

Hoặc:

$$
PP =
$$

\exp

$$
\left(
$$

-
\frac{1}{T}

$$
\sum_{t=1}^{T}
$$

$\log$ P($w_t$ | $w_1$^{t-1})
\right)

Nếu log base 2:

$$
PP = 2^{H}
$$

⸻

4. Diễn giải Trực quan

Perplexity có thể hiểu là:

Số lượng lựa chọn trung bình mà mô hình “phân vân” tại mỗi bước.

Ví dụ:

$$
•	Nếu PP = 10 → mô hình như đang chọn trong 10 từ khả dĩ. •	Nếu PP = 1 → dự đoán hoàn hảo.
$$

⸻

5. Mối liên hệ với Likelihood

Log-likelihood trung bình:

$\ell(

$$
=
$$

\frac{1}{T}

)$
$\sum$_{t=1}^{T}

$$
\log P(w_t  \mid  w_1^{t-1}) Khi đó: PP = e^{-\ell(}
$$

)$

Giảm perplexity ⇔ tăng log-likelihood.

⸻

6. Ví dụ Minh họa

Giả sử mô hình dự đoán xác suất trung bình:

P(w_t  \mid  context) = 0.2

Khi đó:

\hat{H} = -\log(0.2)

$$
PP = \exp-\log 0.2 = \frac{1}{0.2} = 5 ⸻ 7. Perplexity và Mô hình N-gram Trong mô hình n-gram: P(w_t  \mid  w_{t-n+1}^{t-1}) Perplexity giảm khi: •	n tăng •	dữ liệu huấn luyện lớn hơn Tuy nhiên: n \rightarrow lớn \Rightarrow Data\ sparsity ⸻ 8. Perplexity trong Mô hình Neural Với mạng nơ-ron: z_t = W h_t P(w_t  \mid  context)
$$

=

$$
\text{softmax}z_t Cross-entropy loss: \mathcal{L}
$$

=

$$
- \sum_t \log P(w_t  \mid  context) Perplexity:
$$

PP =

$$
\exp
$$

$\le$ft(

\frac{\mathcal{L}}{T}

$$
\right) ⸻ 9. Hạn chế của Perplexity 9.1 Không phản ánh chất lượng sinh văn bản Perplexity thấp ≠ văn bản tự nhiên hơn. 9.2 Phụ thuộc tokenization Nếu thay đổi cách tách từ: T \text{ thay đổi} \Rightarrow PP \text{ thay đổi} Không thể so sánh trực tiếp giữa các tokenizer khác nhau. ⸻ 9.3 Không đo được hiểu ngữ nghĩa Perplexity chỉ đo: P(data) Không đo: •	Độ logic •	Tính đúng sự thật •	Sáng tạo ⸻ 10. Phân tích Giới hạn Thống kê Perplexity thực nghiệm: \hat{PP}
$$

=

$$
\exp\hat{H} Sai số chuẩn của entropy: SEH
$$

=

$$
\frac{\sigma}{\sqrt{T}} Khi T nhỏ → phương sai cao → PP không ổn định. ⸻ 11. Liên hệ với KL-Divergence H(p,q)
$$

=

$$
Hp + D_{KL}(p||q) Do đó: PP
$$

=

$$
\exp(H(p) + D_{KL}(p||q)) Perplexity tối thiểu khi: q = p ⸻ 12. Perplexity trong LLMs Hiện đại Trong mô hình lớn: •	Zero-shot evaluation •	Few-shot evaluation •	Instruction tuning Perplexity thường dùng để: •	So sánh checkpoint •	Phát hiện overfitting •	Đánh giá hội tụ Tuy nhiên, với mô hình instruction-tuned: Perplexity có thể tăng nhưng chất lượng hội thoại tốt hơn. ⸻ 13. Kết luận Perplexity là thước đo toán học chặt chẽ dựa trên entropy và likelihood: PP = e^{H}
$$

