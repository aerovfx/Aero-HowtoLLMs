
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
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Đánh Giá Hộp Đen (Black-box Evaluations) trong Mô Hình Ngôn Ngữ Lớn](aero_LLM_016_Black box evals.md) | [Xem bài viết →](aero_LLM_016_Black box evals.md) |
| [Red Teaming: Đội Đỏ và Thử Nghiệm Đối Kháng trong AI Safety](aero_LLM_017_Red-teaming.md) | [Xem bài viết →](aero_LLM_017_Red-teaming.md) |
| [Độ Chính Xác, Tính Mạch Lạc và Sự Phù Hợp trong Đánh Giá Mô Hình Ngôn Ngữ](aero_LLM_018_Accuracy, coherence, and relevance.md) | [Xem bài viết →](aero_LLM_018_Accuracy, coherence, and relevance.md) |
| [Phân Phối Của Các Kích Hoạt Trạng Thái Ẩn Trong Mô Hình Ngôn Ngữ](aero_LLM_019_Distributions of hidden-state activations.md) | [Xem bài viết →](aero_LLM_019_Distributions of hidden-state activations.md) |
| [Hứa Hẹn và Thách Thức của Đánh Giá Định Lượng trong Mô Hình Học Máy](aero_LLM_01_Promises and challenges of quantitative evaluations.md) | [Xem bài viết →](aero_LLM_01_Promises and challenges of quantitative evaluations.md) |
| [Bản Đồ Nhiệt Của Token Cho Cân Nhắc Định Tính (Text Heatmaps)](aero_LLM_020_Heatmaps of tokens for qualitative inspection.md) | [Xem bài viết →](aero_LLM_020_Heatmaps of tokens for qualitative inspection.md) |
| [Thử Thách Lập Trình: Trực Quan Hóa Dự Đoán Đơn Token](aero_LLM_021_CodeChallenge Visualize single-token predictions.md) | [Xem bài viết →](aero_LLM_021_CodeChallenge Visualize single-token predictions.md) |
| [Các Vấn Đề Số Học trong Logits và Softmax: Phân Tích Toán Học và Giải Pháp Ổn Định](aero_LLM_02_Numerical issues in logits and softmax.md) | [Xem bài viết →](aero_LLM_02_Numerical issues in logits and softmax.md) |
| [Perplexity trong Mô Hình Ngôn Ngữ: Cơ Sở Toán Học, Diễn Giải và Giới Hạn](aero_LLM_03_Perplexity.md) | [Xem bài viết →](aero_LLM_03_Perplexity.md) |
| 📌 **[aero_LLM_04_CodeChallenge Perplexing perplexities.md](aero_LLM_04_CodeChallenge Perplexing perplexities.md)** | [Xem bài viết →](aero_LLM_04_CodeChallenge Perplexing perplexities.md) |
| [aero_LLM_05_Masked word prediction accuracy.md](aero_LLM_05_Masked word prediction accuracy.md) | [Xem bài viết →](aero_LLM_05_Masked word prediction accuracy.md) |
| [aero_LLM_06_HellaSwag.md](aero_LLM_06_HellaSwag.md) | [Xem bài viết →](aero_LLM_06_HellaSwag.md) |
| [aero_LLM_07_Import large models using bitsandbytes.md](aero_LLM_07_Import large models using bitsandbytes.md) | [Xem bài viết →](aero_LLM_07_Import large models using bitsandbytes.md) |
| [aero_LLM_08_CodeChallenge HellaSwag evals in two models (part 1).md](aero_LLM_08_CodeChallenge HellaSwag evals in two models (part 1).md) | [Xem bài viết →](aero_LLM_08_CodeChallenge HellaSwag evals in two models (part 1).md) |
| [aero_LLM_09_CodeChallenge HellaSwag evals in two models (part 2).md](aero_LLM_09_CodeChallenge HellaSwag evals in two models (part 2).md) | [Xem bài viết →](aero_LLM_09_CodeChallenge HellaSwag evals in two models (part 2).md) |
| [aero_LLM_10_KL (Kullback-Leibler) divergence.md](aero_LLM_10_KL (Kullback-Leibler) divergence.md) | [Xem bài viết →](aero_LLM_10_KL (Kullback-Leibler) divergence.md) |
| [aero_LLM_11_MAUVE.md](aero_LLM_11_MAUVE.md) | [Xem bài viết →](aero_LLM_11_MAUVE.md) |
| [aero_LLM_12_CodeChallenge Large and small MAUVE explorations.md](aero_LLM_12_CodeChallenge Large and small MAUVE explorations.md) | [Xem bài viết →](aero_LLM_12_CodeChallenge Large and small MAUVE explorations.md) |
| [aero_LLM_13_SuperGLUE and other amalgamations.md](aero_LLM_13_SuperGLUE and other amalgamations.md) | [Xem bài viết →](aero_LLM_13_SuperGLUE and other amalgamations.md) |
| [aero_LLM_14_Assessing bias and fairness.md](aero_LLM_14_Assessing bias and fairness.md) | [Xem bài viết →](aero_LLM_14_Assessing bias and fairness.md) |
| [aero_LLM_15_Non-technical benchmarks.md](aero_LLM_15_Non-technical benchmarks.md) | [Xem bài viết →](aero_LLM_15_Non-technical benchmarks.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
