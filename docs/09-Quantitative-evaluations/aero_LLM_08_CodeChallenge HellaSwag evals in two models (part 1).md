
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
Đánh giá HellaSwag trên hai mô hình ngôn ngữ: Phân tích định lượng và so sánh xác suất sinh chuỗi

Tiếp cận log-likelihood, chuẩn hoá độ dài và ý nghĩa thống kê

⸻

Tóm tắt

Bài viết này trình bày phương pháp đánh giá bộ dữ liệu HellaSwag trên hai mô hình ngôn ngữ lớn (LLMs), dựa trên nội dung tài liệu đính kèm. Chúng tôi phân tích cơ chế tính điểm bằng log-likelihood, chuẩn hoá theo độ dài chuỗi, và phương pháp tính accuracy. Bài viết mở rộng với các nền tảng lý thuyết từ nghiên cứu của Rowan Zellers et al. (2019), kiến trúc Transformer của Ashish Vaswani et al. (2017), và các phân tích scaling từ OpenAI.

⸻

1. Giới thiệu

HellaSwag là bộ benchmark đo lường khả năng:
	•	Suy luận thường thức (commonsense reasoning)
	•	Hiểu chuỗi hành động vật lý
	•	Phân biệt continuation hợp lý và phi lý

Trong bài toán này, mỗi câu hỏi gồm:
	•	Ngữ cảnh c
	•	4 lựa chọn hoàn thành \{a_1, a_2, a_3, a_4\}

Mục tiêu: chọn đáp án có xác suất cao nhất theo mô hình.

⸻

2. Mô hình xác suất cho bài toán multiple choice

Với mô hình tự hồi quy (autoregressive), xác suất của một đáp án được tính:

P(a_i \mid c) = \prod_{t=1}^{T_i} P(w_t \mid c, w_{<t})

Trong thực nghiệm, ta dùng log để tránh underflow:

\log P(a_i \mid c) = \sum_{t=1}^{T_i} \log P(w_t \mid c, w_{<t})

⸻

3. Vấn đề thiên lệch độ dài (Length Bias)

Nếu dùng tổng log-likelihood trực tiếp:
	•	Chuỗi dài → log nhỏ hơn (âm hơn)
	•	Chuỗi ngắn → được ưu tiên

Do đó cần chuẩn hoá:

Score(a_i) = \frac{1}{T_i} \sum_{t=1}^{T_i} \log P(w_t \mid c, w_{<t})

Đây là ave18-RAGe log-probability.

⸻

4. Quy tắc chọn đáp án

\hat{a} = \arg\max_{a_i} Score(a_i)

Accuracy được tính:

Accuracy = \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}(\hat{a}^{(j)} = a_{\text{true}}^{(j)})

Baseline ngẫu nhiên:

P_{\text{random}} = 25\%

⸻

5. So sánh hai mô hình

Giả sử hai mô hình:
	•	M_1
	•	M_2

Accuracy tương ứng:

\hat{p}_1, \hat{p}_2

Sai số chuẩn:

SE = \sqrt{\frac{\hat{p}(1-\hat{p})}{N}}

Kiểm định sự khác biệt:

z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{SE_1^2 + SE_2^2}}

Nếu:

|z| > 1.96

→ khác biệt có ý nghĩa thống kê (95%).

⸻

6. Liên hệ với Self-Attention

Transformer sử dụng cơ chế:

Attention(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

Cơ chế này giúp mô hình:
	•	Theo dõi quan hệ dài hạn
	•	Liên kết hành động trước–sau
	•	Phát hiện logic vật lý ngầm định

⸻

7. So sánh với Perplexity

Perplexity đo khả năng dự đoán token kế tiếp:

PP = \exp\left(- \frac{1}{N} \sum \log P(w_i)\right)

Trong khi HellaSwag đo:
	•	So sánh chuỗi hoàn chỉnh
	•	Khả năng reasoning cấp cao

Một mô hình có perplexity thấp chưa chắc có accuracy cao trên HellaSwag.

⸻

8. Phân tích scaling

Theo luật scaling của OpenAI:

Loss(N) = A N^{-\alpha} + B

Khi tăng số tham số N:
	•	Log-likelihood tăng
	•	Accuracy trên HellaSwag tăng theo hàm lũy thừa

⸻

9. Hạn chế của phương pháp đánh giá

9.1 Shortcut Learning

Mô hình có thể:
	•	Học phong cách câu hợp lý
	•	Không thực sự hiểu vật lý

⸻

9.2 Dataset Saturation

Nếu fine-tune trực tiếp trên HellaSwag:

D_{train} \cap D_{test} \neq \varnothing

→ Không còn phản ánh năng lực tổng quát.

⸻

9.3 Calibration

Mô hình có thể:
	•	Chọn đúng
	•	Nhưng xác suất không cao

Đo calibration:

ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|

⸻

10. Ý nghĩa thực tiễn

Đánh giá HellaSwag giúp:
	•	So sánh mô hình trước khi fine-tune
	•	Đánh giá khả năng reasoning
	•	Kiểm tra hiệu quả scaling

Trong pipeline triển khai thực tế, cần kết hợp:
	•	Accuracy
	•	Log-likelihood
	•	Calibration
	•	Robustness test

⸻

11. Kết luận

Đánh giá HellaSwag trên hai mô hình yêu cầu:
	•	Tính log-likelihood chính xác
	•	Chuẩn hoá độ dài
	•	So sánh thống kê

Benchmark này không chỉ đo fluency mà đo khả năng suy luận hành động, do đó quan trọng trong đánh giá LLM hiện đại.

⸻

Tài liệu tham khảo
	1.	Zellers, R. et al. (2019). HellaSwag: Can a Machine Really Finish Your Sentence?
	2.	Vaswani, A. et al. (2017). Attention is All You Need.
	3.	Brown et al. (2020). Language Models are Few-Shot Learners.
	4.	Kaplan et al. (2020). Scaling Laws for Neural Language Models.
	5.	Jurafsky & Martin. Speech and Language Processing.
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
| [aero_LLM_04_CodeChallenge Perplexing perplexities.md](aero_LLM_04_CodeChallenge Perplexing perplexities.md) | [Xem bài viết →](aero_LLM_04_CodeChallenge Perplexing perplexities.md) |
| [aero_LLM_05_Masked word prediction accuracy.md](aero_LLM_05_Masked word prediction accuracy.md) | [Xem bài viết →](aero_LLM_05_Masked word prediction accuracy.md) |
| [aero_LLM_06_HellaSwag.md](aero_LLM_06_HellaSwag.md) | [Xem bài viết →](aero_LLM_06_HellaSwag.md) |
| [aero_LLM_07_Import large models using bitsandbytes.md](aero_LLM_07_Import large models using bitsandbytes.md) | [Xem bài viết →](aero_LLM_07_Import large models using bitsandbytes.md) |
| 📌 **[aero_LLM_08_CodeChallenge HellaSwag evals in two models (part 1).md](aero_LLM_08_CodeChallenge HellaSwag evals in two models (part 1).md)** | [Xem bài viết →](aero_LLM_08_CodeChallenge HellaSwag evals in two models (part 1).md) |
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
