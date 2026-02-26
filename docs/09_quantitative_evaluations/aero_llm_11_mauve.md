
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
MAUVE: Đo lường chất lượng và đa dạng của mô hình sinh ngôn ngữ thông qua hình học phân phối

Phân tích lý thuyết, công thức toán học và ứng dụng trong đánh giá LLM

⸻

Tóm tắt

Bài viết này trình bày phương pháp MAUVE – một thước đo hiện đại để đánh giá mô hình sinh ngôn ngữ dựa trên so sánh hình học giữa hai phân phối xác suất: phân phối dữ liệu thật và phân phối do mô hình sinh ra. Nội dung được phát triển dựa trên tài liệu đính kèm và mở rộng từ công trình của Krishna Pillutla et al. (2021), nền tảng lý thuyết phân kỳ thông tin của Solomon Kullback và Richard Leibler, cùng ứng dụng trong các mô hình ngôn ngữ lớn tại OpenAI.

⸻

1. Giới thiệu

Đánh giá mô hình sinh ngôn ngữ (text generation) là bài toán khó vì cần cân bằng:
	•	Chất lượng (quality): câu có hợp lý, trôi chảy?
	•	Đa dạng (diversity): mô hình có sinh lặp lại không?

Các thước đo truyền thống như:
	•	Perplexity
	•	BLEU
	•	ROUGE

không phản ánh đầy đủ sự khác biệt phân phối toàn cục.

MAUVE giải quyết bằng cách:
	•	So sánh phân phối embedding của văn bản thật và văn bản sinh
	•	Xây dựng đường cong trade-off giữa precision và recall

⸻

2. Cơ sở lý thuyết

Giả sử:
	•	P: phân phối dữ liệu thật
	•	Q: phân phối mô hình sinh

Ta muốn đo mức gần nhau giữa P và Q.

⸻

3. KL Divergence và hạn chế

Phân kỳ KL:

$$
D_{KL}$P \\mid  Q$ = $\sum$_x $P(x)$\log \frac{$P(x)$}{Q$x$}
$$

Vấn đề:
	•	Không đối xứng
	•	Không đo đồng thời precision và recall
	•	Không phản ánh hình học phân phối

⸻

4. Ý tưởng của MAUVE

MAUVE dựa trên họ phân kỳ:

D_\lambda$P \\mid  Q$

Tạo phân phối trộn:

R_\lambda = \lambda P + $1-\lambda$ Q

Sau đó tính:

D_{KL}$P \\mid  R_\lambda$
\quad \text{và} \quad
D_{KL}$Q \\mid  R_\lambda$

Khi thay đổi \lambda \in [0,1], ta thu được một đường cong trong không gian hai chiều.

⸻

5. Precision–Recall Curve trong không gian phân phối

MAUVE xây dựng đồ thị:

x$\lambda$ = D_{KL}$P \\mid  R_\lambda$
y$\lambda$ = D_{KL}$Q \\mid  R_\lambda$

Diện tích dưới đường cong này được chuẩn hoá thành điểm MAUVE:

MAUVE \in [0,1]

Giá trị gần 1 → phân phối gần nhau.

⸻

6. Triển khai thực tế

6.1 Embedding

Văn bản được ánh xạ vào không gian embedding:

$$
x_i = f_{\text{LM}}$text_i$
$$

Trong đó f_{\text{LM}} là encoder từ Transformer của Ashish Vaswani et al.

⸻

6.2 Rời rạc hoá không gian

Không gian embedding được phân cụm (k-means):

$$
\min $\sum$_{i=1}^{N} ||x_i - c_{z_i}||^2
$$

Sau đó ước lượng phân phối rời rạc trên các cluster.

⸻

7. So sánh với Perplexity

Perplexity:

$$
PP = \exp$\le$ft$- \frac{1}{N} $\sum$ $\log$ P(w_i$\right)
$$

Perplexity:
	•	Đo chất lượng token-level
	•	Không đo đa dạng toàn cục

MAUVE:
	•	Đo phân phối toàn văn bản
	•	Cân bằng precision–recall

⸻

8. Phân tích hình học

Giả sử:

$$
•	P = Q
$$

→ Với mọi \lambda:

D_{KL}$P \\mid  R_\lambda$ = D_{KL}$Q \\mid  R_\lambda$

→ MAUVE = 1

Nếu:
	•	Q collapse (mode collapse)

→ D_{KL}$P \\mid  Q$ lớn
→ MAUVE giảm mạnh.

⸻

9. Phân tích giới hạn

9.1 Khi Q thiếu đa dạng

Recall thấp:

D_{KL}$P \\mid  R_\lambda$ \uparrow

⸻

9.2 Khi Q sinh nhiễu

Precision thấp:

D_{KL}$Q \\mid  R_\lambda$ \uparrow

⸻

10. So sánh với Jensen–Shannon Divergence

JSD:

JSD$P \\mid  Q$ =
\frac{1}{2} D_{KL}$P \\mid  M$
+
\frac{1}{2} D_{KL}$Q \\mid  M$

với:

M = \frac{1}{2}$P+Q$

MAUVE có thể xem như mở rộng hình học của JSD khi thay đổi \lambda.

⸻

11. Ý nghĩa trong đánh giá LLM

MAUVE đặc biệt hữu ích khi:
	•	So sánh hai mô hình sinh văn bản
	•	Đánh giá fine-tuning
	•	Đo hiệu quả RLHF

Trong pipeline huấn luyện tại OpenAI, MAUVE có thể bổ sung cho perplexity.

⸻

12. Hạn chế
	1.	Phụ thuộc embedding model
	2.	Phụ thuộc số cluster
	3.	Tốn chi phí tính toán

⸻

13. Kết luận

MAUVE là thước đo tiên tiến:
	•	Dựa trên hình học phân phối
	•	Cân bằng chất lượng và đa dạng
	•	Khắc phục hạn chế của perplexity

Nó kết nối lý thuyết phân kỳ KL với đánh giá mô hình sinh hiện đại.

⸻

Tài liệu tham khảo
	1.	Pillutla, K. et al. (2021). MAUVE: Measuring the Gap Between Neural Text and Human Text.
	2.	Kullback, S., Leibler, R. (1951). On Information and Sufficiency.
	3.	Shannon, C. (1948). A Mathematical Theory of Communication.
	4.	Vaswani, A. et al. (2017). Attention is All You Need.
	5.	Goodfellow, I. et al. (2016). Deep Learning.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Đánh Giá Hộp Đen (Black-box Evaluations) trong Mô Hình Ngôn Ngữ Lớn](aero_llm_016_black_box_evals.md) | [Xem bài viết →](aero_llm_016_black_box_evals.md) |
| [Red Teaming: Đội Đỏ và Thử Nghiệm Đối Kháng trong AI Safety](aero_llm_017_red_teaming.md) | [Xem bài viết →](aero_llm_017_red_teaming.md) |
| [Độ Chính Xác, Tính Mạch Lạc và Sự Phù Hợp trong Đánh Giá Mô Hình Ngôn Ngữ](aero_llm_018_accuracy_coherence_and_relevance.md) | [Xem bài viết →](aero_llm_018_accuracy_coherence_and_relevance.md) |
| [Phân Phối Của Các Kích Hoạt Trạng Thái Ẩn Trong Mô Hình Ngôn Ngữ](aero_llm_019_distributions_of_hidden_state_activations.md) | [Xem bài viết →](aero_llm_019_distributions_of_hidden_state_activations.md) |
| [Hứa Hẹn và Thách Thức của Đánh Giá Định Lượng trong Mô Hình Học Máy](aero_llm_01_promises_and_challenges_of_quantitative_evaluations.md) | [Xem bài viết →](aero_llm_01_promises_and_challenges_of_quantitative_evaluations.md) |
| [Bản Đồ Nhiệt Của Token Cho Cân Nhắc Định Tính (Text Heatmaps)](aero_llm_020_heatmaps_of_tokens_for_qualitative_inspection.md) | [Xem bài viết →](aero_llm_020_heatmaps_of_tokens_for_qualitative_inspection.md) |
| [Thử Thách Lập Trình: Trực Quan Hóa Dự Đoán Đơn Token](aero_llm_021_codechallenge_visualize_single_token_predictions.md) | [Xem bài viết →](aero_llm_021_codechallenge_visualize_single_token_predictions.md) |
| [Các Vấn Đề Số Học trong Logits và Softmax: Phân Tích Toán Học và Giải Pháp Ổn Định](aero_llm_02_numerical_issues_in_logits_and_softmax.md) | [Xem bài viết →](aero_llm_02_numerical_issues_in_logits_and_softmax.md) |
| [Perplexity trong Mô Hình Ngôn Ngữ: Cơ Sở Toán Học, Diễn Giải và Giới Hạn](aero_llm_03_perplexity.md) | [Xem bài viết →](aero_llm_03_perplexity.md) |
| [aero llm 04 codechallenge perplexing perplexities](aero_llm_04_codechallenge_perplexing_perplexities.md) | [Xem bài viết →](aero_llm_04_codechallenge_perplexing_perplexities.md) |
| [aero llm 05 masked word prediction accuracy](aero_llm_05_masked_word_prediction_accuracy.md) | [Xem bài viết →](aero_llm_05_masked_word_prediction_accuracy.md) |
| [aero llm 06 hellaswag](aero_llm_06_hellaswag.md) | [Xem bài viết →](aero_llm_06_hellaswag.md) |
| [aero llm 07 import large models using bitsandbytes](aero_llm_07_import_large_models_using_bitsandbytes.md) | [Xem bài viết →](aero_llm_07_import_large_models_using_bitsandbytes.md) |
| [aero llm 08 codechallenge hellaswag evals in two models part 1](aero_llm_08_codechallenge_hellaswag_evals_in_two_models_part_1_.md) | [Xem bài viết →](aero_llm_08_codechallenge_hellaswag_evals_in_two_models_part_1_.md) |
| [aero llm 09 codechallenge hellaswag evals in two models part 2](aero_llm_09_codechallenge_hellaswag_evals_in_two_models_part_2_.md) | [Xem bài viết →](aero_llm_09_codechallenge_hellaswag_evals_in_two_models_part_2_.md) |
| [aero llm 10 kl kullback leibler divergence](aero_llm_10_kl_kullback_leibler_divergence.md) | [Xem bài viết →](aero_llm_10_kl_kullback_leibler_divergence.md) |
| 📌 **[aero llm 11 mauve](aero_llm_11_mauve.md)** | [Xem bài viết →](aero_llm_11_mauve.md) |
| [aero llm 12 codechallenge large and small mauve explorations](aero_llm_12_codechallenge_large_and_small_mauve_explorations.md) | [Xem bài viết →](aero_llm_12_codechallenge_large_and_small_mauve_explorations.md) |
| [aero llm 13 superglue and other amalgamations](aero_llm_13_superglue_and_other_amalgamations.md) | [Xem bài viết →](aero_llm_13_superglue_and_other_amalgamations.md) |
| [aero llm 14 assessing bias and fairness](aero_llm_14_assessing_bias_and_fairness.md) | [Xem bài viết →](aero_llm_14_assessing_bias_and_fairness.md) |
| [aero llm 15 non technical benchmarks](aero_llm_15_non_technical_benchmarks.md) | [Xem bài viết →](aero_llm_15_non_technical_benchmarks.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
