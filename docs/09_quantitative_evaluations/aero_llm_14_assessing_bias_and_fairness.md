
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
Đánh giá thiên lệch (Bias) và công bằng (Fairness) trong mô hình ngôn ngữ lớn

Cơ sở lý thuyết, thước đo định lượng và công thức toán học minh hoạ

⸻

Tóm tắt

Sự phát triển nhanh chóng của các mô hình ngôn ngữ lớn (Large Language Models – LLMs) đặt ra những thách thức nghiêm trọng liên quan đến thiên lệch (bias) và công bằng (fairness). Bài viết này tổng hợp cơ sở lý thuyết về đánh giá thiên lệch trong hệ thống học máy, trình bày các thước đo định lượng quan trọng như Demographic Parity, Equalized Odds, Calibration và KL Divergence, đồng thời phân tích cách các tổ chức nghiên cứu lớn triển khai quy trình đánh giá fairness. Các công thức toán học được bổ sung nhằm làm rõ bản chất thống kê của vấn đề.

⸻

1. Giới thiệu

LLMs được huấn luyện trên dữ liệu web quy mô lớn, dẫn đến nguy cơ hấp thụ và khuếch đại các thiên lệch xã hội. Các tổ chức như:
	•	OpenAI
	•	Anthropic
	•	DeepMind

đã nhấn mạnh rằng đánh giá hiệu năng (accuracy) thôi là chưa đủ; cần có cơ chế đánh giá độ công bằng và an toàn.

⸻

2. Định nghĩa thiên lệch và công bằng

2.1 Thiên lệch (Bias)

Trong học máy, bias được hiểu là sự sai lệch có hệ thống của mô hình đối với một nhóm đặc trưng nhạy cảm A (giới tính, chủng tộc, tôn giáo…).

Giả sử:
	•	X: đặc trưng đầu vào
	•	Y: nhãn thật
	•	A: thuộc tính nhạy cảm
	•	\hat{Y}: dự đoán

⸻

2.2 Công bằng (Fairness)

Một hệ thống được coi là công bằng nếu:

$P(\hat{Y}\mid A=a_1)$ $\approx$ $P(\hat{Y}\mid A=a_2)$

với mọi giá trị a_1, a_2.

⸻

3. Các thước đo công bằng phổ biến

3.1 Demographic Parity (DP)

Điều kiện:

$P(\hat{Y}=1\mid A=0)$ = $P(\hat{Y}=1\mid A=1)$

Độ lệch DP:

\Delta_{DP} = |$P(\hat{Y}=1\mid A=0)$ - $P(\hat{Y}=1\mid A=1)$|

Nếu \Delta_{DP} lớn → có thiên lệch.

⸻

3.2 Equalized Odds (EO)

Yêu cầu:

$P(\hat{Y}=1\mid Y=y, A=0)$ = $P(\hat{Y}=1\mid Y=y, A=1)$

với y \in \{0,1\}.

Điều này kiểm soát cả False Positive và False Negative.

⸻

3.3 Calibration

Một mô hình được calibrated nếu:

$P(Y=1\mid \hat{P}=p, A=a)$ = p

Với mọi nhóm a.

⸻

4. Đo khoảng cách phân phối bằng KL Divergence

Một cách định lượng thiên lệch là so sánh phân phối dự đoán giữa các nhóm:

D_{KL}(P_{A=0} || P_{A=1})

Trong đó:

$$
D_{KL}(P||Q) = $\sum$_x $P(x)$\log \frac{$P(x)$}{Q$x$}
$$

Nếu:

D_{KL} \rightarrow 0

→ phân phối gần nhau → ít thiên lệch.

⸻

5. Fairness trong mô hình ngôn ngữ sinh

Khác với phân loại nhị phân, LLM sinh chuỗi token:

$$
P_\theta$x_1,\dots,x_T$ = $\prod$_{t=1}^{T} P_\theta(x_t|x_{\lt t})
$$

Thiên lệch có thể đo bằng cách so sánh xác suất sinh câu liên quan đến nhóm A:

Bias = $\mathbb${E}_{prompt \in G_1}[$\log$ P_\theta(response)] - $\mathbb${E}_{prompt \in G_2}[$\log$ P_\theta(response)]

⸻

6. Phương pháp đánh giá thực nghiệm

6.1 Counterfactual Evaluation

Tạo cặp prompt:
	•	“The doctor said he…”
	•	“The doctor said she…”

Tính chênh lệch log-likelihood:

$$
\Delta = $\log$ P_\theta(r|he) - $\log$ P_\theta(r|she)
$$

⸻

6.2 Toxicity Score

Sử dụng classifier phụ để ước lượng:

$$
Toxicity = P_{tox}(text)
$$

So sánh kỳ vọng theo nhóm:

$$
\Delta_{tox} = $\mathbb${E}[T|A=0] - $\mathbb${E}[T|A=1]
$$

⸻

7. Phân tích thống kê

Giả sử có hai nhóm:

$$
\mu_1 = 0.62, \quad \mu_2 = 0.54
$$

Kiểm định:

$$
t = \frac{\mu_1 - \mu_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}
$$

Nếu p < 0.05 → khác biệt có ý nghĩa.

⸻

8. Liên hệ lý thuyết thông tin

Theo Elements of Information Theory:

Mutual Information giữa dự đoán và thuộc tính nhạy cảm:

$$
I$\hat{Y};A$ = $\sum$_{a,y} P(a,y)$\log$\frac{P(a,y)}{$P(a)$$P(y)$}
$$

Nếu:

$$
I$\hat{Y};A$ $\approx$ 0
$$

→ ít phụ thuộc → công bằng hơn.

⸻

9. Các chiến lược giảm bias

9.1 Regularization

Thêm penalty:

$$
$\mathcal${L}_{total} = $\mathcal${L}_{task} + \lambda D_{KL}(P_{A=0}||P_{A=1})
$$

⸻

9.2 Adversarial Debiasing

Huấn luyện mô hình phụ đoán A:

$$
\min_\theta \max_\phi $\le$ft$\mathcal{L}_{task} - \lambda $\mathcal${L}_{adv} \right$
$$

⸻

9.3 RLHF với ràng buộc công bằng

Tối ưu:

\max_\theta $\mathbb${E}[R] - \beta D_{KL}$P_\theta \mid \mid P_{ref}$

Trong đó reward bao gồm yếu tố fairness.

⸻

10. Thảo luận

Thách thức
	•	Dữ liệu không cân bằng
	•	Định nghĩa fairness mâu thuẫn nhau
	•	Trade-off giữa accuracy và fairness

Theo nghiên cứu của các nhóm tại Stanford University và MIT, không tồn tại định nghĩa công bằng duy nhất thỏa mãn mọi điều kiện đồng thời.

⸻

11. Kết luận

Đánh giá bias và fairness trong LLM đòi hỏi:
	•	Thước đo thống kê rõ ràng
	•	Phân tích phân phối xác suất
	•	Kiểm định ý nghĩa thống kê
	•	Kết hợp kỹ thuật giảm thiên lệch trong huấn luyện

Trong bối cảnh AI ngày càng ảnh hưởng xã hội, fairness không chỉ là vấn đề kỹ thuật mà còn là yêu cầu đạo đức và pháp lý.

⸻

Tài liệu tham khảo
	1.	Cover & Thomas. Elements of Information Theory.
	2.	Barocas, Hardt & Narayanan. Fairness and Machine Learning.
	3.	Mehrabi et al. (2021). A Survey on Bias and Fairness in Machine Learning.
	4.	OpenAI System Card (các phiên bản gần đây).
	5.	Bender et al. (2021). On the Dangers of Stochastic Parrots.
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
| [aero llm 11 mauve](aero_llm_11_mauve.md) | [Xem bài viết →](aero_llm_11_mauve.md) |
| [aero llm 12 codechallenge large and small mauve explorations](aero_llm_12_codechallenge_large_and_small_mauve_explorations.md) | [Xem bài viết →](aero_llm_12_codechallenge_large_and_small_mauve_explorations.md) |
| [aero llm 13 superglue and other amalgamations](aero_llm_13_superglue_and_other_amalgamations.md) | [Xem bài viết →](aero_llm_13_superglue_and_other_amalgamations.md) |
| 📌 **[aero llm 14 assessing bias and fairness](aero_llm_14_assessing_bias_and_fairness.md)** | [Xem bài viết →](aero_llm_14_assessing_bias_and_fairness.md) |
| [aero llm 15 non technical benchmarks](aero_llm_15_non_technical_benchmarks.md) | [Xem bài viết →](aero_llm_15_non_technical_benchmarks.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
