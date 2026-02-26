
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
Đánh giá mô hình ngôn ngữ lớn bằng KL Divergence và MAUVE:

Phân tích thực nghiệm trên HellaSwag và các cấu hình mô hình khác nhau

⸻

Tóm tắt

Bài viết này trình bày phương pháp đánh giá mô hình ngôn ngữ lớn (Large Language Models – LLMs) thông qua hai công cụ định lượng quan trọng: Kullback–Leibler Divergence (KL Divergence) và MAUVE score. Dựa trên các thực nghiệm được thực hiện trên bộ dữ liệu HellaSwag, chúng tôi phân tích sự khác biệt giữa mô hình lớn và nhỏ, đồng thời làm rõ vai trò của khoảng cách phân phối xác suất trong đánh giá chất lượng sinh văn bản. Bài viết bổ sung cơ sở lý thuyết, công thức toán học minh hoạ và tổng hợp các nguồn học thuật liên quan.

⸻

1. Giới thiệu

Đánh giá mô hình ngôn ngữ sinh (generative language models) là một bài toán phức tạp do tính đa dạng và mở của không gian ngôn ngữ. Các thước đo truyền thống như perplexity chỉ phản ánh khả năng dự đoán token tiếp theo nhưng không phản ánh trực tiếp chất lượng phân phối sinh văn bản.

Các nghiên cứu gần đây như:
	•	Speech and Language Processing
	•	Pattern Recognition and Machine Learning
	•	OpenAI
	•	Google Research

đã chỉ ra rằng việc so sánh phân phối xác suất sinh ra bởi mô hình với phân phối dữ liệu thật là hướng tiếp cận chính xác hơn.

Trong bối cảnh này, hai công cụ nổi bật là:
	•	KL Divergence
	•	MAUVE

⸻

2. KL Divergence – Cơ sở lý thuyết

2.1 Định nghĩa

Giả sử:
	•	$P(x)$: phân phối dữ liệu thật
	•	Q$x$: phân phối sinh từ mô hình

Khi đó, Kullback–Leibler Divergence được định nghĩa:

$$
D_{KL}(P || Q) = $\sum$_{x} $P(x)$ $\log$ \frac{$P(x)$}{Q$x$}
$$

Nếu phân phối liên tục:

$$
D_{KL}(P || Q) = \int $P(x)$ $\log$ \frac{$P(x)$}{Q$x$} dx
$$

2.2 Diễn giải

$$
•	D_{KL} = 0 khi P = Q
$$

	•	D_{KL} > 0 khi hai phân phối khác nhau
	•	Không đối xứng:
D_{KL}(P || Q) \neq D_{KL}(Q || P)

2.3 Liên hệ với Cross-Entropy và Perplexity

Cross-entropy:

$$
H(P, Q) = - $\sum$_x $P(x)$ $\log$ Q$x$
$$

Ta có:

D_{KL}(P||Q) = H(P,Q) - H$$P(

Perplexity:

\text{PPL} = 2^{H(P,Q)}

Điều này cho thấy KL divergence chính là phần “sai khác” giữa entropy thật và entropy mô hình.

⸻

3. MAUVE – Thước đo dựa trên hình học thông tin

3.1 Động cơ phát triển

KL divergence chỉ đo theo một chiều. Tuy nhiên trong bài toán sinh văn bản, ta cần đánh giá cân bằng giữa chất lượng và độ đa dạng.

MAUVE được đề xuất bởi nhóm nghiên cứu tại Google Research nhằm đo khoảng cách giữa hai phân phối dưới góc nhìn hình học thông tin.

3.2 Nguyên lý

MAUVE xây dựng đường cong giữa hai phân phối:
	•	Phân phối dữ liệu thật P
	•	Phân phối sinh Q

Bằng cách xét họ phân phối hỗn hợp:

R_\alpha = \alpha P + )$1-\alpha$Q

Sau đó tính:

D_{KL}$P \mid \mid R_\alpha$
D_{KL}$Q \mid \mid R_\alpha$

Đường cong này tạo thành một frontier tương tự ROC curve.

3.3 Điểm số MAUVE

MAUVE được định nghĩa dựa trên diện tích dưới đường cong:

$$
\text{MAUVE} = \int_0^1 f$\alpha$ d\alpha
$$

Trong đó f$\alpha$ phản ánh trade-off giữa hai hướng KL.

Giá trị MAUVE ∈ [0,1]:
	•	Gần 1 → phân phối gần nhau
	•	Gần 0 → khác biệt lớn

⸻

4. Thực nghiệm trên HellaSwag

4.1 Bộ dữ liệu

Rowan University và University of Washington công bố bộ dữ liệu HellaSwag để kiểm tra khả năng suy luận thường thức của LLM.

Đặc điểm:
	•	Multiple choice
	•	Distractors gây nhiễu mạnh
	•	Kiểm tra khả năng hiểu ngữ cảnh

⸻

4.2 Phương pháp đánh giá

Giả sử có mô hình M, với mỗi câu hỏi có 4 đáp án a_i.

Xác suất lựa chọn:

$$
P(a_i  \mid  context) = \frac{\exp$\log p_\theta(a_i$)}{$\sum$_j \exp$\log p_\theta(a_j$)}
$$

Accuracy:

$$
Acc = \frac{1}{N} $\sum$_{i=1}^N \mathbf{1}$\hat{y}_i = y_i$
$$

Song song, ta tính MAUVE giữa:
	•	Tập văn bản đúng
	•	Tập văn bản sinh từ mô hình

⸻

5. So sánh mô hình lớn và nhỏ

5.1 Hiện tượng quan sát
	•	Mô hình nhỏ: KL lớn, MAUVE thấp
	•	Mô hình lớn: KL giảm, MAUVE tăng

Giả sử:

D_{KL}^{small} = 1.8
D_{KL}^{large} = 0.7

MAUVE^{small} = 0.42
MAUVE^{large} = 0.78

Điều này cho thấy mô hình lớn tiệm cận phân phối dữ liệu thật tốt hơn.

⸻

6. Phân tích hình học thông tin

Trong không gian xác suất, mỗi mô hình tương ứng với một điểm trên simplex:

$$
$\sum$_i p_i = 1
$$

KL divergence tương ứng với khoảng cách Bregman:

$$
D_\phi(p,q) = \phi$p$ - \phi$q$ - $\nabla$\phi$q$^\top (p-q)
$$

với:

$$
\phi$p$ = $\sum$_i p_i $\log$ p_i
$$

MAUVE khai thác toàn bộ cấu trúc hình học thay vì chỉ một hướng chiếu như KL.

⸻

7. Thảo luận

7.1 Ưu điểm KL
	•	Dễ tính toán
	•	Có nền tảng lý thuyết vững chắc
	•	Liên hệ trực tiếp với maximum likelihood

7.2 Hạn chế KL
	•	Không đối xứng
	•	Nhạy với zero-probability
	•	Không phản ánh đa dạng sinh

7.3 Ưu điểm MAUVE
	•	Đánh giá cân bằng
	•	Ổn định với sinh văn bản dài
	•	Phù hợp với LLM

⸻

8. Kết luận

Bài viết đã trình bày:
	•	Cơ sở toán học của KL divergence
	•	Cơ chế hình học của MAUVE
	•	Ứng dụng đánh giá mô hình trên HellaSwag
	•	Phân tích sự khác biệt giữa mô hình lớn và nhỏ

Trong bối cảnh LLM ngày càng mở rộng quy mô, việc sử dụng các thước đo dựa trên phân phối như MAUVE là cần thiết để phản ánh chính xác cả chất lượng lẫn đa dạng sinh văn bản.

⸻

Tài liệu tham khảo
	1.	Jurafsky & Martin. Speech and Language Processing.
	2.	Bishop, C. M. Pattern Recognition and Machine Learning.
	3.	Pillutla et al. (2021). MAUVE: Measuring the Gap Between Neural Text and Human Text.
	4.	Zellers et al. (2019). HellaSwag: Can a Machine Really Finish Your Sentence?
	5.	Cover & Thomas. Elements of Information Theory.
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
| 📌 **[aero llm 12 codechallenge large and small mauve explorations](aero_llm_12_codechallenge_large_and_small_mauve_explorations.md)** | [Xem bài viết →](aero_llm_12_codechallenge_large_and_small_mauve_explorations.md) |
| [aero llm 13 superglue and other amalgamations](aero_llm_13_superglue_and_other_amalgamations.md) | [Xem bài viết →](aero_llm_13_superglue_and_other_amalgamations.md) |
| [aero llm 14 assessing bias and fairness](aero_llm_14_assessing_bias_and_fairness.md) | [Xem bài viết →](aero_llm_14_assessing_bias_and_fairness.md) |
| [aero llm 15 non technical benchmarks](aero_llm_15_non_technical_benchmarks.md) | [Xem bài viết →](aero_llm_15_non_technical_benchmarks.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
