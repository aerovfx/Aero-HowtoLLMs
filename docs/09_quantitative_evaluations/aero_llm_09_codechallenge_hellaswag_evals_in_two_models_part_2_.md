
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [09 quantitative evaluations](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
So sánh thực nghiệm HellaSwag trên hai mô hình ngôn ngữ (Phần 2):

Phân tích log-likelihood, chuẩn hoá độ dài và kiểm định thống kê

⸻

Tóm tắt

Bài viết này tiếp tục phân tích bài toán đánh giá HellaSwag trên hai mô hình ngôn ngữ lớn (LLMs), dựa trên nội dung tài liệu đính kèm (phần 2). Trọng tâm là:
	•	Triển khai tính log-likelihood có điều kiện
	•	So sánh hai mô hình thông qua accuracy
	•	Phân tích độ lệch do chuẩn hoá độ dài
	•	Kiểm định ý nghĩa thống kê

Nền tảng lý thuyết dựa trên nghiên cứu của Rowan Zellers et al. (2019), kiến trúc Transformer của Ashish Vaswani et al. (2017) và các phân tích scaling của OpenAI.

⸻

1. Bài toán đánh giá thực nghiệm

Với mỗi mẫu dữ liệu:
	•	Ngữ cảnh c
	•	4 lựa chọn \{a_1, a_2, a_3, a_4\}
	•	Đáp án đúng a_{\text{true}}

Mục tiêu: so sánh hai mô hình M_1 và M_2.

⸻

2. Tính log-likelihood chi tiết

Với mô hình tự hồi quy:

P(a_i \mid c) = \prod_{t=1}^{T_i} P(w_t \mid c, w_{<t})

Để tránh tràn số:

\log P(a_i \mid c) = \sum_{t=1}^{T_i} \log P(w_t \mid c, w_{<t})

Trong thực tế, ta tính:

Score(a_i) = \frac{1}{T_i^\alpha} \sum_{t=1}^{T_i} \log P(w_t \mid c, w_{<t})

Trong đó:
	•	\alpha = 1 → chuẩn hoá trung bình
	•	0 < \alpha < 1 → giảm thiên lệch độ dài

⸻

3. Cơ chế forward pass trong Transformer

Transformer tính xác suất thông qua:

h_t = \text{Transformer}(c, w_{<t})

Sau đó:

P(w_t) = \text{softmax}(Wh_t)

Trong đó:

\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}

Self-attention:

Attention(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

⸻

4. So sánh Accuracy giữa hai mô hình

Giả sử:
	•	Mô hình M_1: accuracy \hat{p}_1
	•	Mô hình M_2: accuracy \hat{p}_2
	•	Số mẫu: N

Sai số chuẩn:

SE_i = \sqrt{\frac{\hat{p}_i (1-\hat{p}_i)}{N}}

Kiểm định z:

z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{SE_1^2 + SE_2^2}}

Nếu:

|z| > 1.96

→ Khác biệt có ý nghĩa ở mức 95%.

⸻

5. Phân tích sai lệch do độ dài

Nếu không chuẩn hoá:
	•	Chuỗi dài có tổng log nhỏ hơn
	•	Mô hình ưu tiên đáp án ngắn

Giả sử hai đáp án:
	•	T_1 = 5
	•	T_2 = 20

Nếu xác suất token trung bình như nhau:

\sum_{t=1}^{5} \log p = -10
\sum_{t=1}^{20} \log p = -40

Không chuẩn hoá → chọn chuỗi ngắn
Chuẩn hoá:

\frac{-10}{5} = -2
\frac{-40}{20} = -2

→ công bằng.

⸻

6. So sánh với Perplexity

Perplexity:

PP = \exp\left(- \frac{1}{N} \sum \log P(w_i)\right)

HellaSwag đo khả năng phân biệt nhiều chuỗi hoàn chỉnh.

Mô hình có perplexity tốt nhưng thiếu reasoning vẫn có thể:

Accuracy_{\text{HellaSwag}} thấp

⸻

7. Phân tích scaling

Theo luật scaling:

Loss(N) = A N^{-\alpha} + B

Accuracy thường tăng theo:

Accuracy(N) \approx C - D N^{-\beta}

Khi N tăng → performance tiệm cận trần.

⸻

8. Phân tích lỗi

Các lỗi phổ biến:
	1.	Chọn continuation “nghe tự nhiên” nhưng sai logic vật lý.
	2.	Nhầm lẫn do bias dữ liệu huấn luyện.
	3.	Sai do thiếu hiểu biết hành động hiếm gặp.

⸻

9. Calibration và độ tin cậy

Expected Calibration Error (ECE):

ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|

Mô hình tốt không chỉ cần accuracy cao mà còn:

acc \approx conf

⸻

10. Kết quả định tính (theo xu hướng chung nghiên cứu)
	•	Mô hình lớn hơn → accuracy cao hơn
	•	Chuẩn hoá độ dài giúp tăng 1–3%
	•	Scaling cải thiện reasoning emergent

⸻

11. Ý nghĩa khoa học

Đánh giá hai mô hình trên HellaSwag cho phép:
	•	So sánh năng lực reasoning thực tế
	•	Đo ảnh hưởng scaling
	•	Kiểm tra bias độ dài
	•	Phân tích calibration

Benchmark này là cầu nối giữa:
	•	Perplexity (mức token)
	•	Reasoning (mức chuỗi)

⸻

12. Kết luận

Phần 2 cho thấy việc đánh giá HellaSwag yêu cầu:
	1.	Tính log-likelihood chính xác
	2.	Chuẩn hoá độ dài
	3.	Kiểm định thống kê
	4.	Phân tích calibration

So sánh hai mô hình không chỉ dừng ở accuracy mà cần đánh giá toàn diện xác suất và độ tin cậy.

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
| 📌 **[aero llm 09 codechallenge hellaswag evals in two models part 2](aero_llm_09_codechallenge_hellaswag_evals_in_two_models_part_2_.md)** | [Xem bài viết →](aero_llm_09_codechallenge_hellaswag_evals_in_two_models_part_2_.md) |
| [aero llm 10 kl kullback leibler divergence](aero_llm_10_kl_kullback_leibler_divergence.md) | [Xem bài viết →](aero_llm_10_kl_kullback_leibler_divergence.md) |
| [aero llm 11 mauve](aero_llm_11_mauve.md) | [Xem bài viết →](aero_llm_11_mauve.md) |
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
