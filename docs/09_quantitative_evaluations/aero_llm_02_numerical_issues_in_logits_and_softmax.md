
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
# Các Vấn Đề Số Học trong Logits và Softmax: Phân Tích Toán Học và Giải Pháp Ổn Định

Tóm tắt

Trong các mô hình phân loại và mô hình ngôn ngữ, hàm softmax được sử dụng để chuyển logits thành phân phối xác suất. Tuy nhiên, khi logits có giá trị lớn hoặc rất nhỏ, các vấn đề số học như overflow, underflow và mất ổn định gradient có thể xảy ra. Bài viết này phân tích bản chất toán học của những vấn đề này, chỉ ra nguyên nhân từ biểu diễn số dấu chấm động (floating-point), và trình bày các kỹ thuật ổn định như log-sum-exp trick. Nội dung được mở rộng từ các tài liệu kinh điển của Ian Goodfellow, Yoshua Bengio và Geoffrey Hinton.

⸻

1. Giới thiệu

Trong bài toán phân loại nhiều lớp, mô hình xuất ra một vector logits:

\mathbf{z} = (z_1, z_2, \dots, z_K)

Softmax chuyển logits thành xác suất:

\sigma(z_i)
=
\frac{\exp(z_i)}
{\sum_{j=1}^{K} \exp(z_j)}

Tuy nhiên, khi z_i có độ lớn lớn (|z| >> 1), phép tính \exp(z_i) có thể gây lỗi số học.

⸻

2. Phân tích Vấn đề Overflow và Underflow

2.1 Biểu diễn số dấu chấm động

Trong chuẩn IEEE 754 (float32):

\exp(88.7) \approx 3.4 \times 10^{38}

Nếu:

z_i > 88

→ overflow (vượt quá khả năng biểu diễn).

Ngược lại:

\exp(-100) \approx 3.7 \times 10^{-44}

→ underflow (gần 0).

⸻

2.2 Ví dụ minh họa

Giả sử:

\mathbf{z} = (1000, 1001, 999)

Ta có:

\exp(1000) = \infty

Khi đó:

\sigma(z_i)
=
\frac{\infty}{\infty}

→ Không xác định (NaN).

⸻

3. Log-Sum-Exp Trick

Để tránh overflow, ta trừ đi giá trị lớn nhất:

\sigma(z_i)
=
\frac{\exp(z_i - z_{max})}
{\sum_j \exp(z_j - z_{max})}

Trong đó:

z_{max} = \max_j z_j

Vì:

\exp(z_i - z_{max}) \le 1

→ đảm bảo ổn định số học.

⸻

3.1 Dạng log-softmax

Trong nhiều thư viện, ta dùng:

\log \sigma(z_i)
=
z_i
-
\log
\left(
\sum_j \exp(z_j)
\right)

Áp dụng log-sum-exp:

\log
\left(
\sum_j \exp(z_j)
\right)
=
z_{max}
+
\log
\left(
\sum_j \exp(z_j - z_{max})
\right)

⸻

4. Ảnh hưởng đến Gradient

Cross-entropy loss:

\mathcal{L}
=
-
\sum_i y_i \log \sigma(z_i)

Gradient:

\frac{\partial \mathcal{L}}{\partial z_i}
=
\sigma(z_i) - y_i

Nếu softmax không ổn định → gradient NaN → lan truyền lỗi qua backpropagation.

⸻

5. Saturation và Vanishing Gradient

Khi một logit rất lớn:

z_k \gg z_j

Ta có:

\sigma(z_k) \approx 1
\quad
\sigma(z_j) \approx 0

Gradient:

\frac{\partial \mathcal{L}}{\partial z_k}
=
1 - y_k

Nếu dự đoán đúng và tự tin cao → gradient gần 0 → học chậm.

⸻

6. Phân tích Điều kiện Số

Độ điều kiện (condition number):

\kappa =
\frac{\max |z_i|}
{\min |z_i|}

Khi \kappa lớn → dễ mất ổn định.

Trong mô hình lớn (LLMs):

z_i = \mathbf{w}_i^\top \mathbf{h}

Nếu:

||\mathbf{w}_i||, ||\mathbf{h}|| \rightarrow lớn

→ logits tăng → nguy cơ overflow.

⸻

7. Mixed Precision Training

Khi dùng float16:

\exp(11) \approx 59874

Giới hạn nhỏ hơn float32 → dễ overflow hơn.

Giải pháp:
	•	Loss scaling:
\mathcal{L}' = S \cdot \mathcal{L}

Sau đó chia gradient cho S.

⸻

8. Softmax và Nhiệt độ (Temperature Scaling)

Softmax có thể điều chỉnh bằng nhiệt độ T:

\sigma(z_i)
=
\frac{\exp(z_i/T)}
{\sum_j \exp(z_j/T)}
	•	T \rightarrow 0: phân phối sắc nét
	•	T \rightarrow \infty: phân phối gần đều

Tuy nhiên nếu T quá nhỏ → logits hiệu dụng tăng → dễ overflow.

⸻

9. Phân tích Lý thuyết Xác suất

Softmax là nghiệm của bài toán tối ưu:

\max_p
\left(
\sum_i p_i z_i
-
\sum_i p_i \log p_i
\right)

Đây là dạng tối ưu hóa entropy tối đa.

⸻

10. Kết luận

Các vấn đề số học trong logits và softmax xuất phát từ:
	•	Hàm mũ tăng nhanh
	•	Giới hạn biểu diễn số dấu chấm động
	•	Gradient lan truyền

Giải pháp cốt lõi:

\textbf{Log-Sum-Exp Trick}

Đảm bảo:

\sigma(z_i)
=
\frac{\exp(z_i - z_{max})}
{\sum_j \exp(z_j - z_{max})}

Ổn định số học là điều kiện tiên quyết để huấn luyện mô hình sâu thành công, đặc biệt trong các hệ thống lớn như mô hình ngôn ngữ hiện đại.

⸻

Tài liệu tham khảo
	1.	Goodfellow, I., Bengio, Y., & Hinton, G. (2016). Deep Learning. MIT Press.
	2.	Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms. SIAM.
	3.	Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
	4.	Goldberg, D. (1991). What Every Computer Scientist Should Know About Floating-Point Arithmetic.
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
| 📌 **[Các Vấn Đề Số Học trong Logits và Softmax: Phân Tích Toán Học và Giải Pháp Ổn Định](aero_llm_02_numerical_issues_in_logits_and_softmax.md)** | [Xem bài viết →](aero_llm_02_numerical_issues_in_logits_and_softmax.md) |
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
| [aero llm 14 assessing bias and fairness](aero_llm_14_assessing_bias_and_fairness.md) | [Xem bài viết →](aero_llm_14_assessing_bias_and_fairness.md) |
| [aero llm 15 non technical benchmarks](aero_llm_15_non_technical_benchmarks.md) | [Xem bài viết →](aero_llm_15_non_technical_benchmarks.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
