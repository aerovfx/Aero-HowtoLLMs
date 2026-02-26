
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
Phân kỳ Kullback–Leibler (KL Divergence) trong mô hình ngôn ngữ và học sâu

Phân tích lý thuyết, công thức toán học và ứng dụng trong tối ưu hóa phân phối xác suất

⸻

Tóm tắt

Bài viết này trình bày cơ sở lý thuyết và ứng dụng của phân kỳ Kullback–Leibler (KL Divergence) trong học máy và mô hình ngôn ngữ lớn (LLMs). Dựa trên nội dung tài liệu đính kèm, chúng tôi mở rộng phân tích với các nền tảng từ lý thuyết thông tin của Claude Shannon, công trình gốc của Solomon Kullback và Richard Leibler, cùng các ứng dụng hiện đại trong huấn luyện Transformer của Ashish Vaswani et al. và nghiên cứu RLHF tại OpenAI.

⸻

1. Giới thiệu

Trong học máy, ta thường cần đo khoảng cách giữa hai phân phối xác suất:
	•	Phân phối thực P$x$
	•	Phân phối mô hình Q$x$

Phân kỳ KL đo mức “mất mát thông tin” khi dùng Q để xấp xỉ P.

⸻

2. Định nghĩa toán học

2.1 Trường hợp rời rạc

D_{KL}$P \\mid  Q$ = \sum_{x} P$x$ \log \frac{P$x$}{Q$x$}

Điều kiện:

Q$x$ > 0 \quad \text{nếu } P$x$ > 0

⸻

2.2 Trường hợp liên tục

D_{KL}$P \\mid  Q$ = \int P$x$ \log \frac{P$x$}{Q$x$} dx

⸻

3. Các tính chất quan trọng

3.1 Không âm (Non-negativity)

D_{KL}$P \\mid  Q$ \ge 0

và

D_{KL}$P \\mid  Q$ = 0 \iff P = Q

Chứng minh dựa trên bất đẳng thức Jensen.

⸻

3.2 Không đối xứng

D_{KL}$P \\mid  Q$ \neq D_{KL}$Q \\mid  P$

Do đó KL không phải là metric.

⸻

4. Liên hệ với Cross-Entropy

Cross-entropy:

H(P, Q) = - \sum_x P$x$ \log Q$x$

Entropy:

H$P$ = - \sum_x P$x$ \log P$x$

Ta có:

D_{KL}$P \\mid  Q$ = H(P, Q) - H$P$

Trong huấn luyện mô hình, vì H$P$ không phụ thuộc vào tham số mô hình, nên tối thiểu hóa cross-entropy tương đương tối thiểu hóa KL divergence.

⸻

5. KL Divergence trong mô hình ngôn ngữ

Với mô hình dự đoán token:
	•	Phân phối thật: P_{data}
	•	Phân phối mô hình: P_\theta

Hàm mất mát:

\mathcal{L}$\theta$ = D_{KL}$P_{data} \\mid  P_\theta$

Tối ưu:

\theta^* = \arg\min_\theta D_{KL}$P_{data} \\mid  P_\theta$

⸻

6. Liên hệ với Perplexity

Perplexity:

PP = \exp\left$H(P_{data}, P_\theta$\right)

Vì:

H$P_{data}, P_\theta$ = H$P_{data}$ + D_{KL}$P_{data} \\mid  P_\theta$

→ Giảm KL → giảm perplexity.

⸻

7. KL Divergence trong RLHF

Trong Reinforcement Learning from Human Feedback (RLHF), ta tối ưu:

\max_\theta \mathbb{E}_{x \sim P_\theta}[R$x$] - \beta D_{KL}$P_\theta \\mid  P_{ref}$

Trong đó:
	•	R$x$: reward model
	•	P_{ref}: mô hình tham chiếu
	•	\beta: hệ số điều chỉnh

Thành phần KL giúp:
	•	Ngăn mô hình lệch quá xa mô hình gốc
	•	Tránh over-optimization

⸻

8. KL Divergence giữa hai phân phối chuẩn

Giả sử:

P = \mathcal{N}$\mu_1, \sigma_1^2$
Q = \mathcal{N}$\mu_2, \sigma_2^2$

Ta có:

D_{KL}$P \\mid  Q$ =
\log \frac{\sigma_2}{\sigma_1}
+ \frac{\sigma_1^2 + $\mu_1 - \mu_2$^2}{2\sigma_2^2}
- \frac{1}{2}

Công thức này thường dùng trong Variational Autoencoder (VAE).

⸻

9. KL Divergence và Self-Attention

Trong Transformer:

P_\theta$w_t$ = \text{softmax}$Wh_t$

Huấn luyện tối thiểu hóa:

D_{KL}$P_{data} \\mid  P_\theta$

Cơ chế self-attention:

Attention(Q,K,V) =
\text{softmax}\left$\frac{QK^T}{\sqrt{d_k}}\right$V

Giúp mô hình xây dựng phân phối xác suất chính xác hơn.

⸻

10. Trực giác thông tin học

Theo lý thuyết thông tin của Claude Shannon:
	•	Entropy đo độ bất định
	•	KL đo mức thông tin mất đi khi xấp xỉ phân phối

Nếu:

D_{KL}$P \\mid  Q$ = 2

→ Trung bình ta mất 2 nat thông tin mỗi mẫu.

⸻

11. Ứng dụng thực tế

11.1 Distillation

Giữa teacher T và student S:

\mathcal{L} = D_{KL}$P_T \\mid  P_S$

⸻

11.2 Regularization

Thêm điều khoản KL để:
	•	Giảm overfitting
	•	Kiểm soát divergence

⸻

11.3 Variational Inference

Tối ưu:

D_{KL}(q(z) \| p(z|x))

⸻

12. Hạn chế của KL Divergence
	1.	Không đối xứng
	2.	Nhạy khi Q$x$ \to 0
	3.	Không phải metric

Trong một số trường hợp, Jensen-Shannon divergence được dùng thay thế.

⸻

13. Kết luận

Phân kỳ KL là nền tảng của:
	•	Huấn luyện mô hình ngôn ngữ
	•	Cross-entropy loss
	•	Perplexity
	•	RLHF
	•	Distillation

Nó kết nối trực tiếp giữa lý thuyết thông tin và học sâu hiện đại.

⸻

Tài liệu tham khảo
	1.	Kullback, S., Leibler, R. (1951). On Information and Sufficiency.
	2.	Shannon, C. (1948). A Mathematical Theory of Communication.
	3.	Vaswani, A. et al. (2017). Attention is All You Need.
	4.	Goodfellow, I. et al. (2016). Deep Learning.
	5.	Ouyang et al. (2022). Training language models to follow instructions with human feedback.
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
| 📌 **[aero llm 10 kl kullback leibler divergence](aero_llm_10_kl_kullback_leibler_divergence.md)** | [Xem bài viết →](aero_llm_10_kl_kullback_leibler_divergence.md) |
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
