
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
Nhập và triển khai mô hình ngôn ngữ lớn bằng lượng tử hóa 8-bit/4-bit với BitsAndBytes

Phân tích kiến trúc, cơ sở toán học và hiệu năng thực nghiệm

⸻

Tóm tắt

Bài viết này phân tích phương pháp nhập và triển khai mô hình ngôn ngữ lớn (LLMs) bằng kỹ thuật lượng tử hóa (quantization) sử dụng thư viện bitsandbytes. Dựa trên nội dung tài liệu đính kèm, chúng tôi mở rộng với các nền tảng lý thuyết từ Transformer của Ashish Vaswani et al. (2017), nghiên cứu về scaling laws của OpenAI và hệ sinh thái triển khai của Hugging Face.

Bài viết trình bày:
	•	Bài toán giới hạn bộ nhớ khi tải LLM
	•	Nguyên lý lượng tử hóa trọng số 8-bit và 4-bit
	•	Công thức sai số lượng tử hóa
	•	Phân tích độ phức tạp bộ nhớ
	•	So sánh hiệu năng trước và sau lượng tử hóa

⸻

1. Giới thiệu

Mô hình ngôn ngữ lớn hiện đại có số tham số từ:

10^9 \rightarrow 10^{11}

Giả sử:
	•	Mô hình có N tham số
	•	Mỗi tham số ở dạng FP32 (4 bytes)

Dung lượng bộ nhớ:

Memory = 4N \text{ bytes}

Ví dụ:

N = 7 \times 10^9

Memory = 28GB

Điều này vượt quá khả năng của nhiều GPU phổ thông.

⸻

2. Nguyên lý lượng tử hóa (Quantization)

2.1 Định nghĩa

Lượng tử hóa là ánh xạ:

w \in \mathbb{R} \rightarrow \hat{w} \in \mathbb{Z}_k

Trong đó:
	•	k = 2^b
	•	b là số bit (8-bit, 4-bit,…)

⸻

2.2 Lượng tử hóa tuyến tính (Linear Quantization)

Cho trọng số w nằm trong khoảng:

$$
w_{min}, w_{max}
$$
Hệ số scale:

s = \frac{w_{max} - w_{min}}{2^b - 1}

Giá trị lượng tử hóa:

\hat{w} = \text{round}\left$\frac{w - w_{min}}{s}\right$

Giải lượng tử:

w \approx s \hat{w} + w_{min}

⸻

3. Sai số lượng tử hóa

Sai số:

\epsilon = w - \hat{w}

Giả sử phân phối đều:

Var$\epsilon$ = \frac{s^2}{12}

Khi giảm số bit b:
	•	s tăng
	•	Sai số tăng
	•	Mất mát thông tin tăng

⸻

4. 8-bit vs 4-bit

4.1 Bộ nhớ

Với FP32:

Memory_{32} = 32N \text{ bits}

Với 8-bit:

Memory_{8} = 8N \text{ bits}

Giảm:

\frac{Memory_{8}}{Memory_{32}} = \frac{1}{4}

Với 4-bit:

Memory_{4} = 4N \text{ bits}

Giảm:

\frac{Memory_{4}}{Memory_{32}} = \frac{1}{8}

⸻

4.2 Ảnh hưởng đến forward pass

Transformer sử dụng:

Y = XW

Sau lượng tử hóa:

Y = X\hat{W}

Sai số lan truyền:

\Delta Y = X$W - \hat{W}$

Nếu:

||W - \hat{W}||_2 \text{ nhỏ}

→ Ảnh hưởng tới output nhỏ.

⸻

5. Kỹ thuật của BitsAndBytes

Thư viện bitsandbytes triển khai:
	•	Lượng tử hóa động (dynamic quantization)
	•	Lượng tử hóa theo block
	•	NF4 (NormalFloat4)

NF4 giả định trọng số phân phối chuẩn:

w \sim \mathcal{N}$0, \sigma^2$

Mapping phi tuyến giúp giảm sai số so với lượng tử hóa tuyến tính.

⸻

6. Tích hợp với Hugging Face Transformers

Hệ sinh thái của Hugging Face hỗ trợ:
	•	load_in_8bit=True
	•	load_in_4bit=True

Giảm bộ nhớ GPU đáng kể mà không cần huấn luyện lại toàn bộ mô hình.

⸻

7. Ảnh hưởng đến Perplexity

Perplexity:

PP = \exp\left$- \frac{1}{N} \sum \log P(w_i$\right)

Sau lượng tử hóa:

PP_{quant} = PP_{fp32} + \delta

Trong thực nghiệm:
	•	8-bit: \delta \approx 1\% - 3\%
	•	4-bit: \delta \approx 3\% - 8\%

Phụ thuộc kích thước mô hình.

⸻

8. Phân tích độ phức tạp tính toán

Phép nhân ma trận:

O$n^3$

Nhưng khi dùng int8:
	•	Giảm băng thông bộ nhớ
	•	Tăng throughput
	•	Tối ưu Tensor Core

Tốc độ thực tế tăng 1.5–2x trên GPU hỗ trợ INT8.

⸻

9. Lượng tử hóa và Scaling Law

Theo nghiên cứu scaling law của OpenAI:

Loss$N$ = A N^{-\alpha}

Nếu lượng tử hóa làm tăng loss một lượng nhỏ \delta,
thì có thể bù bằng tăng nhẹ số tham số N.

⸻

10. So sánh với Pruning

Kỹ thuật	Giảm bộ nhớ	Giảm FLOPs	Ảnh hưởng độ chính xác
Quantization	✔	✖	Thấp–Trung
Pruning	✔	✔	Trung
Distillation	✔	✔	Thấp

Quantization phù hợp cho triển khai inference.

⸻

11. Hạn chế
	•	Gradient không ổn định khi fine-tune trực tiếp 4-bit
	•	Một số layer nhạy cảm (LayerNorm, Embedding)
	•	Cần mixed-precision

⸻

12. Kết luận

Lượng tử hóa bằng bitsandbytes:
	•	Giảm 4–8 lần bộ nhớ
	•	Giữ chất lượng gần tương đương FP32
	•	Phù hợp triển khai LLM trên GPU tầm trung

Trong tương lai:
	•	QLoRA
	•	Post-training quantization nâng cao
	•	Mixed precision adaptive

⸻

Tài liệu tham khảo
	1.	Vaswani, A. et al. (2017). Attention is All You Need.
	2.	Dettmers, T. et al. (2022). 8-bit Optimizers via Block-wise Quantization.
	3.	Kaplan et al. (2020). Scaling Laws for Neural Language Models.
	4.	Goodfellow et al. (2016). Deep Learning.
	5.	Hugging Face Transformers Documentation.
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
| 📌 **[aero llm 07 import large models using bitsandbytes](aero_llm_07_import_large_models_using_bitsandbytes.md)** | [Xem bài viết →](aero_llm_07_import_large_models_using_bitsandbytes.md) |
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
