
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
# Phân Phối Của Các Kích Hoạt Trạng Thái Ẩn Trong Mô Hình Ngôn Ngữ

## Tóm tắt

Ngay cả khi có quyền truy cập vào toàn bộ các thông số nội bộ của một mô hình ngôn ngữ lớn (LLM), bản chất phi tuyến tính và phức tạp của chúng làm cho việc hiểu cách mô hình nhận thức và xử lý thông tin trở nên rất khó khăn. Bài viết này khám phá các phương pháp trích xuất mẫu kích hoạt nội tại từ các lớp `transformer` và trực quan hóa phân phối của chúng thông qua biểu đồ phân tán (scatter plots), ma trận hiệp phương sai (covariance matrix), và biểu đồ tần suất (histograms).

---

## 1. Cơ sở về Trạng Thái Ẩn (Hidden States)

Trong một LLM như GPT-2, văn bản đầu vào được mã hóa thành các chỉ số token, sau đó được ánh xạ thành các **vectors nhúng** (embedding vectors). Tại mỗi khối `transformer`, các vector này lại được biến đổi, quay, co giãn, để rồi hình thành nên biểu diễn cuối cùng cho việc dự đoán token.

Bằng cách chạy một lượt lan truyền xuôi (forward pass), ta có thể kích hoạt tùy chọn xuất trạng thái ẩn:

`output_hidden_states = True`

Trong GPT-2 nhỏ, tính toán này sẽ trả ra 13 ten-xơ (tensors), bao gồm:
1 đầu ra từ Lớp Nhúng (Embeddings layer).
12 đầu ra tương ứng từ 12 khối transformer.
Mỗi mạng lưới có cấu hình kích thước dạng `[Batch Size, Sequences, Embedding Dimension]`. Trong GPT, thiết lập này thường là `[1, 62, 768]`.

---

## 2. Các Công Cụ Trực Quan Hóa 

### 2.1 Biểu Đồ Phân Tán (Scatter Plots)

Với biểu đồ phân tán, ta đối chiếu các chỉ số token và chiều biểu diễn (embedding dimensions) với giá trị kích hoạt.

Điểm quan trọng rút ra là **yếu tố nhiễu của token đầu tiên**. Trong tự nhiên, việc xử lý token đầu tiên là phi chuẩn vì không có context (ngữ cảnh) đứng trước nó. Để việc quan sát không bị sai lệch, thông thường token này cần bị loại trừ (sử dụng token có chỉ số 1 trở lên).

### 2.2 Ma Trận Hiệp Phương Sai và $R^2$ (Covariance & $R^2$ Matrix)

Để hiểu được các phép tính ẩn liên đới như thế nào qua từng lớp, ta sử dụng ma trận **Hiệp phương sai** (Covariance) và ma trận tương quan được bình phương ($R^2$, giải thích lượng phương sai được chia sẻ). 

$$ R^2 = \text{Corr}(X, Y)^2 $$

Hai đại lượng $X$ và $Y$ hoàn toàn không tương quan sẽ có $R^2 \approx 0$. Ngược lại, nếu chúng giống hệt, kết quả trả về 1 (hoặc 100%).

- Lớp nhúng (Embeddings) và Khối transformer kết quả (Layer cuối) có rất ít sự hiệp biến so với các blocks khác. 
- Giữa khối Transformer trung gian, có sự chia sẻ phương sai khá mật thiết. Vectơ embedding thay đổi qua mỗi block, nhưng không đột biến. Có sự tinh chỉnh từ từ, làm cấu trúc giống như chiếc "cánh chim" nở rộng ra dần.

### 2.3 Biểu Đồ Tần Suất (Histograms)

Sử dụng phân Scale logarith của trục $y$ cho biểu đồ Histogram giúp biểu thị các giá trị lệch (side lobes) của phân phối theo cách nhạy bén nhất.

- Tại lớp nhúng ban đầu, phân phối khá hẹp do thiếu ngữ cảnh.
- Càng đi sâu, phân phối tản mạn hơn với nhiều điểm có các Activation Values mang tính thái quá (extreme activation). Mở rộng sự tham khảo tri thức trên pre-trained weights.
- Ở block Transformer cuối cùng, đường cong phân phối lại hẹp dần. Điều này mang ý nghĩa sự không chắc chắn (Uncertainty) bị triệt tiêu, và model tập trung ưu tiên bỏ phiếu cho các token mạnh nhất để sinh chuỗi ký tự.

---

## 3. Ý Nghĩa

Mặc dù có được số liệu, việc hiểu được thực sự từng chuỗi kích hoạt ứng với pattern nào của thông tin ngôn ngữ vẫn rất thử thách. Phương pháp trực quan hóa số vĩ mô như Scatter hay Covariances giúp ta "chạm" gần hơn vào kiến trúc của **Khả năng Báo tích Cơ học (Mechanistic Interpretability)**.

---

## Tài liệu tham khảo

1. **Radford, A. et al. (2019).** *Language Models are Unsupervised Multitask Learners.*
2. **Elhage, N. et al. (2021).** *A Mathematical Framework for Transformer Circuits.* Anthropic.
3. **Clark, K. et al. (2019).** *What Does BERT Look At? An Analysis of BERT's Attention.*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Đánh Giá Hộp Đen (Black-box Evaluations) trong Mô Hình Ngôn Ngữ Lớn](aero_llm_016_black_box_evals.md) | [Xem bài viết →](aero_llm_016_black_box_evals.md) |
| [Red Teaming: Đội Đỏ và Thử Nghiệm Đối Kháng trong AI Safety](aero_llm_017_red_teaming.md) | [Xem bài viết →](aero_llm_017_red_teaming.md) |
| [Độ Chính Xác, Tính Mạch Lạc và Sự Phù Hợp trong Đánh Giá Mô Hình Ngôn Ngữ](aero_llm_018_accuracy_coherence_and_relevance.md) | [Xem bài viết →](aero_llm_018_accuracy_coherence_and_relevance.md) |
| 📌 **[Phân Phối Của Các Kích Hoạt Trạng Thái Ẩn Trong Mô Hình Ngôn Ngữ](aero_llm_019_distributions_of_hidden_state_activations.md)** | [Xem bài viết →](aero_llm_019_distributions_of_hidden_state_activations.md) |
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
| [aero llm 14 assessing bias and fairness](aero_llm_14_assessing_bias_and_fairness.md) | [Xem bài viết →](aero_llm_14_assessing_bias_and_fairness.md) |
| [aero llm 15 non technical benchmarks](aero_llm_15_non_technical_benchmarks.md) | [Xem bài viết →](aero_llm_15_non_technical_benchmarks.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
