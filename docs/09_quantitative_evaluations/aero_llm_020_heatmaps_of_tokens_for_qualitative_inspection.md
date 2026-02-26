
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
# Bản Đồ Nhiệt Của Token Cho Cân Nhắc Định Tính (Text Heatmaps)

## Tóm tắt

Các phân tích nội tại của một Mô hình Ngôn ngữ Lớn sinh ra lượng lớn thông tin về số liệu vô tri khá trừu tượng. Để có cảm quan (intuition) về cách LLM hoạt động trên văn bản con người, phương pháp tạo ra **Bản đồ nhiệt văn bản** (Text Heatmaps) trở nên phổ biến. Bài viết nêu bật sự liên kết số tĩnh vào nền của chuỗi con từ tự nhiên, chuyển hóa thông số phân rã định lượng trở thành trực quan định tính.

---

## 1. Phương Pháp Lập Bản Đồ Nhiệt Văn Bản

Mỗi một `token` ($t_i$) ứng với một con số cụ thể thể hiện một đại lượng $X_i$ cho LLM. Kỹ thuật sau sử dụng sự đối sánh trực tiếp để tô màu vào hộp văn bản theo thông số liên kết.

### 1.1 Tính Toán Kích Cỡ

Do môi trường lập trình thường xuất dữ liệu thông qua cửa sổ hiển thị (như matplotlib), các chữ cái (characters) cần sử dụng một font đồng nhịp như Monospace để tính diện tích.

Với thiết lập: `Figure = 10 \times 2`, tỷ lệ cố định của 1 token sẽ được chuyển thành giá trị hình hộp (bounding box) cụ thể có tọa độ và chiều dài được lấy trực tiếp bởi thuật toán đồ họa. Từ đó lấy làm đơn vị cho $t_1, t_2...$

### 1.2 Biến Đổi Tỷ Lệ (Min-Max Scaling)

Để vẽ bản đồ nhiệt dựa trên sự chuyển sắc (color map - như đỏ nhạt sang đô), tập số nội tại cần được liên kết lên một khoảng giá trị tiêu chuẩn từ $0$ tới $1$. Phép biến đổi chuẩn được sử dụng là **Min-Max Scaling**.

Giả sử $x_i$ là số lượng ký tự trong chuỗi chữ $i$:

$$x_{norm} = \frac{x_i - X_{min}}{X_{max} - X_{min}}$$

Phép đổi chuẩn là tuyến tính (linear transformation). Nó không phá vỡ tính tương quan gốc rễ mà chỉ co ép số liệu vào khuôn khổ $[0,1]$ nhằm kết xuất màu thông qua hệ số RGB.

---

## 2. Ứng Dụng vào Ví Dụ Thực Tế

Ban đầu, thay vì gắn kích hoạt (activations) từ mạng Neural, bản vẽ Heatmap được giả lập thông qua độ dài dòng chữ `Lorem Ipsum`. Chữ có màu đỏ càng đậm ứng với các từ kéo dài (nhiều ký tự), chữ sáng trắng thuộc các phần tử từ vụn ngắn.

Điều này mô phỏng các giá trị logit nội bộ $Z$ (sẽ được tìm trong quá trình huấn luyện/trích xuất mô hình):
$$Z \rightarrow \text{Softmax}(\cdot) \rightarrow P_i \rightarrow X_i$$
Càng đậm màu tương đương với năng lực dự đoán tiếp theo càng chính xác định tính.

---

## 3. Thuận Lợi Và Rủi Ro

Mặc dù có nhiều lợi ích:
- Làm trực quan sự liên kết của vô vàn chỉ số mạng NN với quá trình sinh ra chữ của trí thông minh.
- Phân tách ra từng từ (hoặc Sub-word) rõ ràng.

Nhưng cũng hiện diện cả nguy cơ diễn giải sai lệch (over interpretation) vì nhiễu hoặc các mẫu ngẫu nhiên (noise and unrepresentative examples). Con người rất nhạy cảm với hình ảnh màu sắc và dễ gắn cho nó các quy luật giả (Phantom patterns), dù cho đôi khi số liệu đó bị sai hoặc lỗi.

---

## Tài liệu tham khảo

1. **Rethmeier, N. et al. (2020).** *Visualizing and Understanding the Interpretability of Natural Language.*
2. **Karpathy, A. (2015).** *The Unreasonable Effectiveness of Recurrent Neural Networks.* Blog.
3. **Elhage, N. et al. (2021).** *A Mathematical Framework for Transformer Circuits.* Anthropic.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Đánh Giá Hộp Đen (Black-box Evaluations) trong Mô Hình Ngôn Ngữ Lớn](aero_llm_016_black_box_evals.md) | [Xem bài viết →](aero_llm_016_black_box_evals.md) |
| [Red Teaming: Đội Đỏ và Thử Nghiệm Đối Kháng trong AI Safety](aero_llm_017_red_teaming.md) | [Xem bài viết →](aero_llm_017_red_teaming.md) |
| [Độ Chính Xác, Tính Mạch Lạc và Sự Phù Hợp trong Đánh Giá Mô Hình Ngôn Ngữ](aero_llm_018_accuracy_coherence_and_relevance.md) | [Xem bài viết →](aero_llm_018_accuracy_coherence_and_relevance.md) |
| [Phân Phối Của Các Kích Hoạt Trạng Thái Ẩn Trong Mô Hình Ngôn Ngữ](aero_llm_019_distributions_of_hidden_state_activations.md) | [Xem bài viết →](aero_llm_019_distributions_of_hidden_state_activations.md) |
| [Hứa Hẹn và Thách Thức của Đánh Giá Định Lượng trong Mô Hình Học Máy](aero_llm_01_promises_and_challenges_of_quantitative_evaluations.md) | [Xem bài viết →](aero_llm_01_promises_and_challenges_of_quantitative_evaluations.md) |
| 📌 **[Bản Đồ Nhiệt Của Token Cho Cân Nhắc Định Tính (Text Heatmaps)](aero_llm_020_heatmaps_of_tokens_for_qualitative_inspection.md)** | [Xem bài viết →](aero_llm_020_heatmaps_of_tokens_for_qualitative_inspection.md) |
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
