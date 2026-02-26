
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [12 investigating neurons dimensions](index.md)

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
# Thử thách Lập trình: Tính Chọn lọc Ngữ pháp của Nơ-ron MLP (Phần 2)

## Tóm tắt (Abstract)
Báo cáo này hoàn thiện thử thách tìm kiếm "nơ-ron ngôn ngữ" bằng cách áp dụng các phép kiểm định thống kê trên dữ liệu hoạt hóa đã thu thập. Thông qua kiểm định t-test mẫu cặp (paired samples t-test) và hiệu chỉnh Bonferroni cho đa so sánh, nghiên cứu xác định được các nơ-ron có sự khác biệt về kích hoạt đạt mức ý nghĩa thống kê giữa danh từ và động từ. Giai đoạn cuối của thực nghiệm kiểm chứng tính tổng quát hóa (generalizability) của kết quả trên một văn bản Wikipedia hoàn toàn mới, sử dụng bản đồ nhiệt (heatmaps) để quan sát sự tương quan định tính. Kết quả cho thấy sự tồn tại của tính chọn lọc sơ khai, đồng thời chỉ ra những hạn chế cố hữu của việc phân tích nơ-ron đơn lẻ trong các hệ thống phân tán phức tạp.

---

## 1. Mở Đầu (Introduction)
Sau khi đã thu thập được ma trận hoạt hóa thô ở Phần 1, thách thức tiếp theo là tách biệt tín hiệu thực sự khỏi nhiễu ngẫu nhiên. Trong khoa học dữ liệu, việc quan sát thấy sự khác biệt bằng mắt thường là chưa đủ; chúng ta cần một khung xác suất để khẳng định liệu nơ-ron 512 có thực sự "ưa thích" danh từ hơn động từ hay đó chỉ là sự biến thiên ngẫu nhiên của mẫu thử.

---

## 2. Phân tích Thống kê (Statistical Analysis)

### 2.1. Kiểm định T-test và Hiệu chỉnh Đa so sánh
- **Phép thử:** Sử dụng `scipy.stats.ttest_1samp` trên giá trị hiệu số (difference scores) giữa hoạt hóa danh từ và động từ. Đây là cách tiếp cận tương đương với paired t-test nhằm cô lập biến số nơ-ron.
- **Hiệu chỉnh Bonferroni:** Với 3072 nơ-ron được kiểm định đồng thời, ngưỡng ý nghĩa $\alpha = 0.05$ là quá lỏng lẻo. Ngưỡng mới được thiết lập là $\alpha_{adj} = 0.05 / 3072 \approx 1.6 \times 10^{-5}$ để kiểm soát tỷ lệ lỗi loại I.

### 2.2. Phân loại Nơ-ron
- **T-value dương:** Nơ-ron kích hoạt mạnh hơn đáng kể cho Danh từ.
- **T-value âm:** Nơ-ron kích hoạt mạnh hơn đáng kể cho Động từ.
Thực nghiệm cho thấy một tỷ lệ nhỏ nơ-ron vượt qua ngưỡng Bonferroni, chứng minh tính chuyên biệt hóa không phải là ngẫu nhiên.

---

## 3. Kiểm chứng Tính Tổng quát hóa (Generalizability Test)

### 3.1. Dữ liệu Văn bản Mới
Sử dụng một đoạn văn bản trích từ Wikipedia về chủ đề "Ngẫu nhiên" (Randomness) – một ngữ cảnh hoàn toàn khác với các từ đơn lẻ ban đầu. Mục tiêu là xem liệu nơ-ron "đỉnh" vừa tìm được có phản ứng chính xác với các danh từ/động từ xuất hiện tự nhiên trong câu hay không.

### 3.2. Trực quan hóa bằng Heatmap
Văn bản được tô màu dựa trên cường độ hoạt hóa của hai nơ-ron cực đoan nhất:
- **Nơ-ron Danh từ (Max T-value):** Các từ như "entropy", "uncertainty", "information" được tô màu đỏ đậm. Các hư từ hoặc động từ có màu nhạt.
- **Nơ-ron Động từ (Min T-value):** Các từ như "is", "applies", "follow" có mức độ kích hoạt cao hơn (màu xanh đậm).

---

## 4. Thảo Luận: Hạn chế và Hướng đi tiếp theo
Dù kết quả mang tính khích lệ, báo cáo chỉ ra các rào cản quan trọng:
1. **Sự đa nghĩa (Polysemanticity):** Một nơ-ron có thể vừa chọn lọc danh từ, vừa phản ứng với một ký tự đặc biệt như dấu chấm phẩy (;).
2. **Vấn đề ngữ cảnh (Context Gap):** LLM vốn được huấn luyện để xử lý chuỗi. Việc kiểm tra từ đơn lẻ (out-of-context) có thể không phản ánh đúng chức năng thực tế của nơ-ron trong các mạch điện (circuits) phức tạp.
3. **Tính chọn lọc tương đối:** Để khẳng định "chọn lọc danh từ", cần kiểm soát thêm nhiều từ loại khác (tính từ, trạng từ) thay vì chỉ so sánh nhị phân.

---

## 5. Kết Luận
Thử thách này minh chứng rằng các thành phần nội bộ của LLM (đặc biệt là MLP) chứa đựng những cấu trúc ngôn ngữ có thể giải mã được. Mặc dù không hoàn hảo, nhưng phương pháp Hooks kết hợp với thống kê cổ điển mở ra một lối đi hứa hẹn cho việc "đọc vị" tư duy máy móc, chuyển từ quan sát hành vi đầu ra sang hiểu biết về các biểu diễn ngôn ngữ nội tại.

---

## Tài liệu tham khảo (Citations)
1. Kiểm định thống kê và tổng quát hóa tính chọn lọc nơ-ron trên GPT-Neo dựa trên `aero_LLM_09_CodeChallenge Grammar tuning in MLP neurons (part 2).md`. Phân tích T-values và kiểm chứng qua Heatmaps trên dữ liệu Wikipedia.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 12_investigating_neurons_dimensions](README.md) | [Xem bài viết →](README.md) |
| [Cực đại hóa Hoạt hóa (Activation Maximization): Cơ sở Lý thuyết và Những thách thức trong LLM](aero_llm_01_activation_maximization_via_gradient_ascent_theory_.md) | [Xem bài viết →](aero_llm_01_activation_maximization_via_gradient_ascent_theory_.md) |
| [Triển khai Cực đại hóa Hoạt hóa: Từ Gradient Ascent đến Giải mã Token (Activation Maximization Implementation)](aero_llm_02_activation_maximization_code_.md) | [Xem bài viết →](aero_llm_02_activation_maximization_code_.md) |
| [Cực đại hóa Hoạt hóa qua Lấy mẫu Dữ liệu (Activation Maximization via Data Sampling)](aero_llm_03_activation_maximization_via_data_sampling.md) | [Xem bài viết →](aero_llm_03_activation_maximization_via_data_sampling.md) |
| [Thử thách Lập trình: Kiểm chứng Tính lặp lại của Cực đại hóa Hoạt hóa (Reproducibility of Activation Maximization)](aero_llm_04_codechallenge_reproducibility_of_activation_maximization.md) | [Xem bài viết →](aero_llm_04_codechallenge_reproducibility_of_activation_maximization.md) |
| [Giải phẫu Nội tại Mô hình bằng Hooks: Kỹ thuật Trích xuất Hoạt hóa (Extracting Activations via Hooks)](aero_llm_05_extracting_activations_using_hooks.md) | [Xem bài viết →](aero_llm_05_extracting_activations_using_hooks.md) |
| [Mối tương quan giữa Hooks và Hidden States: Giải cấu trúc Khối Transformer (Reconstructing Transformer Blocks)](aero_llm_06_relation_between_hooks_and_output_hidden_states.md) | [Xem bài viết →](aero_llm_06_relation_between_hooks_and_output_hidden_states.md) |
| [Làm rõ về Hidden States Tầng cuối: Vai trò của LayerNorm (Clarification of Final Hidden States)](aero_llm_07_clarification_of_final_hidden_states_output.md) | [Xem bài viết →](aero_llm_07_clarification_of_final_hidden_states_output.md) |
| [Thử thách Lập trình: Tính Chọn lọc Ngữ pháp của Nơ-ron MLP (Phần 1)](aero_llm_08_codechallenge_grammar_tuning_in_mlp_neurons_part_1_.md) | [Xem bài viết →](aero_llm_08_codechallenge_grammar_tuning_in_mlp_neurons_part_1_.md) |
| 📌 **[Thử thách Lập trình: Tính Chọn lọc Ngữ pháp của Nơ-ron MLP (Phần 2)](aero_llm_09_codechallenge_grammar_tuning_in_mlp_neurons_part_2_.md)** | [Xem bài viết →](aero_llm_09_codechallenge_grammar_tuning_in_mlp_neurons_part_2_.md) |
| [Thử thách Lập trình: Sự Điều chế Ngữ cảnh trong Hoạt hóa MLP (Context-modulated Activation)](aero_llm_10_codechallenge_context_modulated_activation_in_mlp.md) | [Xem bài viết →](aero_llm_10_codechallenge_context_modulated_activation_in_mlp.md) |
| [Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 1)](aero_llm_11_codechallenge_activation_histograms_by_token_length_part_1_.md) | [Xem bài viết →](aero_llm_11_codechallenge_activation_histograms_by_token_length_part_1_.md) |
| [Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 2)](aero_llm_12_codechallenge_activation_histograms_by_token_length_part_2_.md) | [Xem bài viết →](aero_llm_12_codechallenge_activation_histograms_by_token_length_part_2_.md) |
| [Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 3)](aero_llm_13_codechallenge_activation_histograms_by_token_length_part_3_.md) | [Xem bài viết →](aero_llm_13_codechallenge_activation_histograms_by_token_length_part_3_.md) |
| [Xử lý Biểu diễn Nơ-ron cho các Từ đa Token (Multi-token Words)](aero_llm_14_dealing_with_multitoken_word_embeddings.md) | [Xem bài viết →](aero_llm_14_dealing_with_multitoken_word_embeddings.md) |
| [Thử thách Lập trình: Hình chiếu MLP Điều chỉnh theo Danh mục (Phần 1)](aero_llm_15_codechallenge_category_tuned_mlp_projections_part_1_.md) | [Xem bài viết →](aero_llm_15_codechallenge_category_tuned_mlp_projections_part_1_.md) |
| [Thử thách Lập trình: Hình chiếu MLP Điều chỉnh theo Danh mục (Phần 2)](aero_llm_16_codechallenge_category_tuned_mlp_projections_part_2_.md) | [Xem bài viết →](aero_llm_16_codechallenge_category_tuned_mlp_projections_part_2_.md) |
| [Hồi quy Logistic: Lý thuyết và Triển khai Phân loại Nơ-ron](aero_llm_17_classification_via_logistic_regression_theory_and_code.md) | [Xem bài viết →](aero_llm_17_classification_via_logistic_regression_theory_and_code.md) |
| [Đối chiếu Hồi quy Logistic và Kiểm định T-test: Giả định và Ứng dụng](aero_llm_18_logistic_regression_vs_t_test_assumptions_and_applications.md) | [Xem bài viết →](aero_llm_18_logistic_regression_vs_t_test_assumptions_and_applications.md) |
| [Điều chỉnh Danh từ riêng trong GPT-2 Medium](aero_llm_19_proper_noun_tuning_in_gpt2_medium.md) | [Xem bài viết →](aero_llm_19_proper_noun_tuning_in_gpt2_medium.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 1)](aero_llm_20_codechallenge_negation_tuning_in_mlp_neurons_part_1_.md) | [Xem bài viết →](aero_llm_20_codechallenge_negation_tuning_in_mlp_neurons_part_1_.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 2)](aero_llm_21_codechallenge_negation_tuning_in_mlp_neurons_part_2_.md) | [Xem bài viết →](aero_llm_21_codechallenge_negation_tuning_in_mlp_neurons_part_2_.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 3)](aero_llm_22_codechallenge_negation_tuning_in_mlp_neurons_part_3_.md) | [Xem bài viết →](aero_llm_22_codechallenge_negation_tuning_in_mlp_neurons_part_3_.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron QVK (Attention)](aero_llm_23_codechallenge_negation_tuning_in_qvk_neurons.md) | [Xem bài viết →](aero_llm_23_codechallenge_negation_tuning_in_qvk_neurons.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
