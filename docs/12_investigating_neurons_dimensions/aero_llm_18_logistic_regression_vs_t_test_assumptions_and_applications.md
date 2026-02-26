
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
# Đối chiếu Hồi quy Logistic và Kiểm định T-test: Giả định và Ứng dụng

## Tóm tắt (Abstract)
Báo cáo này thực hiện một phân tích đối chiếu giữa Hồi quy Logistic và Kiểm định T-test (T-test) – hai phương pháp thống kê phổ biến nhất trong việc phân tích hành vi nơ-ron. Thông qua việc phân tích cấu trúc toán học và thực nghiệm trên dữ liệu giả lập, chúng ta làm rõ các kịch bản sử dụng tối ưu cho từng phương pháp. Mặc dù có những điểm khác biệt về giả định (đặc biệt là giả định về phân phối chuẩn), kết quả thực nghiệm bộc lộ một sự tương hợp (concordance) mạnh mẽ về ý nghĩa thống kê giữa hai phương pháp khi áp dụng trên cùng một bộ dữ liệu hoạt hóa nơ-ron.

---

## 1. Phân tích Cấu trúc Toán học

### 1.1. Hồi quy Logistic
- **Mục tiêu:** Xây dựng phương trình toán học để tạo ra xác suất một mẫu dữ liệu thuộc về một trong hai danh mục.
- **Kết quả:** Cung cấp các tham số ($\beta$) dùng để xây dựng mô hình dự báo.
- **Ứng dụng:** Thích hợp khi cần dự đoán nhãn (label) của token dựa trên hoạt hóa nơ-ron hoặc khi phân tích đa biến (nhiều nơ-ron cùng lúc).

### 1.2. Kiểm định T-test
- **Mục tiêu:** Đo lường sự khác biệt chuẩn hóa giữa giá trị trung bình của hai nhóm dữ liệu (ví dụ: Nouns vs. Verbs).
- **Kết quả:** Cung cấp trị số thống kê $t$ – một thước đo về quy mô hiệu ứng (effect size).
- **Ứng dụng:** Thích hợp để xác định sự khác biệt có ý nghĩa giữa hai loại token trong một nơ-ron cụ thể.

---

## 2. So sánh Giả định và Đặc tính

| Tiêu chí | Hồi quy Logistic | Kiểm định T-test |
| :--- | :--- | :--- |
| **Giả định Phân phối** | Không yêu cầu phân phối chuẩn. | Yêu cầu dữ liệu có phân phối chuẩn. |
| **Giả định Quần thể** | Mẫu đến từ một quần thể có tỉ lệ thuộc về danh mục thay đổi theo hoạt hóa. | Hai nhóm đến từ hai quần thể thực sự khác biệt. |
| **Khả năng dự báo** | Cho phép dự đoán xác suất ở cấp độ từng mẫu dữ liệu đơn lẻ. | Không thiết kế để dự đoán cho mẫu đơn lẻ. |
| **Tính mở rộng** | Dễ dàng mở rộng cho nhiều biến độc lập. | Giới hạn trong một biến phụ thuộc và hai nhóm. |

---

## 3. Thực nghiệm Đối chiếu trên Dữ liệu Giả lập

### 3.1. Sự Tương hợp về Hệ số và Trị số T
Thực nghiệm mô phỏng dữ liệu với quy mô hiệu ứng thay đổi cho thấy sự tương quan cực kỳ chặt chẽ giữa hệ số $\beta$ của Hồi quy Logistic và trị số $t$.
- **Lưu ý về tính ổn định:** Tại các giá trị cực hạn (nơi dữ liệu hai nhóm tách biệt hoàn toàn), Hồi quy Logistic bộc lộ sự không ổn định về mặt số học (numerical instability) do tính chất phi tuyến của thuật toán ước lượng, trong khi T-test vẫn giữ được tính ổn định tuyến tính.

### 3.2. Sự Tương hợp về Giá trị P (P-values)
Sử dụng giá trị $-$\log$(p)$ để trực quan hóa toàn bộ dải phân phối. Kết quả khẳng định:
- Khi kết quả không có ý nghĩa thống kê ở phương pháp này, nó cũng thường không có ý nghĩa ở phương pháp kia.
- Khi một nơ-ron được xác định là "có ý nghĩa" ($p < 0.05$), cả hai phương pháp đều đưa ra kết luận đồng nhất.

---

## 4. Các điểm lưu ý Kỹ thuật

1. **Thứ tự Nhãn (Label Ordering):** Việc thay đổi thứ tự nhập dữ liệu (ví dụ: Noun trước Verb hoặc ngược lại) chỉ làm thay đổi dấu (sign) của hệ số $\beta$ hoặc giá trị $t$ mà không làm thay đổi bản chất thống kê.
2. **Tính Tương đương Toán học:** Kiểm định T-test mẫu cặp (Paired samples T-test) về mặt toán học là tương đương với kiểm định T-test một mẫu (One-sample T-test) trên hiệu số giữa các cặp dữ liệu.

---

## 5. Kết Luận
Việc lựa chọn giữa Hồi quy Logistic và T-test phụ thuộc vào câu hỏi nghiên cứu:
- Chọn **Hồi quy Logistic** khi muốn xây dựng "bộ giải mã" (decoder) để dự đoán danh mục từ hoạt hóa.
- Chọn **T-test** khi muốn kiểm chứng nhanh sự khác biệt đặc tính giữa các nhóm nơ-ron hoặc các đầu Attention.

---

## Tài liệu tham khảo (Citations)
1. Đối chiếu Hồi quy Logistic và T-test trên LLM dựa trên `aero_LLM_18_Logistic regression vs. t-test assumptions and applications.md`. Phân tích tương hợp p-value và tính ổn định số học.
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
| [Thử thách Lập trình: Tính Chọn lọc Ngữ pháp của Nơ-ron MLP (Phần 2)](aero_llm_09_codechallenge_grammar_tuning_in_mlp_neurons_part_2_.md) | [Xem bài viết →](aero_llm_09_codechallenge_grammar_tuning_in_mlp_neurons_part_2_.md) |
| [Thử thách Lập trình: Sự Điều chế Ngữ cảnh trong Hoạt hóa MLP (Context-modulated Activation)](aero_llm_10_codechallenge_context_modulated_activation_in_mlp.md) | [Xem bài viết →](aero_llm_10_codechallenge_context_modulated_activation_in_mlp.md) |
| [Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 1)](aero_llm_11_codechallenge_activation_histograms_by_token_length_part_1_.md) | [Xem bài viết →](aero_llm_11_codechallenge_activation_histograms_by_token_length_part_1_.md) |
| [Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 2)](aero_llm_12_codechallenge_activation_histograms_by_token_length_part_2_.md) | [Xem bài viết →](aero_llm_12_codechallenge_activation_histograms_by_token_length_part_2_.md) |
| [Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 3)](aero_llm_13_codechallenge_activation_histograms_by_token_length_part_3_.md) | [Xem bài viết →](aero_llm_13_codechallenge_activation_histograms_by_token_length_part_3_.md) |
| [Xử lý Biểu diễn Nơ-ron cho các Từ đa Token (Multi-token Words)](aero_llm_14_dealing_with_multitoken_word_embeddings.md) | [Xem bài viết →](aero_llm_14_dealing_with_multitoken_word_embeddings.md) |
| [Thử thách Lập trình: Hình chiếu MLP Điều chỉnh theo Danh mục (Phần 1)](aero_llm_15_codechallenge_category_tuned_mlp_projections_part_1_.md) | [Xem bài viết →](aero_llm_15_codechallenge_category_tuned_mlp_projections_part_1_.md) |
| [Thử thách Lập trình: Hình chiếu MLP Điều chỉnh theo Danh mục (Phần 2)](aero_llm_16_codechallenge_category_tuned_mlp_projections_part_2_.md) | [Xem bài viết →](aero_llm_16_codechallenge_category_tuned_mlp_projections_part_2_.md) |
| [Hồi quy Logistic: Lý thuyết và Triển khai Phân loại Nơ-ron](aero_llm_17_classification_via_logistic_regression_theory_and_code.md) | [Xem bài viết →](aero_llm_17_classification_via_logistic_regression_theory_and_code.md) |
| 📌 **[Đối chiếu Hồi quy Logistic và Kiểm định T-test: Giả định và Ứng dụng](aero_llm_18_logistic_regression_vs_t_test_assumptions_and_applications.md)** | [Xem bài viết →](aero_llm_18_logistic_regression_vs_t_test_assumptions_and_applications.md) |
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
