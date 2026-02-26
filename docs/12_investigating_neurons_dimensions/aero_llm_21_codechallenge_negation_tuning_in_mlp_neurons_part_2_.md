
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [12 investigating neurons dimensions](../index.md)

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
# Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 2)

## Tóm tắt (Abstract)
Báo cáo này tiếp tục quy trình nghiên cứu về "nơ-ron phủ định" bằng cách triển khai phân tích hồi quy logistic diện rộng trên quy mô hàng nghìn đơn vị nơ-ron. Chúng ta tập trung vào việc phát triển bộ phân loại (classifier) để định lượng khả năng phân biệt giữa các khái niệm Phủ định và Khẳng định của từng nơ-ron tại một tầng Transformer cụ thể. Quy trình bao gồm việc xử lý các thách thức về hội tụ số học, đánh giá độ chính xác của mô hình dự báo và trực quan hóa định tính thông qua bản đồ nhiệt văn bản nhằm xác thực tính chọn lọc chức năng của nơ-ron.

---

## 1. Triển khai Hồi quy Logistic diện rộng (Exercise 4)

### 1.1. Thiết lập Mô hình và Nhãn danh mục
Chúng ta xây dựng vector nhãn `category_labels` có kích thước tương ứng với tổng số mẫu:
- **Nhãn 0:** Các từ Khẳng định (Affirmations).
- **Nhãn 1:** Các từ Phủ định (Negations).
Hệ số Beta dương ($\beta > 0$) sẽ trực tiếp đồng nghĩa với việc nơ-ron nhạy cảm hơn với các cấu trúc phủ định.

### 1.2. Kỹ thuật Xử lý Hồi quy trên 5120 nơ-ron
Do tính chất đa dạng của dữ liệu hoạt hóa nơ-ron, một số đơn vị có thể gây lỗi cho thuật toán ước lượng phi tuyến. Các biện pháp kỹ thuật được áp dụng bao gồm:
1. **Tăng cường lặp:** Thiết lập `maxiter=3000` để hỗ trợ hội tụ trong các trường hợp phân tách dữ liệu phức tạp.
2. **Khối ngoại lệ (Try-Except):** Bảo vệ chương trình khỏi bị dừng đột ngột bởi các nơ-ron có dữ liệu quá nhiễu hoặc tách rời hoàn hảo (perfect separability), đồng thời đánh dấu các trường hợp này bằng giá trị `NaN`.
3. **Phân tách Tham số:** Chỉ tập trung vào hệ số góc (slope) của biến nhãn, loại bỏ tham số hằng số (intercept) vì nó chỉ đại diện cho mức hoạt hóa nền của nơ-ron.

### 1.3. Đánh giá Độ chính xác Dựa trên Xác suất
Với nơ-ron có hiệu ứng mạnh nhất (ví dụ: index 2022 tại tầng 13), chúng ta sử dụng hàm `predict()` để thu được xác suất logit. Áp dụng ngưỡng 0.5 để so sánh với nhãn thực tế, từ đó tính toán được **Độ chính xác (Accuracy)**. Kết quả thực nghiệm cho thấy một số nơ-ron đơn lẻ có khả năng phân loại đúng các mẫu phủ định với độ chính xác vượt trội so với mức ngẫu nhiên.

---

## 2. Trực quan hóa Bản đồ nhiệt Văn bản (Exercise 5)

### 2.1. Phân tích Định tính nơ-ron "Vô địch"
Để hiểu rõ hơn về hành vi của nơ-ron có hệ số Beta cao nhất, chúng ta ánh xạ hoạt hóa của nó lên chuỗi từ ngữ thực tế. Quy trình thực hiện:
- **Min-Max Scaling:** Chuẩn hóa biên độ hoạt hóa về dải $[0, 1]$ để phù hợp với thang màu (Colormap).
- **Bản đồ nhiệt (Heatmap):** Các từ phủ định như "not", "won't" thường xuyên kích hoạt mức "sáng" cao nhất trên bản đồ, trong khi các từ như "can", "will" trong cùng một ngữ cảnh lại có mức hoạt hóa thấp.

---

## 3. Thảo luận về Ý nghĩa Thống kê
Mặc dù nơ-ron có hệ số Beta lớn nhất thường có ý nghĩa thống kê cao, nhưng chúng không nhất thiết là nơ-ron có $p$-value nhỏ nhất. Sự khác biệt này đến từ sự cân bằng giữa quy mô hiệu ứng (effect size) và độ biến thiên (variance) của dữ liệu. Hiện tượng này nhấn mạnh tầm quan trọng của việc kết hợp cả chỉ số tham số ($\beta$) và độ tin cậy ($p$) trong Mechanistic Interpretability.

---

## 4. Kết Luận Phần 2
Chúng ta đã chứng minh được rằng lớp MLP chứa các đơn vị chức năng có khả năng hoạt động như "bộ phát hiện phủ định" (negation detectors). Trong giai đoạn tiếp theo, nghiên cứu sẽ mở rộng phạm vi ra toàn bộ 36 tầng của GPT-2 Large để tìm kiếm sự phân bổ của các nơ-ron này trong toàn bộ cấu trúc mạng.

---

## Tài liệu tham khảo (Citations)
1. Hồi quy Logistic xuyên tầng trên GPT-2 Large dựa trên `aero_LLM_21_CodeChallenge Negation tuning in MLP neurons (part 2).md`. Phân tích hệ số Beta và độ chính xác phân loại.
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
| [Đối chiếu Hồi quy Logistic và Kiểm định T-test: Giả định và Ứng dụng](aero_llm_18_logistic_regression_vs_t_test_assumptions_and_applications.md) | [Xem bài viết →](aero_llm_18_logistic_regression_vs_t_test_assumptions_and_applications.md) |
| [Điều chỉnh Danh từ riêng trong GPT-2 Medium](aero_llm_19_proper_noun_tuning_in_gpt2_medium.md) | [Xem bài viết →](aero_llm_19_proper_noun_tuning_in_gpt2_medium.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 1)](aero_llm_20_codechallenge_negation_tuning_in_mlp_neurons_part_1_.md) | [Xem bài viết →](aero_llm_20_codechallenge_negation_tuning_in_mlp_neurons_part_1_.md) |
| 📌 **[Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 2)](aero_llm_21_codechallenge_negation_tuning_in_mlp_neurons_part_2_.md)** | [Xem bài viết →](aero_llm_21_codechallenge_negation_tuning_in_mlp_neurons_part_2_.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 3)](aero_llm_22_codechallenge_negation_tuning_in_mlp_neurons_part_3_.md) | [Xem bài viết →](aero_llm_22_codechallenge_negation_tuning_in_mlp_neurons_part_3_.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron QVK (Attention)](aero_llm_23_codechallenge_negation_tuning_in_qvk_neurons.md) | [Xem bài viết →](aero_llm_23_codechallenge_negation_tuning_in_qvk_neurons.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
