
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
# Thử thách Lập trình: Hình chiếu MLP Điều chỉnh theo Danh mục (Phần 2)

## Tóm tắt (Abstract)
Báo cáo này hoàn tất thử thách nghiên cứu về tính chọn lọc danh mục của nơ-ron hình chiếu MLP thông qua việc kiểm chứng tính bền vững thống kê. Chúng ta thực hiện hai bước kiểm định quan trọng: (1) Phân tích định tính sự phân hóa cấp độ từ (word-level differentiation) để loại trừ trường hợp nơ-ron chỉ phản ứng với một từ đơn lẻ, và (2) Kiểm chứng chéo (cross-validation) trên một tập dữ liệu độc lập có cấu trúc phức tạp. Kết quả cho thấy một sự tương quan mạnh mẽ giữa hai tập dữ liệu, xác nhận rằng các nơ-ron được xác định thực sự mã hóa các khái niệm phạm trù thay vì quá khớp với cấu trúc câu cụ thể.

---

## 1. Kiểm tra Tính Chuyên biệt hóa Danh mục (Exercise 4)

### 1.1. Phân tích Đơn vị tại Tầng 16
Để đảm bảo kết quả T-test ở Phần 1 không bị chi phối bởi duy nhất một từ đích (ví dụ: một nơ-ron chỉ "thích" từ "toothpaste"), chúng ta cô lập các nơ-ron có giá trị $|T|$ cực đại tại tầng 16 và trực quan hóa toàn bộ 40 điểm hoạt hóa.
- **Quan sát:** Các nơ-ron có T-value dương cực đại bộc lộ mức hoạt hóa cao đồng nhất cho cả 4 từ thuộc nhóm "Nha khoa" và thấp đồng nhất cho nhóm "Nội thất", và ngược lại cho các nơ-ron có T-value âm cực đại. Điều này khẳng định sự phân hóa diễn ra ở cấp độ **danh mục ngữ nghĩa**.

---

## 2. Kiểm chứng chéo trên Dữ liệu Mới (Exercise 5)

### 2.1. Tập dữ liệu Độc lập (Sentences Data 2)
Chúng ta đưa vào 20 câu văn mới với độ phức tạp cao hơn:
- **Cấu trúc hỗn hợp:** Một câu có thể chứa nhiều từ đích thuộc cả hai danh mục (ví dụ: "She placed her toothbrush in the dishwasher").
- **Thử thách lập trình:** Do tính chất đa target trên mỗi dòng, quy trình trích xuất phải sử dụng ma trận mặt nạ (mask matrix) để ánh xạ chính xác hoạt hóa của từng token đích vào đúng nhóm so sánh.

---

## 3. Phân tích Tương hợp: Biểu đồ "Pistachio Cannoli" (Exercise 6)

### 3.1. So sánh T-values xuyên tập dữ liệu
Nghiên cứu đối chiếu giá trị T thu được từ tập dữ liệu 1 ($T_1$) và tập dữ liệu 2 ($T_2$) cho tất cả các nơ-ron hình chiếu. 
- **Kết quả trực quan:** Biểu đồ scatter plot bộc lộ một đường chéo rõ rệt, đặc biệt là ở các nơ-ron có ý nghĩa thống kê cao (vùng màu xanh).
- **Phân loại nơ-ron:**
    - *Xanh lá (Green):* Có ý nghĩa thống kê ở cả hai tập dữ liệu.
    - *Đỏ (Red circles):* Chỉ có ý nghĩa ở một tập.
    - *Dấu gạch chéo (Red x's):* Không có ý nghĩa ở cả hai.

### 3.2. Định lượng Độ Tương hợp (Concordance)
Chỉ số tương hợp được tính toán dựa trên tỷ lệ các nơ-ron giữ nguyên hướng điều chỉnh (cùng dấu T-value) và duy trì ý nghĩa thống kê trên cả hai tập dữ liệu. Việc đạt được độ tương hợp cao chứng minh rằng các nơ-ron này là các thành phần "phổ quát" trong việc xử lý khái niệm của mô hình.

---

## 4. Thảo luận và Kết luận
Thử thách này làm nổi bật hai khía cạnh quan trọng của Diễn giải học thực nghiệm:
1. **Dữ liệu là vô tận:** Khác với y sinh, việc tạo thêm dữ liệu để kiểm chứng giả thuyết trong LLM là cực kỳ dễ dàng, cho phép chúng ta đạt được độ tin cậy thống kê rất cao.
2. **Từ khái niệm đến mã nguồn:** Những lý thuyết đơn giản về "tích hợp thông tin" thường đòi hỏi các kỹ thuật lập trình phức tạp (như indexing đa target) để biến thành bằng chứng định lượng.

Nghiên cứu kết luận rằng các nơ-ron hình chiếu MLP trong GPT-2 Large thực sự vận hành như các bộ lọc ngữ nghĩa bền vững, đóng góp vào khả năng phân loại và hiểu thế giới của mô hình.

---

## Tài liệu tham khảo (Citations)
1. Kiểm chứng chéo tính chọn lọc danh mục trên GPT-2 Large dựa trên `aero_LLM_16_CodeChallenge Category-tuned MLP projections (part 2).md`. Phân tích tương hợp và biểu đồ Pistachio Cannoli.
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
| 📌 **[Thử thách Lập trình: Hình chiếu MLP Điều chỉnh theo Danh mục (Phần 2)](aero_llm_16_codechallenge_category_tuned_mlp_projections_part_2_.md)** | [Xem bài viết →](aero_llm_16_codechallenge_category_tuned_mlp_projections_part_2_.md) |
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
