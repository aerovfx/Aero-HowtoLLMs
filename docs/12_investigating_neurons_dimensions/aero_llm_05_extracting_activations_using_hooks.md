
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
# Giải phẫu Nội tại Mô hình bằng Hooks: Kỹ thuật Trích xuất Hoạt hóa (Extracting Activations via Hooks)

## Tóm tắt (Abstract)
Báo cáo này hướng dẫn phương pháp sử dụng "Hooks" – các hàm can thiệp đặc biệt trong PyTorch – để truy cập và trích xuất dữ liệu từ các lớp ẩn bên trong Transformer. Trong khi các phương thức thông thường chỉ cho phép quan sát Logits đầu ra hoặc Hidden States của toàn bộ khối Transformer, kỹ thuật Hook cho phép nhà nghiên cứu cô lập các thành phần vi mô như ma trận Query $Q$, Key $K$, Value $V$ hoặc các lớp MLP. Báo cáo cũng thảo luận về cơ chế quản lý Hook (đăng ký và gỡ bỏ) và cách quản lý bộ nhớ thông qua việc ghi đè hoặc tích lũy dữ liệu.

---

## 1. Mở Đầu (Introduction)
Để thực hiện Diễn giải học cơ học (Mechanistic Interpretability), việc biết trọng số (weights) của mô hình là chưa đủ. Chúng ta cần biết cách các nơ-ron thực sự phản ứng (activations) khi dữ liệu cụ thể đi qua. Hooks đóng vai trò như các "cảm biến" được cấy vào dòng chảy dữ liệu của mô hình trong quá trình forward-pass, cho phép ta chụp lại trạng thái của bất kỳ nơ-ron nào mà không cần sửa đổi cấu trúc cốt lõi của mạng.

---

## 2. Cơ chế Hoạt động của PyTorch Hooks

### 2.1. Định nghĩa Hàm Hook
Một hàm Hook tiêu chuẩn nhận ba tham số đầu vào:
1. **Module:** Lớp (layer) mà hook được gắn vào.
2. **Input:** Dữ liệu đi vào lớp đó.
3. **Output:** Kết quả tính toán đi ra khỏi lớp đó.
Bên trong hàm này, ta có thể trích xuất `output`, thực hiện các phép toán (như tách các chiều Q, K, V) và lưu trữ kết quả vào một biến bên ngoài (thường là Dictionary hoặc List).

### 2.2. Đăng ký và Quản lý (Registration & Handles)
Sử dụng phương thức `register_forward_hook` để cấy hàm vào mô hình. Kết quả trả về là một `handle`, có thể được sử dụng để gỡ bỏ (`remove()`) hook khi không còn cần thiết, giúp tối ưu hóa hiệu năng và tránh rò rỉ bộ nhớ.

---

## 3. Quản lý Dữ liệu Hoạt hóa (Data Management)

### 3.1. Ghi đè (Overwriting via Dictionary)
Nếu lưu trữ dữ liệu vào một `Dictionary` với key là tên tầng, mỗi lượt forward-pass mới sẽ ghi đè lên dữ liệu cũ. Đây là cách tiếp cận phổ biến khi ta chỉ quan tâm đến phản hồi của mô hình đối với câu lệnh hiện tại. 
*Lưu ý:* Nếu câu lệnh mới có các token đầu tiên giống câu lệnh cũ, các hàng tương ứng trong ma trận hoạt hóa sẽ giống nhau do tính chất truyền tin theo trình tự.

### 3.2. Tích lũy (Accumulation via List)
Bằng cách sử dụng `List` và phương thức `append()`, ta có thể lưu trữ lịch sử hoạt hóa của tất cả các câu lệnh đã đi qua mô hình. Điều này hữu ích cho các phân tích thống kê diện rộng hoặc so sánh sự biến thiên của nơ-ron qua nhiều ngữ cảnh khác nhau.

---

## 4. Phân tích Dữ liệu trích xuất
Khi đã có dữ liệu qua Hook, ta có thể thực hiện các phân tích trực quan:
- **Scatter Plots:** So sánh hoạt hóa của hai token khác nhau trên toàn bộ các nơ-ron của một tầng.
- **Correlation Matrices:** Đo lường sự tương quan giữa các token. Quan sát thực nghiệm cho thấy token đầu tiên thường có độ tương quan thấp với phần còn lại do thiếu hụt ngữ cảnh tiền đề.

---

## 5. Kết Luận
Hooks là công cụ mạnh mẽ nhất để biến một mô hình "hộp đen" thành một hệ thống có thể quan sát được ở mọi cấp độ hạt. Việc làm chủ kỹ thuật này không chỉ giúp trích xuất dữ liệu mà còn đặt nền móng cho việc chỉnh sửa hoạt hóa (activation editing) – một kỹ thuật can thiệp nhân quả sâu sắc hơn sẽ được thảo luận ở các chương sau.

---

## Tài liệu tham khảo (Citations)
1. Kỹ thuật trích xuất hoạt hóa bằng Hooks trên GPT-2 dựa trên `aero_LLM_05_Extracting activations using hooks.md`. Phân tích sự khác biệt giữa cơ chế Overwriting và Concatenation.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 12_investigating_neurons_dimensions](README.md) | [Xem bài viết →](README.md) |
| [Cực đại hóa Hoạt hóa (Activation Maximization): Cơ sở Lý thuyết và Những thách thức trong LLM](aero_llm_01_activation_maximization_via_gradient_ascent_theory_.md) | [Xem bài viết →](aero_llm_01_activation_maximization_via_gradient_ascent_theory_.md) |
| [Triển khai Cực đại hóa Hoạt hóa: Từ Gradient Ascent đến Giải mã Token (Activation Maximization Implementation)](aero_llm_02_activation_maximization_code_.md) | [Xem bài viết →](aero_llm_02_activation_maximization_code_.md) |
| [Cực đại hóa Hoạt hóa qua Lấy mẫu Dữ liệu (Activation Maximization via Data Sampling)](aero_llm_03_activation_maximization_via_data_sampling.md) | [Xem bài viết →](aero_llm_03_activation_maximization_via_data_sampling.md) |
| [Thử thách Lập trình: Kiểm chứng Tính lặp lại của Cực đại hóa Hoạt hóa (Reproducibility of Activation Maximization)](aero_llm_04_codechallenge_reproducibility_of_activation_maximization.md) | [Xem bài viết →](aero_llm_04_codechallenge_reproducibility_of_activation_maximization.md) |
| 📌 **[Giải phẫu Nội tại Mô hình bằng Hooks: Kỹ thuật Trích xuất Hoạt hóa (Extracting Activations via Hooks)](aero_llm_05_extracting_activations_using_hooks.md)** | [Xem bài viết →](aero_llm_05_extracting_activations_using_hooks.md) |
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
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 2)](aero_llm_21_codechallenge_negation_tuning_in_mlp_neurons_part_2_.md) | [Xem bài viết →](aero_llm_21_codechallenge_negation_tuning_in_mlp_neurons_part_2_.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 3)](aero_llm_22_codechallenge_negation_tuning_in_mlp_neurons_part_3_.md) | [Xem bài viết →](aero_llm_22_codechallenge_negation_tuning_in_mlp_neurons_part_3_.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron QVK (Attention)](aero_llm_23_codechallenge_negation_tuning_in_qvk_neurons.md) | [Xem bài viết →](aero_llm_23_codechallenge_negation_tuning_in_qvk_neurons.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
