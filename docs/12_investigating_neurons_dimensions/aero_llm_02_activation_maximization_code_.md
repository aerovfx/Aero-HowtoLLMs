
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
# Triển khai Cực đại hóa Hoạt hóa: Từ Gradient Ascent đến Giải mã Token (Activation Maximization Implementation)

## Tóm tắt (Abstract)
Báo cáo này hướng dẫn chi tiết quy trình thực nghiệm để triển khai kỹ thuật Cực đại hóa Hoạt hóa trên mô hình GPT-2 Small bằng PyTorch. Thí nghiệm tập trung vào việc tối ưu hóa một ma trận nhúng ngẫu nhiên (random embeddings) để kích thích tối đa một chiều hoạt hóa cụ thể trong residual stream. Mặc dù quá trình tối ưu hóa toán học đạt được thành công rực rỡ (tăng cường độ hoạt hóa lên 3 bậc độ lớn), kết quả giải mã (decoding) sang văn bản cho thấy các chuỗi token thu được thiếu tính liên kết ngữ nghĩa đối với con người. Kết quả này củng cố giả thuyết về "tính phân tán" và "không gian biểu diễn phi ngôn ngữ" của các nơ-ron bên trong LLM.

---

## 1. Mở Đầu (Introduction)
Trong thực hành, Cực đại hóa Hoạt hóa biến quá trình suy diễn của mô hình thành một bài toán tối ưu hóa ngược. Thay vì truyền văn bản qua tokenizer, chúng ta tác động trực tiếp vào không gian nhúng (embedding space). Mục tiêu là tìm ra "chuỗi token lý tưởng" – dù có thể không tồn tại trong từ điển thực tế – mà mô hình coi là tín hiệu mạnh nhất cho một thành phần nội tại.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Khởi tạo Ma trận Nhúng giả lập
Chúng ta tạo một ma trận nhúng ngẫu nhiên cho một chuỗi gồm 5 tokens. Để đảm bảo tính tương đồng về mặt toán học với mô hình gốc, ma trận này được chuẩn hóa để có cùng độ lệch chuẩn (Standard Deviation) với ma trận nhúng đã huấn luyện của GPT-2.
- **Tham số tối ưu:** `requires_grad = True` được thiết lập cho ma trận nhúng để cho phép PyTorch tính toán gradient.

### 2.2. Cơ chế Pushing Embeddings trực tiếp
Một kỹ thuật quan trọng được sử dụng là tham số `inputs_embeds` trong hàm forward của Hugging Face. Điều này cho phép bỏ qua lớp Tokenizer và Position Embeddings, đẩy trực tiếp giá trị vector vào các Transformer Blocks.

### 2.3. Thiết lập Hàm Loss và Gradient Ascent
- **Mục tiêu:** Cực đại hóa hoạt hóa $a$ tại tầng 4, chiều 90.
- **Hàm tổn thất (Loss):** $L = -a + \lambda \|\theta\|_2^2$. Việc lấy dấu trừ biến bài toán thành cực tiểu hóa, phù hợp với hầu hết các bộ tối ưu (optimizers). Thành phần L2 được thêm vào để ngăn chặn hiện tượng bùng nổ trọng số.
- **Bộ tối ưu:** Adam Optimizer với tốc độ học (learning rate) 0.001 qua 500 bước lặp.

---

## 3. Kết Quả Thực Nghiệm (Results & Analysis)

### 3.1. Hiệu quả của Tối ưu hóa
Đồ thị giám sát cho thấy cường độ hoạt hóa của chiều mục tiêu tăng vọt từ mức gần 0 lên các giá trị dương rất lớn. Đồng thời, các chiều lân cận (neighboring dimensions) bị ức chế, chứng tỏ quá trình tối ưu hóa đã cô lập thành công tính chất đặc trưng của nơ-ron đích.

### 3.2. Nghịch lý Giải mã (The Decoding Paradox)
Bước cuối cùng là chuyển vector nhúng đã tối ưu về token thực thông qua độ tương quan Cosine (Cosine Similarity) với toàn bộ 50.257 tokens trong vocab. 
- **Kết quả văn bản:** "ad pc brisk brisk breast" hoặc các chuỗi vô nghĩa tương tự.
- **Phân tích:** Độ tương quan Cosine cao nhất thường chỉ dừng lại ở mức 0.17. Điều này chỉ ra rằng "vector lý tưởng" mà nơ-ron tìm kiếm nằm ở một vùng không gian không có word nhúng nào thực sự đại diện cho nó.

---

## 4. Thảo Luận: Tại sao phương pháp này "thất bại" trong việc diễn giải?
Dù toán học vận hành chính xác, Activation Maximization trong LLM thường không mang lại tri thức con người có thể hiểu ngay lập tức (human-interpretable). Điều này phản ánh:
- **Sự khác biệt với Vision Models:** Trong hình ảnh, các pixel có tính liên tục. Trong ngôn ngữ, các điểm nhúng nằm rải rác và không có "vùng chuyển tiếp" giữa các khái niệm.
- **Tính đa ngữ (Polysemanticity):** Nơ-ron mục tiêu có thể đang phản ứng với một mô thức cấu trúc phức tạp (như "từ 2 âm tiết bắt đầu bằng phụ âm") hơn là một khái niệm ngữ nghĩa đơn giản.

---

## 5. Kết Luận
Việc thực hiện Activation Maximization không chỉ là bài tập lập trình về Hooks và Gradients, mà còn là một quy trình pháp chứng (forensic process) để hiểu về giới hạn của mô hình. Thất bại trong việc tạo ra văn bản có nghĩa của phương pháp này chính là bằng chứng quan trọng nhất về tính phức tạp của không gian biểu diễn trong LLM, đặt nền móng cho việc sử dụng các kỹ thuật cao cấp hơn như Sparse Autoencoders.

---

## Tài liệu tham khảo (Citations)
1. Quy trình triển khai Code cho Activation Maximization dựa trên `aero_LLM_02_Activation maximization (code).md`. Phân tích việc sử dụng `inputs_embeds` và nghịch lý trong Decoding.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 12_investigating_neurons_dimensions](README.md) | [Xem bài viết →](README.md) |
| [Cực đại hóa Hoạt hóa (Activation Maximization): Cơ sở Lý thuyết và Những thách thức trong LLM](aero_llm_01_activation_maximization_via_gradient_ascent_theory_.md) | [Xem bài viết →](aero_llm_01_activation_maximization_via_gradient_ascent_theory_.md) |
| 📌 **[Triển khai Cực đại hóa Hoạt hóa: Từ Gradient Ascent đến Giải mã Token (Activation Maximization Implementation)](aero_llm_02_activation_maximization_code_.md)** | [Xem bài viết →](aero_llm_02_activation_maximization_code_.md) |
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
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 2)](aero_llm_21_codechallenge_negation_tuning_in_mlp_neurons_part_2_.md) | [Xem bài viết →](aero_llm_21_codechallenge_negation_tuning_in_mlp_neurons_part_2_.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 3)](aero_llm_22_codechallenge_negation_tuning_in_mlp_neurons_part_3_.md) | [Xem bài viết →](aero_llm_22_codechallenge_negation_tuning_in_mlp_neurons_part_3_.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron QVK (Attention)](aero_llm_23_codechallenge_negation_tuning_in_qvk_neurons.md) | [Xem bài viết →](aero_llm_23_codechallenge_negation_tuning_in_qvk_neurons.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
