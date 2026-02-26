
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
# Thử thách Lập trình: Tính Chọn lọc Ngữ pháp của Nơ-ron MLP (Phần 1)

## Tóm tắt (Abstract)
Báo cáo này trình bày giai đoạn đầu của thử thách tìm kiếm các nơ-ron chuyên biệt cho các từ loại (parts of speech) trong phân đoạn mở rộng của lớp MLP trên mô hình GPT-Neo. Nghiên cứu tập trung vào việc so sánh hoạt hóa của nơ-ron trước hai danh mục từ vựng: Danh từ (Nouns) và Động từ (Verbs). Quy trình thực nghiệm bao gồm việc cấy Hooks vào các nơ-ron mở rộng (expansion neurons) – nơi được giả thuyết là trích xuất các đặc trưng phi tuyến từ residual stream – và thu thập dữ liệu phản hồi từ 200 từ vựng phổ biến. Kết quả sơ bộ cho thấy sự tồn tại của các thiên kiến (biases) và sự biến thiên nơ-ron rõ rệt, đặt nền tảng cho phân tích thống kê chuyên sâu ở phần tiếp theo.

---

## 1. Mở Đầu (Introduction)
Một câu hỏi trung tâm trong Diễn giải học là: Các LLM có sở hữu những "nơ-ron ngôn ngữ" chuyên biệt không? Ví dụ, có tồn tại một nơ-ron chỉ kích hoạt mạnh khi nhìn thấy danh từ mà im lặng trước động từ? Báo cáo này thiết lập môi trường thực nghiệm để kiểm chứng giả thuyết đó, tập trung vào lớp MLP (Multi-Layer Perceptron) – thành phần được coi là "kho tri thức" và "bộ trích xuất đặc trưng" của kiến trúc Transformer.

---

## 2. Cơ sở Lý thuyết: Tại sao lại là MLP?
Trong một khối Transformer:
- **Lớp Attention:** Đóng vai trò tích hợp ngữ cảnh từ các token xung quanh (ai đang làm gì cho ai).
- **Lớp MLP:** Đóng vai trò nhận diện các thuộc tính nội tại của token (đây là một vật thể hay một hành động).
Đặc biệt, lớp mở rộng (C_FC) tăng số chiều lên gấp 4 lần (từ 768 lên 3072 trong GPT-2 Small), tạo ra một không gian rộng lớn để mô hình phân tách các khái niệm ngữ nghĩa và ngữ pháp.

---

## 3. Thiết lập Thực nghiệm (Methodology)

### 3.1. Chuẩn bị Dữ liệu và Mô hình
- **Mô hình:** GPT-Neo 125M (sử dụng tokenizer EleutherAI).
- **Dữ liệu:** Danh sách 100 động từ và 100 danh từ thông dụng nhất được trích xuất từ các nguồn công khai.
- **Trạng thái:** Mô hình được thiết lập ở chế độ `eval()` để đảm bảo tính ổn định của các hoạt hóa.

### 3.2. Cấy Hook vào Lớp Mở rộng (Expansion Layer)
Sử dụng `register_forward_hook` vào thành phần `c_fc` của Transformer Block thứ 9 (index 8). Điểm thu thập dữ liệu nằm ngay sau khi thực hiện phép nhân ma trận trọng số nhưng trước khi đi qua hàm kích hoạt phi tuyến (GELU). Điều này cho phép ta quan sát "tư duy thô" của nơ-ron trước khi bị nén bởi cơ chế thưa thớt (sparsity).

### 3.3. Thu thập Hoạt hóa Diện rộng
Dữ liệu được lưu trữ trong một mảng 3 chiều có kích thước `[2, 100, 3072]`:
- `2`: Danh mục (0: Động từ, 1: Danh từ).
- `100`: Số lượng từ trong mỗi danh mục.
- `3072`: Số lượng nơ-ron MLP.
*Kỹ thuật quan trọng:* Sử dụng `mean(dim=1)` để xử lý các từ bị tách thành nhiều tokens, đảm bảo mỗi từ (word) chỉ đại diện bởi một vector hoạt hóa duy nhất.

---

## 4. Kết Quả Sơ Bộ và Quan Sát
Đồ thị phân bố hoạt hóa cho thấy:
1. **Sự Thiên lệch Hệ thống (Systematic Offsets):** Các nơ-ron không hoạt động quanh mức 0 mà thường có một điểm dừng (mean offset) cố định cho hầu hết các từ (thường là giá trị âm).
2. **Các Băng dọc (Vertical Bands):** Một số nơ-ron cho thấy biên độ hoạt hóa khác biệt rõ rệt so với số đông trên toàn bộ dải từ vựng thử nghiệm.
3. **Tính Biến thiên:** Mặc dù nhìn tổng thể có vẻ đồng nhất, nhưng các nơ-ron riêng lẻ bắt đầu bộc lộ sự ưu tiên nhẹ đối với danh từ hoặc động từ khi nhìn chi tiết vào các điểm dữ liệu.

---

## 5. Kết Luận Phần 1
Chúng ta đã thành công trong việc xây dựng hệ thống trích xuất hoạt hóa quy mô lớn từ nơ-ron MLP. Việc quan sát thấy các dải hoạt hóa ổn định là dấu hiệu tích cực cho thấy các nơ-ron này đang "mã hóa" những thuộc tính nhất định của ngôn ngữ. Phần tiếp theo sẽ triển khai các phép kiểm định thống kê (t-test) để xác định xem sự khác biệt giữa danh từ và động từ có đạt mức ý nghĩa khoa học hay không.

---

## Tài liệu tham khảo (Citations)
1. Thử thách về Grammar tuning trên GPT-Neo dựa trên `aero_LLM_08_CodeChallenge Grammar tuning in MLP neurons (part 1).md`. Thiết lập Hooks và quy trình thu thập dữ liệu nơ-ron mở rộng.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 12-Investigating-neurons-dimensions](README.md) | [Xem bài viết →](README.md) |
| [Cực đại hóa Hoạt hóa (Activation Maximization): Cơ sở Lý thuyết và Những thách thức trong LLM](aero_llm_01_activation_maximization_via_gradient_ascent_theory_.md) | [Xem bài viết →](aero_llm_01_activation_maximization_via_gradient_ascent_theory_.md) |
| [Triển khai Cực đại hóa Hoạt hóa: Từ Gradient Ascent đến Giải mã Token (Activation Maximization Implementation)](aero_llm_02_activation_maximization_code_.md) | [Xem bài viết →](aero_llm_02_activation_maximization_code_.md) |
| [Cực đại hóa Hoạt hóa qua Lấy mẫu Dữ liệu (Activation Maximization via Data Sampling)](aero_llm_03_activation_maximization_via_data_sampling.md) | [Xem bài viết →](aero_llm_03_activation_maximization_via_data_sampling.md) |
| [Thử thách Lập trình: Kiểm chứng Tính lặp lại của Cực đại hóa Hoạt hóa (Reproducibility of Activation Maximization)](aero_llm_04_codechallenge_reproducibility_of_activation_maximization.md) | [Xem bài viết →](aero_llm_04_codechallenge_reproducibility_of_activation_maximization.md) |
| [Giải phẫu Nội tại Mô hình bằng Hooks: Kỹ thuật Trích xuất Hoạt hóa (Extracting Activations via Hooks)](aero_llm_05_extracting_activations_using_hooks.md) | [Xem bài viết →](aero_llm_05_extracting_activations_using_hooks.md) |
| [Mối tương quan giữa Hooks và Hidden States: Giải cấu trúc Khối Transformer (Reconstructing Transformer Blocks)](aero_llm_06_relation_between_hooks_and_output_hidden_states.md) | [Xem bài viết →](aero_llm_06_relation_between_hooks_and_output_hidden_states.md) |
| [Làm rõ về Hidden States Tầng cuối: Vai trò của LayerNorm (Clarification of Final Hidden States)](aero_llm_07_clarification_of_final_hidden_states_output.md) | [Xem bài viết →](aero_llm_07_clarification_of_final_hidden_states_output.md) |
| 📌 **[Thử thách Lập trình: Tính Chọn lọc Ngữ pháp của Nơ-ron MLP (Phần 1)](aero_llm_08_codechallenge_grammar_tuning_in_mlp_neurons_part_1_.md)** | [Xem bài viết →](aero_llm_08_codechallenge_grammar_tuning_in_mlp_neurons_part_1_.md) |
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
