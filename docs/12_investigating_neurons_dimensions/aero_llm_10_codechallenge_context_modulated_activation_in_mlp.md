
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
# Thử thách Lập trình: Sự Điều chế Ngữ cảnh trong Hoạt hóa MLP (Context-modulated Activation)

## Tóm tắt (Abstract)
Báo cáo này khám phá một khía cạnh cơ bản trong hoạt động của LLM: sự khác biệt giữa việc xử lý một token đơn lẻ (in isolation) và token đó khi nằm trong một chuỗi văn bản (in context). Thông qua thực nghiệm trên lớp MLP của GPT-Neo, nghiên cứu so sánh hoạt hóa của nơ-ron trước hai biến số: (1) Token có và không có khoảng trắng phía trước, và (2) Token đơn lẻ so với token trong câu văn do mô hình tự tạo. Kết quả cho thấy trong khi khoảng trắng chỉ gây ra sự thay đổi nhẹ, ngữ cảnh tác động sâu sắc đến biểu diễn nội tại của nơ-ron, đặt ra thách thức lớn cho việc xác định các "đặc trưng thuần túy" trong Diễn giải học.

---

## 1. Mở Đầu (Introduction)
Chúng ta thường có xu hướng nghĩ về các từ như những thực thể độc lập với ý nghĩa cố định. Tuy nhiên, LLM không bao giờ được huấn luyện trên các từ rời rạc; chúng học từ những chuỗi văn bản khổng lồ nơi mỗi từ luôn bị bao quanh bởi ngữ cảnh. Thử thách này tìm cách định lượng mức độ "biến dạng" của hoạt hóa nơ-ron khi ta tách một từ ra khỏi môi trường tự nhiên của nó.

---

## 2. Thực Nghiệm 1: Tác động của Khoảng trắng (Preceding Spaces)

### 2.1. Bản chất của Tokenization
Hầu hết các tokenizer (như của GPT-2/Neo) coi " Apple" và "Apple" là hai token hoàn toàn khác nhau với các ID riêng biệt. 
- **Quy trình:** Lấy 100 danh từ phổ biến, đo hoạt hóa nơ-ron MLP ở Tầng 9 cho cả hai định dạng (có và không có dấu cách).

### 2.2. Quan sát Sơ bộ
Đồ thị phân tán cho thấy sự tương quan cực cao ($r \approx 0.99$). Mặc dù là hai thực thể toán học khác nhau, mô hình đã học được cách xử lý chúng gần như đồng nhất. Các nơ-ron chủ yếu nằm trên đường chéo chính, chỉ có một vài trường hợp ngoại lệ (outliers) bộc lộ sự nhạy cảm đặc biệt với ký tự trắng đầu tiên.

---

## 3. Thực Nghiệm 2: Sự Điều chế bởi Ngữ cảnh (Contextual Modulation)

### 3.1. Tạo văn bản và Trích xuất
Thay vì sử dụng các từ đơn lẻ, chúng ta cho mô hình tự sinh một đoạn văn bản (200 tokens) bắt đầu bằng câu lệnh: *"I think the world could be better if..."*. Sau đó, ta tiến hành so sánh:
- **Xử lý theo câu:** Đẩy toàn bộ 200 tokens qua mô hình trong một lượt (có ngữ cảnh).
- **Xử lý đơn lẻ:** Đẩy từng token trong số 200 tokens đó qua mô hình một cách độc lập (không ngữ cảnh).

### 3.2. Sự Đứt gãy của Tính Đồng nhất
Khác với thực nghiệm khoảng trắng, đồ thị phân tán ở đây bộc lộ sự phân tán cực lớn. Cùng một token, cùng một nơ-ron, nhưng hoạt hóa khi có ngữ cảnh khác xa so với khi đứng một mình.
- **Giải thích:** Lớp Attention ở các tầng trước đó đã "nhào nặn" vector nhúng dựa trên các từ xung quanh trước khi nó đi tới lớp MLP. Do đó, MLP không nhìn thấy "từ thuần túy" mà nhìn thấy một "khái niệm đã được điều chế".

---

## 4. Thách thức đối với Diễn giải học (Mechanistic Interpretability)
Kết quả này làm nảy sinh một vấn đề triết học và kỹ thuật trong nghiên cứu AI:
1. **Sự thiếu hụt biểu diễn gốc:** LLM không có khái niệm về một "từ đơn lẻ" thực thụ. Mọi hoạt hóa chúng ta trích xuất được luôn là sản phẩm của một ngữ cảnh nào đó (ngay cả khi ngữ cảnh đó chỉ là "không có gì").
2. **Vấn đề lặp lại:** Một nơ-ron được coi là nơ-ron "danh từ" trong văn bản này có thể không hành động như vậy trong văn bản khác do sự điều chế ngược từ các tầng Attention phía trên.

---

## 5. Kết Luận
Sự điều chế ngữ cảnh là "friction" (lực ma sát) trong vật lý của LLM – nó luôn hiện diện và không thể bị loại bỏ hoàn toàn trong các môi trường thực tế. Báo cáo khẳng định rằng mọi kết luận về tính chọn lọc của nơ-ron (như đã thấy ở bài về nơ-ron Danh từ/Động từ) cần được xem xét dưới lăng kính của sự biến thiên ngữ cảnh. Việc hiểu rõ mức độ biến thiên này là bước đi tiên quyết để xây dựng các phương pháp giải mã mô hình bền vững hơn.

---

## Tài liệu tham khảo (Citations)
1. Phân tích sự điều chế ngữ cảnh trên GPT-Neo dựa trên `aero_LLM_10_CodeChallenge Context-modulated activation in MLP.md`. So sánh hoạt hóa đơn lẻ (isolated) và hoạt hóa có ngữ cảnh (embedded).
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
| 📌 **[Thử thách Lập trình: Sự Điều chế Ngữ cảnh trong Hoạt hóa MLP (Context-modulated Activation)](aero_llm_10_codechallenge_context_modulated_activation_in_mlp.md)** | [Xem bài viết →](aero_llm_10_codechallenge_context_modulated_activation_in_mlp.md) |
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
