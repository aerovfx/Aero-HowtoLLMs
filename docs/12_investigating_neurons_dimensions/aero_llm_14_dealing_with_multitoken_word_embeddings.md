
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
# Xử lý Biểu diễn Nơ-ron cho các Từ đa Token (Multi-token Words)

## Tóm tắt (Abstract)
Báo cáo này giải quyết một thách thức thực tiễn trong Diễn giải học (Mechanistic Interpretability): Cách trích xuất hoạt hóa cho các từ bị chia thành nhiều tokens bởi tokenizer (ví dụ: "toothpaste" $\rightarrow$ ["tooth", "paste"]). Qua phân tích lý thuyết và thực nghiệm trên GPT-2, nghiên cứu khẳng định rằng việc tập trung vào **token cuối cùng** là phương pháp tối ưu. Lý do cốt lõi nằm ở bản chất nhân quả (causal) của mô hình: tại token cuối, biểu diễn đã tích hợp toàn bộ thông tin từ các token thành phần phía trước, tạo thành một khái niệm ngữ nghĩa hoàn chỉnh. Báo cáo cũng cung cấp khung mã nguồn Python để xác định vị trí và phân tích sự biến thiên của các multi-token embeddings xuyên suốt các tầng.

---

## 1. Mờ Đầu (Introduction)
Trong tiếng Anh và nhiều ngôn ngữ khác, không phải từ nào cũng tương ứng với một token duy nhất. Các từ ghép ("toothpaste"), từ phức hoặc từ hiếm thường bị bẻ gãy. Khi nghiên cứu tính chọn lọc của nơ-ron đối với một "từ", câu hỏi đặt ra là chúng ta nên lấy dữ liệu từ token nào? Báo cáo này thiết lập một quy trình chuẩn hóa để xử lý các "đơn vị ngữ nghĩa đa thành phần" này.

---

## 2. Giả thuyết "Token cuối cùng là chìa khóa"

### 2.1. Cơ chế Tích hợp Ngữ cảnh
Xét từ "toothpaste":
1. Khi mô hình xử lý token "tooth", nó chưa biết từ tiếp theo là gì (có thể là "ache", "brush", hoặc "paste"). Biểu diễn tại đây chỉ mang tính dự đoán (predictive).
2. Khi mô hình xử lý token "paste" (đặc biệt là khi không có khoảng trắng phía trước và đi sau "tooth"), lớp Attention sẽ điều chế vector này dựa trên thông tin "tooth" đã có trong residual stream.
3. **Kết luận:** Tại vị trí "paste", mô hình mới thực sự sở hữu biểu diễn của khái niệm "toothpaste" hoàn chỉnh. "Tooth" không biết gì về "paste", nhưng "paste" biết rất nhiều về "tooth".

---

## 3. Quy trình Thực nghiệm và Triển khai Mã nguồn

### 3.1. Xác định vị trí Thuật toán (Algorithmic Indexing)
Trong một batch văn bản phức tạp, việc tìm vị trí của một cụm token đích (target sequence) đòi hỏi một quy trình kiểm duyệt nghiêm ngặt:
- Duyệt qua từng câu trong batch.
- Kiểm tra sự trùng khớp của token hiện tại và $k$ tokens phía trước với chuỗi đích.
- Lưu trữ index của token cuối cùng để phục vụ trích xuất `hidden_states`.

### 3.2. Quản lý Batch và Padding
Để xử lý các câu có độ dài khác nhau, nghiên cứu sử dụng kỹ thuật padding và `attention_mask`. Việc unpack dictionary thông qua toán tử `**` trong PyTorch giúp đẩy dữ liệu qua mô hình một cách hiệu quả, đảm bảo các token padding không làm nhiễu kết quả phân tích.

---

## 4. Phân tích Sự biến thiên Vector (Vector Displacement)
Nghiên cứu giới thiệu một phép đo thực nghiệm: Độ dài quỹ đạo của vector nhúng khi đi qua mô hình.
- **Công thức:** $\\mid $v_l$ - v_{l-1}\\mid$, trong đó $v_l$ là biểu diễn tại tầng $l$.
- **Quan sát:** Sự thay đổi này phản ánh khối lượng công việc tính toán mà các lớp Attention và MLP đã thực hiện để tinh chỉnh ý nghĩa của token. Đối với các từ đa token, token cuối cùng thường bộc lộ sự biến thiên lớn ở các tầng giữa, nơi "phép cộng ngữ nghĩa" thực sự diễn ra.

---

## 5. Kết Luận
Việc hiểu rõ cách tokenizer phân rã ngôn ngữ là điều kiện tiên quyết cho mọi nghiên cứu nội soi mô hình. Báo cáo xác lập quy tắc: Để phân tích một khái niệm, hãy luôn nhìn vào token kết thúc chuỗi biểu diễn khái niệm đó. Phương pháp này không chỉ đảm bảo tính chính xác về mặt ngữ nghĩa mà còn nhất quán với cơ cấu vận hành của kiến trúc Transformer.

---

## Tài liệu tham khảo (Citations)
1. Kỹ thuật xử lý multi-token word embeddings trên GPT-2 dựa trên `aero_LLM_14_Dealing with multitoken word embeddings.md`. Lý thuyết về tích hợp thông tin tại token cuối và quy trình trích xuất vector.
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
| 📌 **[Xử lý Biểu diễn Nơ-ron cho các Từ đa Token (Multi-token Words)](aero_llm_14_dealing_with_multitoken_word_embeddings.md)** | [Xem bài viết →](aero_llm_14_dealing_with_multitoken_word_embeddings.md) |
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
