
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
# Làm rõ về Hidden States Tầng cuối: Vai trò của LayerNorm (Clarification of Final Hidden States)

## Tóm tắt (Abstract)
Báo cáo này giải quyết một sự khác biệt quan trọng trong việc trích xuất hoạt hóa giữa phương pháp Hooks và `output.hidden_states` tại tầng cuối cùng của mô hình Transformer (GPT-2). Trong khi ở các tầng trung gian, hai phương pháp này cho kết quả trùng khớp hoàn hảo, thì tại tầng cuối cùng, giá trị trích xuất từ Hidden States đã được đi qua một lớp chuẩn hóa bổ sung gọi là Final LayerNorm ($L_f$). Nghiên cứu thực nghiệm chứng minh sự khác biệt này và giải thích lý do tại sao các mẫu tính toán ở tầng cuối cùng thường mang các đặc tính định lượng khác biệt so với phần còn lại của mô hình.

---

## 1. Mở Đầu (Introduction)
Trong các bài báo trước, chúng ta đã giả định rằng `hidden_states[i]` tương đương với đầu ra của Transformer Block thứ `i-1`. Tuy nhiên, khi đi sâu vào phân tích cơ học, chúng ta phát hiện một ngoại lệ tại điểm kết thúc của residual stream. Việc hiểu rõ ranh giới giữa Khối Transformer cuối cùng và lớp chuẩn hóa cuối cùng là tối quan trọng để giải mã chính xác các biểu diễn trước khi chúng được chuyển đổi thành Logits.

---

## 2. Phẫu thuật Kiến trúc: Transformer Block và Final LayerNorm

### 2.1. Cấu trúc chuẩn của Hidden States
Đối với các tầng từ $0$ đến $N-2$ (với $N$ là tổng số tầng):
- **Hook Output:** Giá trị hoạt hóa ngay sau lớp MLP Projection.
- **Hidden State Output:** Trùng khớp $100\%$ với Hook Output.

### 2.2. Sự khác biệt tại tầng $N-1$
Tại khối Transformer cuối cùng:
- **Hook Output:** Là kết quả của MLP cuối cùng cộng vào residual stream.
- **Hidden State Output:** Là Hook Output đã được đẩy qua `model.transformer.ln_f`.

---

## 3. Thực Nghiệm Đối Chứng (Experimental Verification)

### 3.1. Thử nghiệm "Penultimate vs. Final"
Bằng cách sử dụng phép trừ ma trận giữa dữ liệu Hook và dữ liệu Hidden States:
1. **Tại tầng áp chót (Penultimate):** Hiệu số bằng 0 tuyệt đối. Điều này xác nhận sự đồng nhất của hai phương pháp trích xuất ở các tầng giữa.
2. **Tại tầng cuối (Final):** Hiệu số khác 0 đáng kể. Điều này chỉ ra rằng có một phép biến đổi toán học đã xảy ra giữa điểm trích xuất của Hook và điểm trích xuất của Hidden States.

### 3.2. Chứng minh bằng Final LayerNorm
Khi lấy kết quả từ Hook tại tầng cuối và chủ động đẩy nó qua lớp `model.transformer.ln_f`, hiệu số so với `hidden_states[-1]` trở về bằng 0. Đây là bằng chứng thực nghiệm khẳng định rằng Hidden State cuối cùng thực chất là một trạng thái "đã chuẩn hóa" để chuẩn bị cho bước nhân với ma trận nhúng đầu ra (un-embedding).

---

## 4. Thảo Luận: Tại sao tầng cuối cùng lại "đặc biệt"?
Nhà nghiên cứu cần lưu ý hai lý do khiến dữ liệu tầng cuối thường trông khác biệt trên đồ thị:
1. **Áp lực tính toán:** Đây là cơ hội cuối cùng để mô hình tinh chỉnh vector dự báo, do đó các nơ-ron có xu hướng hoạt động với cường độ và tính chọn lọc cao hơn.
2. **Biến đổi toán học:** Sự hiện diện của Final LayerNorm làm nén các giá trị hoạt hóa về một vùng phân phối ổn định hơn, che lấp đi các biến động biên độ cực lớn thường thấy trong dòng dư chưa chuẩn hóa.

---

## 5. Kết Luận
Báo cáo khẳng định rằng khi thực hiện các phân tích so sánh xuyên tầng (laminar profile), cần phải đồng nhất phương pháp trích xuất. Nếu sử dụng `output.hidden_states`, hãy nhớ rằng tầng cuối cùng đã được chuẩn hóa. Nếu muốn quan sát "tư duy thô" (raw thinking) của mô hình ở tầng cuối, sử dụng Hooks là lựa chọn tối ưu hơn.

---

## Tài liệu tham khảo (Citations)
1. Phân tích sự khác biệt cơ học tại tầng cuối cùng của GPT-2 dựa trên `aero_LLM_07_Clarification of final hidden_states output.md`. Xác minh vai trò của `ln_f` đối với Hidden States.
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
| 📌 **[Làm rõ về Hidden States Tầng cuối: Vai trò của LayerNorm (Clarification of Final Hidden States)](aero_llm_07_clarification_of_final_hidden_states_output.md)** | [Xem bài viết →](aero_llm_07_clarification_of_final_hidden_states_output.md) |
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
