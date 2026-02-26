
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
# Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 3)

## Tóm tắt (Abstract)
Báo cáo này tổng kết thử thách nghiên cứu về cơ chế phủ định trong GPT-2 Large bằng cách phân tích sự biến đổi của nơ-ron xuyên suốt toàn bộ chiều sâu của mạng Transformer. Chúng ta thực hiện một phân tích hệ thống trên 36 tầng, định lượng mật độ và hiệu năng của các "nơ-ron phủ định". Kết quả thực nghiệm cho thấy một xu hướng suy giảm rõ rệt về cả số lượng nơ-ron chuyên biệt hóa lẫn độ chính xác phân loại khi đi về phía các tầng cuối. Khám phá này ủng hộ giả thuyết về sự chuyển dịch chức năng của mô hình: từ việc giải mã đặc tính của token hiện tại sang việc tích hợp ngữ cảnh để dự báo tương lai.

---

## 1. Phân tích chức năng mở rộng (Exercise 5)

### 1.1. Sự chọn lọc ngữ nghĩa (Semantic Selectivity)
Thông qua bản đồ nhiệt hoạt hóa, chúng ta quan sát thấy nơ-ron có Beta cao nhất không chỉ phản ứng với các từ phủ định thuần túy (*not, neither, won't*) mà còn kích hoạt mạnh với các từ mang sắc thái tiêu cực hoặc suy tàn (ví dụ: *rusty, dead, corroded*). 
- **Giải thích:** Điều này chứng tỏ nơ-ron đã học được một "tính năng" (feature) trừu tượng hơn là chỉ một danh sách từ vựng. Nó mã hóa một khái niệm logic/ngữ nghĩa rộng hơn về sự phủ định hoặc sự thiếu vắng.

---

## 2. Xu hướng xuyên Tầng (Exercise 6)

### 2.1. Mật độ Nơ-ron có Ý nghĩa Thống kê
Chúng ta chạy hồi quy trên toàn bộ 36 tầng và áp dụng hiệu chỉnh Bonferroni khắt khe ($p < 0.05 / 5120$).
- **Tầng đầu (Early Layers):** Lên đến 70% số nơ-ron trong tầng bộc lộ tính chọn lọc phủ định rõ rệt.
- **Tầng cuối (Late Layers):** Tỷ lệ này giảm xuống chỉ còn khoảng 25%. Mặc dù giảm mạnh, nhưng vẫn còn khoảng 1000 nơ-ron duy trì được tín hiệu, cho thấy cơ chế này vẫn được bảo tồn một phần ở giai đoạn cuối.

### 2.2. Hiệu năng Dự báo (Accuracy)
Độ chính xác trung bình của các nơ-ron được chọn lọc (Significant Positive Beta) cũng bộc lộ xu hướng tương tự:
- Đạt đỉnh khoảng 75-80% tại các tầng thấp.
- Giảm xuống mức 65% ở các tầng cao nhất (vượt trên mức ngẫu nhiên 50%).

---

## 3. Kiến giải về Kiến trúc Transformer

### 3.1. Chuyển dịch từ Hiện tại sang Tương lai
Sự suy giảm của các "nơ-ron phủ định" ở các tầng sâu có thể được giải thích bằng nhiệm vụ chính của mô hình: **Dự báo Token tiếp theo (Next Token Prediction)**.
- **Giai đoạn đầu:** Residual stream chứa thông tin đậm đặc về thuộc tính của chính token đó. Các nơ-ron MLP tập trung giải mã logic của token hiện tại.
- **Giai đoạn cuối:** Mô hình ưu tiên việc chuẩn bị Logits cho từ tiếp theo. Do đó, các biểu diễn về "hiện tại" (như phủ định) bị mờ nhạt dần để nhường chỗ cho các dự báo về "tương lai".

---

## 4. Kỹ thuật Lập trình: Ma trận Mặt nạ (Masked Arrays)
Để tính toán trung bình trên các tập hợp nơ-ron thỏa mãn đồng thời hai điều kiện ($\beta > 0$ và $p < \alpha$), chúng ta sử dụng `np.ma.masked_array`.
- **Lưu ý quan trọng:** Trong NumPy, mask có giá trị `True` nghĩa là điểm dữ liệu bị che khuất (không tính). Vì vậy, để lấy các nơ-ron ý nghĩa, chúng ta phải sử dụng toán tử nghịch đảo (`~`) trên mask điều kiện.

---

## 5. Kết Luận Chung
Thử thách này làm nổi bật tính phức tạp và thú vị của Mechanistic Interpretability. Chúng ta đã biến một câu hỏi ngôn ngữ học trừu tượng ("Mô hình xử lý sự phủ định như thế nào?") thành một bài toán định lượng với dữ liệu thực nghiệm. Việc hiểu rõ bản chất thống kê và xu hướng của nơ-ron theo tầng là bước chuẩn bị quan trọng để khám phá các thành phần khác như các đầu Attention (Attention Heads) trong các nghiên cứu tiếp theo.

---

## Tài liệu tham khảo (Citations)
1. Tổng kết Negation tuning trên GPT-2 Large dựa trên `aero_LLM_22_CodeChallenge Negation tuning in MLP neurons (part 3).md`. Phân tích xu hướng Accuracy theo tầng và giả thuyết chuyển dịch chức năng.
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
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 2)](aero_llm_21_codechallenge_negation_tuning_in_mlp_neurons_part_2_.md) | [Xem bài viết →](aero_llm_21_codechallenge_negation_tuning_in_mlp_neurons_part_2_.md) |
| 📌 **[Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 3)](aero_llm_22_codechallenge_negation_tuning_in_mlp_neurons_part_3_.md)** | [Xem bài viết →](aero_llm_22_codechallenge_negation_tuning_in_mlp_neurons_part_3_.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron QVK (Attention)](aero_llm_23_codechallenge_negation_tuning_in_qvk_neurons.md) | [Xem bài viết →](aero_llm_23_codechallenge_negation_tuning_in_qvk_neurons.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
