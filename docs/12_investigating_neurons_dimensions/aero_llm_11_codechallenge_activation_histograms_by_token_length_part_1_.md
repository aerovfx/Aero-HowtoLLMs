
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
# Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 1)

## Tóm tắt (Abstract)
Báo cáo này bắt đầu một thử thách nghiên cứu đa giai đoạn nhằm định lượng mối quan hệ giữa độ dài của token (tính theo số ký tự) và cường độ hoạt hóa của các nơ-ron MLP trên toàn bộ các tầng của mô hình GPT-Neo. Trong phần này, chúng ta tập trung vào việc thiết lập hệ thống Hooks đa tầng, chuẩn bị dữ liệu từ bộ dữ liệu FineWeb và thực hiện phân tích thống kê về phân phối độ dài token. Kết quả thiết lập cho thấy khả năng trích xuất đồng thời hoạt hóa từ 12 khối Transformer với quy mô 3072 nơ-ron mỗi khối, tạo điều kiện cho các phân tích so sánh liên tầng ở các giai đoạn sau.

---

## 1. Mở Đầu (Introduction)
Các token trong LLM không có độ dài đồng nhất: một số chỉ là một ký tự đơn giản, trong khi số khác đại diện cho các từ phức tạp dài nhiều ký tự. Câu hỏi đặt ra là: Liệu mô hình có dành nhiều "năng lượng tính toán" (hoạt hóa nơ-ron) hơn cho các token dài - vốn thường mang nhiều thông tin ngữ nghĩa hơn - hay không? Báo cáo này xây dựng khung thực nghiệm để kiểm chứng giả thuyết này thông qua phân tích dải tần hoạt hóa (histograms).

---

## 2. Thiết lập Hệ thống trích xuất Đa tầng

### 2.1. Hooks Đa mục tiêu
Khác với các thực nghiệm trước chỉ tập trung vào một tầng đơn lẻ, nghiên cứu này yêu cầu quan sát hành vi của mô hình theo chiều sâu. Một vòng lặp `for` được sử dụng để cấy 12 Hooks vào thành phần `c_fc` (MLP expansion) của tất cả các khối Transformer. Mỗi Hook lưu trữ dữ liệu vào một `Dictionary` với key định danh duy nhất (ví dụ: `MLP_0`, `MLP_1`,...), cho phép chụp lại trạng thái toàn cục của mô hình trong một lượt forward-pass duy nhất.

### 2.2. Hiệu năng tính toán (CPU vs. GPU)
Dù việc vận hành mô hình GPT-Neo 125M trên CPU chỉ mất khoảng 1 phút cho 8192 tokens, báo cáo khuyến nghị sử dụng GPU để giảm thời gian xuống mức vài giây. Điều này đặc biệt quan trọng khi mở rộng quy mô sang các mô hình lớn hơn (như GPT-Neo 1.3B) ở các giai đoạn sau của thử thách.

---

## 3. Phân tích Dữ liệu Đầu vào (FineWeb Dataset)

### 3.1. Thu thập và Tokenization
Dữ liệu được lấy từ FineWeb cho đến khi đạt chính xác 8192 tokens. Con số này được chọn để khớp hoàn hảo với cấu trúc batch $16 \times 512$, tối ưu hóa việc sử dụng bộ nhớ và tính toán trên tensor.

### 3.2. Thống kê Độ dài Token
Một phát hiện thú vị từ phân tích thống kê:
- **Phạm vi:** Tokens có độ dài từ 1 đến 16 ký tự.
- **Trung vị (Median):** Độ dài trung vị quan sát được là 4 ký tự.
- **Phân nhóm:** Dựa trên trung vị, dữ liệu được chia thành 3 nhóm: "Ngắn hơn trung vị", "Bằng trung vị" và "Dài hơn trung vị". Do token là các giá trị nguyên, một lượng lớn dữ liệu (khoảng 1/8) tập trung chính xác tại giá trị trung vị, tạo nên một đặc thù thống kê cần lưu ý khi thực hiện các phép so sánh sau này.

---

## 4. Kiểm chứng Trạng thái Hoạt hóa
Sau khi chạy batch dữ liệu qua mô hình, chúng ta thu được 12 ma trận hoạt hóa, mỗi ma trận có kích thước $[16, 512, 3072]$. 
- `16`: Số chuỗi trong batch.
- `512`: Số tokens trong mỗi chuỗi.
- `3072`: Số nơ-ron MLP mở rộng.
Sự đồng nhất về kích thước trên tất cả các tầng xác nhận hệ thống Hooks đã hoạt động chính xác và sẵng sàng cho việc tính toán thống kê cường độ (magnitude) ở Phần 2.

---

## 5. Kết Luận Phần 1
Chúng ta đã hoàn tất việc xây dựng "phòng thí nghiệm nội soi" cho GPT-Neo. Việc phân nhóm token theo độ dài ký tự cung cấp một biến độc lập rõ ràng để nghiên cứu sự tác động lên biến phụ thuộc là hoạt hóa nơ-ron. Giai đoạn tiếp theo sẽ đi sâu vào việc xây dựng các biểu đồ histogram để so sánh trực tiếp các nhóm này, nhằm tìm kiếm các xu hướng chọn lọc độ dài xuyên suốt các tầng của mô hình.

---

## Tài liệu tham khảo (Citations)
1. Thử thách về Activation histograms trên GPT-Neo dựa trên `aero_LLM_11_CodeChallenge Activation histograms by token length (part 1).md`. Thiết lập hệ thống Hooks đa tầng và phân tích thống kê độ dài token từ FineWeb.
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
| 📌 **[Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 1)](aero_llm_11_codechallenge_activation_histograms_by_token_length_part_1_.md)** | [Xem bài viết →](aero_llm_11_codechallenge_activation_histograms_by_token_length_part_1_.md) |
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
