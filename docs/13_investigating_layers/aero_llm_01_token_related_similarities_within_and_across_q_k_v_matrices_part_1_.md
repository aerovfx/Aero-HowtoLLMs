
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [13 investigating layers](index.md)

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
# Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 1)

## Tóm tắt (Abstract)
Bài viết này điều tra các cơ chế biểu diễn của mạng nơ-ron ngôn ngữ lớn (LLMs), cụ thể là mô hình GPT-2, ở cấp độ tầng ẩn (layer-level). Trọng tâm nghiên cứu là phân tích mối quan hệ giữa các vector kích hoạt Truy vấn (Query - $Q$), Khóa (Key - $K$) và Giá trị (Value - $V$) dựa trên hệ số tương quan (Correlation) và độ tương đồng Cosine (Cosine Similarity). Thông qua phương pháp cô lập token đích định sẵn (từ "her") trong các ngữ cảnh câu khác biệt, chúng tôi phát hiện ra những quy luật phân bố tương đồng hội tụ rất mạnh trong không gian biểu diễn cơ cấu mạng Attention.

---

## 1. Mở Đầu (Introduction)
Phân tích mô hình ngôn ngữ lớn ở cấp độ các tầng ẩn (layer-level) cung cấp một cái nhìn tổng quan về cách thông tin được tổ chức và xử lý theo từng khối cấu trúc, cao hơn so với việc nghiên cứu từng nơ-ron (neuron) rời rạc. 

Bằng việc tìm hiểu cách biểu diễn token (token embeddings) biến đổi và tương tác qua các ma trận Tự chú ý (Self-Attention sublayers), ta có thể giải mã dần cơ chế nắm bắt ngữ cảnh của mô hình. Trong bài nghiên cứu này, chúng tôi đi sâu vào việc đối chiếu các đa tạp biểu diễn nội tại trong $Q$, $K$, $V$ khi một token hoàn toàn giống nhau được truyền qua các quy trình văn cảnh ngữ pháp (context) khác nhau.

---

## 2. Phương Pháp Thực Nghiệm & Đo Lường (Methodology)

### 2.1. Thiết Kế Tập Dữ Liệu và Bối Cảnh
Thực nghiệm sử dụng mô hình GPT-2 (small), tiến hành trích xuất hàm kích hoạt (hooking activations) thẳng từ vòng Transformer. Một hệ thống bao gồm $54$ câu văn ngắn được đưa vào mạng.
- **Cấu trúc chung:** Mọi câu đều chứa chung một *token đích* cố định (ví dụ: chuỗi `[space] her`). Do đó, bản sắc cốt lõi của token đích là hoàn toàn giống hệt nhau (identical) về định danh đầu vào.
- **Tính độc lập:** Điều làm nên khối dữ liệu đối sánh là token đứng trước/sau và tổng chiều dài mỗi chuỗi chứa token thay đổi - buộc hệ thống tự động chèn thêm đệm (padding). 

### 2.2. Đo Lường Bằng Độ Tương Đồng Cosine (Cosine Similarity)
Để kiểm chứng ma trận vector kích hoạt nội tại, ta áp dụng công thức Độ tương đồng Cosine, định nghĩa sự trùng lặp góc đo định hướng giữa vector $\mathbf{x}$ và vector $\mathbf{y}$:

$$

\text{Cosine Similarity}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2}\sqrt{\sum_{i=1}^{n} y_i^2}}

$$

*Khai triển tính toán vi mô:* Các độ đo này được ánh xạ đại số bằng cấu trúc phép nhân vô hướng của ma trận dữ liệu chuyển vị (transpose) lên chính nó, và chia cho chuẩn $L_2$ (matrix norm) nhằm tạo ra các tập hợp phân bổ nằm giới hạn trong khung giá trị lý tưởng $[-1, 1]$.

---

## 3. Khám Phá Khối Dữ Liệu Nội Tại (Results & Analysis)

Hệ thống trích xuất Tensor xuất bản ghi với dạng biến chiều $\mathbf{54 \times 8 \times 2304}$ (Tương ứng: Khối 54 chuỗi câu $\times$ 8 tokens mặc định tính padding $\times$ tổng concat của Q, K, V vì GPT-2 small có mức $n\_embed$ là $768$).

**Phân Tích Cấu Tạo:** Qua lăng kính đồ thị phân bố (Histogram) của độ tương đồng Cosine ở các chiều $Q-Q$ hoặc tương tác cặp, ta thấy:
- Khảo sát các kích hoạt ở từ "her" dọc theo 54 ngữ cảnh cho một kết quả kinh ngạc: Độ tương đồng Cosine của các vector đích biểu thị một hình trạng hội tụ hướng hai cực, thường là **dương rất đậm** hoặc thỉnh thoảng sẽ mang xu hướng **âm tương phản rõ rệt** (Strong Negative/Positive).
- Sự tồn tại của token đích giống nhau áp đảo ngữ cảnh khác nhau lên kích hoạt không gian, giữ các điểm trên đồ thị phân tán (scatter-plot) gộp vào một hệ tương quan hệ số cao (ví dụ: tương tác $ > 0.9$ trên hệ quy chiếu chéo của các câu văn).

---

## 4. Kết Luận (Conclusion)
Thông qua thủ pháp móc nối các tầng ẩn của mô hình tại vòng lặp thứ $6$ (Layer-6), bài thực nghiệm chứng minh sự ổn định cơ học đáng lưu tâm tại $Q$, $K$, $V$ đối với nhóm token đích mang tính nguyên bản liên kết. Việc tính độ đo Cosinus tiết lộ khả năng xuất sắc của mạng trong việc duy trì ý nghĩa định danh ban đầu, chống lại xu hướng thay đổi hoàn toàn quỹ đạo số học do xáo trộn văn cảnh xung quanh.

Dữ liệu thực nghiệm này cung cấp tiền đề để nghiên cứu sâu thêm về nhóm cụm chức năng học thuật trên mạng ngôn ngữ nhiều tỉ tham số hơn.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh gốc: `aero_LLM_01_Token-related similarities within and across Q, K, V matrices (part 1).md` (Khảo cứu cách tính Cosinus, xây dựng kịch bản 54 câu mô phỏng token đích và quy mô Tensor GPT-2 small PyTorch / Numpy).
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 1)](aero_llm_01_token_related_similarities_within_and_across_q_k_v_matrices_part_1_.md)** | [Xem bài viết →](aero_llm_01_token_related_similarities_within_and_across_q_k_v_matrices_part_1_.md) |
| [Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 2)](aero_llm_02_token_related_similarities_within_and_across_q_k_v_matrices_part_2_.md) | [Xem bài viết →](aero_llm_02_token_related_similarities_within_and_across_q_k_v_matrices_part_2_.md) |
| [Thử Thách Lập Trình (Code Challenge): Phân Tích Độ Tương Đồng Của Token Xuyên Suốt Các Tầng Ẩn](aero_llm_03_codechallenge_token_related_similarities_across_layers.md) | [Xem bài viết →](aero_llm_03_codechallenge_token_related_similarities_across_layers.md) |
| [Phân Tích Sự Phân Cụm và Tương Đồng Biểu Diễn (RSA) Trong Ma Trận Q và K](aero_llm_04_grouping_and_rsa_in_q_and_k_matrices.md) | [Xem bài viết →](aero_llm_04_grouping_and_rsa_in_q_and_k_matrices.md) |
| [Khảo Sát Phân Tầng (Laminar Profile) Về RSA Và Sự Chọn Lọc Phân Nhóm](aero_llm_05_codechallenge_laminar_profile_of_rsa_and_category_selectivity.md) | [Xem bài viết →](aero_llm_05_codechallenge_laminar_profile_of_rsa_and_category_selectivity.md) |
| [Phân Tích Số Chiều Hiệu Quả (Effective Dimensionality) Thông Qua PCA](aero_llm_06_effective_dimensionality_analysis_with_pca.md) | [Xem bài viết →](aero_llm_06_effective_dimensionality_analysis_with_pca.md) |
| [Thử Thách Lập Trình (Code Challenge): Khảo Sát Số Chiều Hiệu Quả Trên Pythia 2.8B](aero_llm_07_codechallenge_dimensionalities_in_pythia_2_3b.md) | [Xem bài viết →](aero_llm_07_codechallenge_dimensionalities_in_pythia_2_3b.md) |
| [Lý Thuyết Thông Tin: Đo Lường Entropy Và Mutual Information](aero_llm_08_mutual_information_theory_and_code.md) | [Xem bài viết →](aero_llm_08_mutual_information_theory_and_code.md) |
| [Phân Tích Thông Tin Tương Hỗ Dọc Theo Các Tầng Của Mô Hình Ngôn Ngữ (Pairwise Mutual Information Through LLMs)](aero_llm_09_pairwise_mutual_information_through_the_llm.md) | [Xem bài viết →](aero_llm_09_pairwise_mutual_information_through_the_llm.md) |
| [Phân Tích Đối Chiếu Đo Lường Tương Quan: Mutual Information và Covariance](aero_llm_10_mutual_information_vs_covariance.md) | [Xem bài viết →](aero_llm_10_mutual_information_vs_covariance.md) |
| [Thử Thách Lập Trình (Code Challenge): MI Và Khoảng Cách Token (Phần 1)](aero_llm_11_codechallenge_attention_to_coffee_mi_and_token_distances_part_1_.md) | [Xem bài viết →](aero_llm_11_codechallenge_attention_to_coffee_mi_and_token_distances_part_1_.md) |
| [Thử Thách Lập Trình (Code Challenge): MI Và Khoảng Cách Token (Phần 2)](aero_llm_12_codechallenge_attention_to_coffee_mi_and_token_distances_part_2_.md) | [Xem bài viết →](aero_llm_12_codechallenge_attention_to_coffee_mi_and_token_distances_part_2_.md) |
| [Phân Khảo Cấu Trúc Cụm (Clusters): Dấu Câu Nội Bộ vs Dấu Câu Kết Thúc Tập 1](aero_llm_13_codechallenge_clusters_in_internal_vs_terminal_punctuation_part_1_.md) | [Xem bài viết →](aero_llm_13_codechallenge_clusters_in_internal_vs_terminal_punctuation_part_1_.md) |
| [Phân Khảo Cấu Trúc Cụm (Clusters): Dấu Câu Nội Bộ vs Dấu Câu Kết Thúc Tập 2](aero_llm_14_codechallenge_clusters_in_internal_vs_terminal_punctuation_part_2_.md) | [Xem bài viết →](aero_llm_14_codechallenge_clusters_in_internal_vs_terminal_punctuation_part_2_.md) |
| [Thấu Kính Logit (The Logit Lens): Soi Sáng Tư Duy Tầng Trung Gian Của Mô Hình Ngôn Ngữ](aero_llm_15_the_logit_lens.md) | [Xem bài viết →](aero_llm_15_the_logit_lens.md) |
| [Thử Thách Lập Trình (Code Challenge): Ứng Dụng Logit Lens Trong Mạng BERT (Phần 1)](aero_llm_16_codechallenge_logit_lens_in_bert_part_1_.md) | [Xem bài viết →](aero_llm_16_codechallenge_logit_lens_in_bert_part_1_.md) |
| [Thử Thách Lập Trình (Code Challenge): Ứng Dụng Logit Lens Trong Mạng BERT (Phần 2)](aero_llm_17_codechallenge_logit_lens_in_bert_part_2_.md) | [Xem bài viết →](aero_llm_17_codechallenge_logit_lens_in_bert_part_2_.md) |
| [Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 1)](article_aero_llm_01_vn.md) | [Xem bài viết →](article_aero_llm_01_vn.md) |
| [Phân tích Chuyên Sâu Các Tầng Ẩn Trong Mô Hình Ngôn Ngữ Lớn (LLMs): Đo Lường, Biểu Diễn và Giải Mã Nội Tại](scientific_article_vn.md) | [Xem bài viết →](scientific_article_vn.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
