
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
# Thử Thách Lập Trình (Code Challenge): Khảo Sát Số Chiều Hiệu Quả Trên Pythia 2.8B

## Tóm tắt (Abstract)
Thực nghiệm này mở rộng ranh giới phân tích số chiều hiệu quả (Effective Dimensionality) từ tập kích hoạt PCA & SVD sang một LLM ở quy mô tỷ tham số: Mô hình **Pythia 2.8B** (họ EleutherAI). Thông qua việc đánh giá tỷ lệ phần trăm số chiều tối đa (Percent of maximum possible dimensionality) tại 2 mốc phương sai $95\%$ và $99\%$ thay vì đong đếm số lượng components thô, nghiên cứu phát hiện ra sự tương phản kịch tính giữa việc mô hình xử lý văn bản tự nhiên (Tiếng Anh phức tạp) so với ngôn ngữ lập trình thuần túy (HTML/CSS).

---

## 1. Mở Đầu (Introduction)
Phân tích Dimensionalities càng mang nhiều ý nghĩa diễn dịch khi chúng ta đánh giá trên hệ thống có bộ nhớ cực lớn. Pythia 2.8B sở hữu một thân hình với hàm nhúng (Embeddings) đạt chiều sâu 2560 chiều, và khi đi qua tầng MLP, nó nở ra gấp 4 lần lên đến $\sim 10,000$ chiều. 

Thử thách code này có hai nhiệm vụ:
- Định lượng "tính sử dụng thực dụng" của mô hình khổng lồ này so với khoảng không gian lý thuyết (Ambient Dimensionality).
- Tương phản độ phức tạp của hai dạng dữ liệu: Văn xuôi (Natural texts) và Mã lệnh cấu trúc (Structured tags như HTML/CSS) để tìm hiểu tại sao một mô hình bị nén tham số vẫn có thể code rất tốt.

---

## 2. Nâng Cấp Phương Pháp Luận (Methodology)

### 2.1. Đổi Đơn Vị Tỷ Lệ (y-axis Scaling)
Do các LLMs có biên độ kích thước ma trận khác nhau, việc dùng số nguyên (ví dụ cần 55 hay 150 components) không đem lại giá trị mang tính so sánh chung. Giải pháp là quy đổi sang **Tỷ lệ phần trăm tổng số chiều tối đa**.
Bằng việc chia cho số lượng singular values $\sigma$ khả dụng (số lượng min giữa token span và architecture vectors), toàn bộ đồ thị biểu diễn sẽ chuyển tải thông điệp chung: *"Phần trăm khu vực làm việc thực tế hệ thống đang phải huy động"*.

### 2.2. Kiểm Thử Hệ Sinh Thái Văn Bản Mới
Ngoài sử dụng khối 1000 tokens của sách *Alice in Wonderland*, hệ thống thu nạp đoạn code HTML/CSS hoàn chỉnh từ một website thực tế. 
- Biến ngẫu nhiên đối chứng: Vẫn duy trì cơ chế đánh tráo vị trí (Shuffled tokens) như file 06 để kiểm định tính trật tự.
- Gắn 2 ngưỡng PCA Thresholds là $\ge 95\%$ và $\ge 99\%$ phương sai (variance explained) nhằm kiểm định độ bền bỉ của phép toán. Sức giãn nở của dimensionalities dù chênh lệch thông số nhưng không được làm gãy đi tổng thể đồ hình chung.

---

## 3. Khám Phá Đặc Tính Không Gian Dữ Liệu (Analysis & Visualizations)

### 3.1. Sự Tương Đồng Về Kết Nối Văn Xuôi
Đối với khối văn bản tự nhiên, bản đồ Laminar của Pythia 2.8B tiếp tục cho thấy sự thống trị của quy luật mở rộng số chiều (dimensionality expansion process). Càng vào sâu, văn xuôi buộc hệ thống lôi kéo một chiều sâu nhận thức liên quan phức tạp hơn hẳn so với những tokens bị xới tung mất trật tự (shuffled text).

### 3.2. Cú Sốc Tối Ưu Hóa Của Mã HTML/CSS
Xảy ra hiện tượng đảo chiều ngoạn mục khi đẩy HTML/CSS vào Transformer:
1. **Lượng tiêu thụ không gian siêu thấp:** Các thẻ đánh dấu HTML/CSS chỉ dùng đến tối đa mức biến thiên là xấp xỉ $\mathbf{20\%}$ tổng số chiều vector khả dụng (Maximum Dimensionality) của mô hình. Tức là 80% bộ não của mô hình đang "rảnh rỗi" hoặc không cần viện tới để sản xuất code.
2. **Nghịch lý Dữ Liệu Shuffled:** Đối lập hoàn toàn với văn xuôi, tập token HTML khi bị sắp xếp rời rạc lộn xộn (shuffled) lại khiến mô hình tiêu tốn nhiều chiều liên kết hơn. Điều kiện trật tự (ordered structure) của HTML quá tĩnh tại và máy móc, dẫn đến độ rộng nén thông tin (compressibility) là vô cùng hoàn hảo.

---

## 4. Kết Luận (Conclusion)
Thông qua phân tích quy mô kiến trúc Pythia 2.8B, bài thí nghiệm minh chứng thêm sự thành công của Phương pháp PCA. 
Việc xử lý ngôn ngữ lập trình không tiêu thụ một vùng không gian tổ hợp điên cuồng như văn xuôi, giải thích lý do vì sao những mô hình ngôn ngữ kích thước bé (Compressed / Small Models - chừng 1-3 tỉ tham số) cũng có thể trở thành chuyên gia lập trình xuất sắc. Phép đo (Effective dimensionality dimension metrics) này không những hé mở trật tự cơ học bên trong LLM mà còn mở ra nền móng cho việc tối ưu bộ nhớ.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và kiến trúc nguồn: `aero_LLM_07_CodeChallenge Dimensionalities in Pythia 2.3B.md` (Giới thiệu các hàm tính toán Scaling phần trăm PCA Dimension Dimensionality cho Model lớn và sự so sánh chênh lệch giữa Semantic Texts với HTML code block).
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 1)](aero_llm_01_token_related_similarities_within_and_across_q_k_v_matrices_part_1_.md) | [Xem bài viết →](aero_llm_01_token_related_similarities_within_and_across_q_k_v_matrices_part_1_.md) |
| [Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 2)](aero_llm_02_token_related_similarities_within_and_across_q_k_v_matrices_part_2_.md) | [Xem bài viết →](aero_llm_02_token_related_similarities_within_and_across_q_k_v_matrices_part_2_.md) |
| [Thử Thách Lập Trình (Code Challenge): Phân Tích Độ Tương Đồng Của Token Xuyên Suốt Các Tầng Ẩn](aero_llm_03_codechallenge_token_related_similarities_across_layers.md) | [Xem bài viết →](aero_llm_03_codechallenge_token_related_similarities_across_layers.md) |
| [Phân Tích Sự Phân Cụm và Tương Đồng Biểu Diễn (RSA) Trong Ma Trận Q và K](aero_llm_04_grouping_and_rsa_in_q_and_k_matrices.md) | [Xem bài viết →](aero_llm_04_grouping_and_rsa_in_q_and_k_matrices.md) |
| [Khảo Sát Phân Tầng (Laminar Profile) Về RSA Và Sự Chọn Lọc Phân Nhóm](aero_llm_05_codechallenge_laminar_profile_of_rsa_and_category_selectivity.md) | [Xem bài viết →](aero_llm_05_codechallenge_laminar_profile_of_rsa_and_category_selectivity.md) |
| [Phân Tích Số Chiều Hiệu Quả (Effective Dimensionality) Thông Qua PCA](aero_llm_06_effective_dimensionality_analysis_with_pca.md) | [Xem bài viết →](aero_llm_06_effective_dimensionality_analysis_with_pca.md) |
| 📌 **[Thử Thách Lập Trình (Code Challenge): Khảo Sát Số Chiều Hiệu Quả Trên Pythia 2.8B](aero_llm_07_codechallenge_dimensionalities_in_pythia_2_3b.md)** | [Xem bài viết →](aero_llm_07_codechallenge_dimensionalities_in_pythia_2_3b.md) |
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
