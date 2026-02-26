
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [13 investigating layers](../index.md)

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
# Thử Thách Lập Trình (Code Challenge): MI Và Khoảng Cách Token (Phần 2)

## Tóm tắt (Abstract)
Thực nghiệm này nâng cấp khảo sát về tác động của khoảng cách vật lý giữa hai từ giống nhau ("coffee") đối với Mutual Information (MI), mở rộng trên toàn bộ biểu đồ 48 Layers của GPT-2 XL. Quá trình tính toán diễn ra song song trên cả biến thể nhánh $Attention$ và mạng $MLP$. Thông qua việc kết hợp các tiêu chuẩn kiểm định mạnh như *T-tests độc lập*, *Chuyển đổi Fisher Z-Transform cho hệ số tương quan*, và *Hiệu chỉnh đa so sánh FDR (False Discovery Rate/Bonferroni)*, báo cáo vạch ra ranh giới rẽ nhánh rõ rệt giữa nhiệm vụ dung nạp bối cảnh mở rộng của Attention và cơ chế nhớ tĩnh của MLP. 

---

## 1. Mở Đầu (Introduction)
Ở Phần 1, chúng ta đã chứng minh cơ bản tại Layer 3: Hai từ cùng gốc đứng càng xa nhau thì thông tin chia sẻ nội bộ của chúng càng nghèo nàn. Tuy nhiên, kiến trúc máy học LLM là một hành trình đi dần vào chiều sâu (depth propagation). Phần 2 đặt ra hai giải pháp nâng cao hơn: 
1. Diễn giải sự thay đổi hiệu ứng khoảng cách này xuyên qua 48 Transformer blocks.
2. Đối chiếu trực diện vai trò tạo lập liên kết (M.I) giữa hạt nhân truy vấn song song ($Attention\ C\_proj$) và hạt nhân tuyến tính xử lý hàm tiến ($MLP\ C\_proj$).

---

## 2. Nâng Cấp Phương Pháp Thống Kê (Methodology Expansions)

### 2.1. Vòng Lặp Trải Phẳng (Laminar Loop)
Thiết lập mảng 3 chiều ma trận `my_results = (2 x 48 x 2)` tương đương: [Attention / MLP] $\times$ [Layers 1...48] $\times$ [Ave18-RAGe MI / Kendall tau correlation]. Việc loại bỏ nhiễu Z-score $> 4$ (Outliers Trimming) vẫn luôn được duy trì ở toàn bộ các cấp tính toán.

### 2.2. Kiểm Định T-Test Giữa MLP Và Attention
Để xác nhận MI tại nhánh Attention có thực sự khác biệt so với MI của nhánh MLP ngay tại cùng một Layer hay không, ta lấy mảng dữ liệu (Tất cả Pairwise MI non-zero) của hai bên và cho chạy mô hình $Independent\ T-Test$ (Thu được $t-statistic$ và $p-value$). Để ngăn chặn sai lầm loại I do "test mỏi tay" 48 lần, bộ hiệu chỉnh đa biến Bonferroni hoặc FDR được kích hoạt.

### 2.3. Chuyển Đổi Fisher Z-Transform Cho So Sánh Correlation
Để so sánh hai hệ số tương quan (Kendall) của Attention và MLP, ta không thể dùng T-test vì nó không phải mẫu phân bổ đo lường tuyệt đối. Ta sử dụng Fisher Z-transform:
$$ Z = \frac{ \text{arctanh}(r_{att}) - \text{arctanh}(r_{mlp}) }{\sqrt{2 / (N - 3)}} $$
Kiểm tra Z-score này trên Phân phối tích lũy chuẩn (Normal CDF) sẽ cho phép xác định độ khác biệt mang ý nghĩa thống kê của lực hút nghịch biến giữa hai phân mảng.

---

## 3. Khám Phá Biểu Đồ Lớp (Analysis & Visualizations)

### 3.1. Sự Trỗi Dậy Của Attention Chống Lại MLP
Biểu đồ *Ave18-RAGe M.I Profile* trình bày một khuynh hướng lôi cuốn:
- **Tầng Nông (Early Layers):** Cơ chế $MLP$ chứa M.I cao hơn so với $Attention$. Giai đoạn đầu, MLP bám sát vào định nghĩa thô của từ tĩnh, bảo toàn bộ nhớ về mặt khái niệm độc lập. Do đó các Token giống nhau "tương thông" thông tin rất lớn.
- **Tầng Sâu (Deep Layers):** Quỹ đạo $Attention$ đi lên tiệm cận trên, kéo mức trung bình chia sẻ M.I ngày một mạnh, trái ngược với $MLP$ rơi rớt cắm mỏ và đi ngang rập khuôn. Lý giải cơ học: Càng chìm sâu, Attention bị áp lực phải kết nối "ngữ cảnh vĩ mô". Để có thể đoán từ tiếp theo, nó phải lôi kéo lịch sử chồng chéo từ cực xa $\to$ nó chủ động làm giàu thông tin cho mọi liên kết cặp của chữ "coffee". 

### 3.2. Chênh Lệch Tương Quan Nghịch Biến (Kendall Correlation Stats)
Khuynh hướng khoảng cách xa sinh ra MI yếu luôn đạt biểu số Correlation Negative (Xoay quanh khoảng $-0.5$). Biểu đồ Z-value cho thấy sự phân ly rõ rệt: $Attention$ xử lý vấn đề token xa nhau mượt mà và linh động hơn nhiều so với hệ tĩnh tại $MLP$ sau Tầng thứ 10. 

### 3.3. So Sánh Thuật Toán Thủ Công Và Scikit-Learn
Thực hiện chạy toàn bộ hệ quy trình với nhân KDE Scikit-learn (Mất tầm khoảng 2 phút do Data Cặp nhỏ). So sánh trực quan đối chứng cho thấy: Các sai khác về đồ thị Laminar hoàn toàn mang tính chất tịnh tiến vô hại. Mọi tỷ lệ tương đối (Relative Values) giữa các không gian được bảo toàn tuyệt đối, gia cố thêm niềm tin rằng thuật toán tính Histogram MI Manual là giải pháp thay thế hoàn hảo cho tập dữ liệu Big Data.

---

## 4. Kết Luận
Bằng việc triển khai kiểm định độ lệch cực đỉnh và đo lường khoảng cách từ định hạng, tính năng Mutual Information là một trạm radar nhạy bén để bắt sóng cơ học lõi: $MLP$ đóng khuôn khái niệm ở tầng cao, còn $Attention$ đan kết mạng nhện vĩ mô dải dài tít tận đáy phễu.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh khảo sát tĩnh: `aero_LLM_12_CodeChallenge Attention to coffee MI and token distances (part 2).md` (Thiết lập hàm Fisher Z-Transform, Independent T-Test, Loop Laminar Analysis, so gánh đặc tính Attention - MLP).
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
| [Thử Thách Lập Trình (Code Challenge): Khảo Sát Số Chiều Hiệu Quả Trên Pythia 2.8B](aero_llm_07_codechallenge_dimensionalities_in_pythia_2_3b.md) | [Xem bài viết →](aero_llm_07_codechallenge_dimensionalities_in_pythia_2_3b.md) |
| [Lý Thuyết Thông Tin: Đo Lường Entropy Và Mutual Information](aero_llm_08_mutual_information_theory_and_code.md) | [Xem bài viết →](aero_llm_08_mutual_information_theory_and_code.md) |
| [Phân Tích Thông Tin Tương Hỗ Dọc Theo Các Tầng Của Mô Hình Ngôn Ngữ (Pairwise Mutual Information Through LLMs)](aero_llm_09_pairwise_mutual_information_through_the_llm.md) | [Xem bài viết →](aero_llm_09_pairwise_mutual_information_through_the_llm.md) |
| [Phân Tích Đối Chiếu Đo Lường Tương Quan: Mutual Information và Covariance](aero_llm_10_mutual_information_vs_covariance.md) | [Xem bài viết →](aero_llm_10_mutual_information_vs_covariance.md) |
| [Thử Thách Lập Trình (Code Challenge): MI Và Khoảng Cách Token (Phần 1)](aero_llm_11_codechallenge_attention_to_coffee_mi_and_token_distances_part_1_.md) | [Xem bài viết →](aero_llm_11_codechallenge_attention_to_coffee_mi_and_token_distances_part_1_.md) |
| 📌 **[Thử Thách Lập Trình (Code Challenge): MI Và Khoảng Cách Token (Phần 2)](aero_llm_12_codechallenge_attention_to_coffee_mi_and_token_distances_part_2_.md)** | [Xem bài viết →](aero_llm_12_codechallenge_attention_to_coffee_mi_and_token_distances_part_2_.md) |
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
