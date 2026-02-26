
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
# Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 2)

## Tóm tắt (Abstract)
Tiếp nối phần 1, nghiên cứu này đi sâu vào việc đối chiếu và làm rõ sự khác biệt giữa hai phương pháp đo lường thống kê: Độ tương đồng Cosine (Cosine Similarity) và Hệ số Tương quan (Correlation Coefficient) trên các không gian ma trận Truy vấn (Query - $Q$), Khóa (Key - $K$), và Giá trị (Value - $V$). Bằng các thiết lập mặt nạ ma trận (matrix masks) để cô lập dữ liệu và so sánh với phân phối ngẫu nhiên, kết quả chỉ ra rằng **độ lệch trung bình (mean offsets)** đóng vai trò là một đặc tính mã hóa thực thụ và mang tính quyết định trong mạng lưới ngôn ngữ nội tại.

---

## 1. Mở Đầu (Introduction)
Trong lý thuyết phân tích biểu diễn mạng, cách chúng ta thiết lập phép đo có thể làm thay đổi hoàn toàn diễn giải về hoạt động của các nơ-ron:
- **Độ tương đồng Cosine** bảo toàn hoàn toàn các vector gốc, bao gồm cả khoảng lệch trung bình (mean offset) của tín hiệu.
- **Hệ số Tương quan (Pearson Correlation)** lại yêu cầu thực hiện bước chuẩn hóa trung tâm (mean-centering) – tức trừ đi giá trị trung bình trước khi xét sự đồng biến. 

Việc so sánh hình thái dữ liệu khi có và không có chuẩn hóa trung bình mở ra mảnh ghép quan trọng giúp giải thích tại sao LLMs lại sinh ra các vector tương đồng mang giá trị cực biên (âm hoặc dương tuyệt đối).

---

## 2. Phương Pháp Phân Tích (Methodology)

### 2.1. Phân Tách Dữ Liệu Bằng Ma Trận Mặt Nạ (Matrix Masking)
Do khối dữ liệu tensor gộp cả Q, K, V $(\text{size} = 2304)$, thực nghiệm xây dựng một ma trận mặt nạ bằng phép nhân ngoài (Outer Product) kết hợp trích xuất tam giác trên (Upper Triangular).
Ta gán giá trị định danh: $Q = 1$, $K = 2$, $V = 3$ để tạo mốc lưới:
- Tương tác nội bộ cụm: $Q-Q$ (nhân ra 1), $K-K$ (nhân ra 4), $V-V$ (nhân ra 9).
- Tương tác chéo cụm: Ví dụ $Q-K$ (nhân ra 2), $Q-V$ (nhân ra 3)...
Nhờ mạng mask này, hàm phân tích dễ dàng cô lập và quét được hàng trăm nghìn tương tác đơn lẻ cho từng thành phần (Targets/Non-targets/Random).

### 2.2. Xây Dựng Biến Đối Chứng Ngẫu Nhiên (Randomization Baseline)
Để đảm bảo các quy luật tìm thấy là có ý nghĩa, toàn bộ dữ liệu kích hoạt (activations) gốc được giữ nguyên nhưng vị trí bị xáo trộn ngẫu nhiên toàn hạt (shuffling). Hệ số tính toán trên tập xáo trộn này cho ra phổ phân phối chuẩn Gaussian đơn giản trung tâm tại $0$, đóng vai trò làm mẫu đối chứng nền.

---

## 3. Khám Phá Khối Dữ Liệu Nội Tại (Results & Analysis)

### 3.1. Phân Tích Với Điểm Cosine Similarity
Qua lăng kính bảo toàn nguyên bản trung bình:
- **Biến thiên của Q-Q và K-K:** Đồ thị Histogram bộc lộ hình dáng uốn cong hướng nụ cười (smile) – phân bổ tập trung vào hai thái cực $\approx 1$ và $\approx -1$. Các đa tạp Q và K có hiện tượng kết xích mạnh để định vị ý nghĩa của token chỉ định.
- **Biến thiên của V-V:** Dải phân bố bị dàn mỏng hơn và không cựcan (decoupled). Ma trận $V$ có xu hướng thu hồi đa diện hơn dựa trên lịch sử văn cảnh thay vì mã hóa token tĩnh như $Q, K$.
- **Giao thoa Q-K:** Mật độ Cosine rất cao. Điều này dễ lý giải vì mặt bản chất cơ học, $Q$ và $K$ sinh ra để dot-product với nhau nhằm tính mức độ Attention Score. Sự tương quan giữa chúng chính là tiền đề để truyền tín hiệu tới $V$.

### 3.2. Phân Tích Với Hệ Số Tương Quan (Correlation Coefficient) 
Khi áp dụng việc trừ đi số trung bình (Mean-Centering), kết quả trả về vô cùng kinh ngạc:
- Các điểm cực tả/cực hữu hoàn toàn biến mất. Đồ thị trở thành phân phối chuẩn Gaussian quanh ngưỡng 0 tương tự như dữ liệu ngẫu nhiên.
- Các lưới biểu diễn giữa cụm ma trận (như Q, K, V) cũng mất đi sự sai biệt cá tính nhận dạng. Phân bổ của chúng trở nên đồng hóa với nhau.

---

## 4. Kết Luận (Conclusion)
Sự sai lệch ngoạn mục giữa hệ quy chiếu chứa trung bình (Cosine) và triệt tiêu trung bình (Correlation) khẳng định một tuyên bố vật lý mạng trọng yếu:  
**Khoảng lệch trung bình (Mean Offsets) không phải là "nhiễu" số học, mà chính là những mã thông tin cốt lõi (coding normalities) mà LLMs dùng để vận hành.**  
Thay vì các nơ-ron hoạt động nhảy nhót biên độ (Variance) riêng lẻ, mạng nơ-ron nhúng thông tin liên kết trực tiếp vào việc cả cụm vector "nổi lên" hay "chìm xuống" một cách đại cục. Khám phá này (Mechanistic Interpretability) chỉ ra tính phức hợp vi lượng đòi hỏi nhiều nghiên cứu tách biệt (như theo từng Head độc lập) để truy vết trọn vẹn ý đồ của AI.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh gốc: `aero_LLM_02_Token-related similarities within and across Q, K, V matrices (part 2).md` (Mô tả kỹ thuật Masking, Randomization, so sánh đối chiếu Cosine vs. Coefficient Correlation).
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 1)](aero_llm_01_token_related_similarities_within_and_across_q_k_v_matrices_part_1_.md) | [Xem bài viết →](aero_llm_01_token_related_similarities_within_and_across_q_k_v_matrices_part_1_.md) |
| 📌 **[Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 2)](aero_llm_02_token_related_similarities_within_and_across_q_k_v_matrices_part_2_.md)** | [Xem bài viết →](aero_llm_02_token_related_similarities_within_and_across_q_k_v_matrices_part_2_.md) |
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
