
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
# Thử Thách Lập Trình (Code Challenge): MI Và Khoảng Cách Token (Phần 1)

## Tóm tắt (Abstract)
Báo cáo này giải quyết bài toán định lượng thông tin tương hỗ giữa các "ngữ cảnh nội bộ" (local context) của một nhóm các cụm từ giống hệt nhau, chạy trên mô hình khổng lồ GPT-2 XL (48 Layers). Thông qua đoạn văn bản mồi thuộc chủ đề Cà phê Thổ Nhĩ Kỳ, có tổng cộng 7 lần từ "coffee" lặp lại. Thay vì đo lường Mutual Information (MI) dọc trên các Token, ta đo lường MI giữa 7 từ "coffee" này kết nối qua $1600$ chiều ẩn (Hidden Dimensions). Đồng thời, bài thực nghiệm giới thiệu kỹ thuật loại bỏ biệt lệ (Outliers Trimming) bằng Z-Score và sử dụng Hệ số tương quan hạng Kendall (Kendall's Tau) để khám phá mối liên hệ nghịch biến giữa Độ lớn của MI và Khoảng cách vị trí của hệ Token.

---

## 1. Mở Đầu (Introduction)
Trong việc xử lý ngôn ngữ tự nhiên, một từ đơn lẻ (Ví dụ "coffee") có thể lặp lại nhiều lần trong đoạn văn, mỗi lần lại mang một tiểu ngữ cảnh (local context) hơi khác nhau. Điều này đặt ra câu hỏi hấp dẫn:
*"Với cùng một gốc Token id, biểu diễn kích hoạt $Attention$ ở các vị trí khác nhau có chia sẻ thông tin gì không? Và nếu chúng xa nhau về mặt vật lý, liệu khả năng mang tin cấu trúc có bị sụt giảm không?"*

Để tìm lời giải, chúng tôi khai thác mạng GPT-2 XL, tập trung vào kết xuất cuối của Attention Block (được gọi là $c\_proj$).

---

## 2. Tiền Xử Lý Dữ Liệu Và Quy Trình Loại Bỏ Nhiễu (Outliers)

### 2.1. Nạp Hàm Kích Hoạt "Target Words"
Mô hình nhập nội dung đoạn văn có chứa 7 lần xuất hiện từ "coffee". 
Tại Layer 3, dữ liệu kích hoạt của mỗi từ "coffee" tương ứng là một vector dài 1600 chiều (1600 Dimensions). Chúng ta tiến hành vẽ Scatter Plot so khớp Vector thứ $1$ và Vector thứ $3$.

### 2.2. Xử Lý Các Điểm Dữ Liệu Cực Đoan (Extreme Values)
Khi quan sát biểu đồ hoạt động của mạng LLM, thường xuất hiện khoảng 1-2 điểm nhiễu (neurons) có cường độ kích hoạt "phóng vút" lên rất cao so với đám mây phân bổ trung tâm. Mặc dù đây là các tín hiệu mạng bình thường (không phải lỗi bộ nhớ), hiện tượng cực đỉnh (extreme values) lại phá nát các thuật toán đo chia Histogram của MI.

**Cách khắc phục:** Không gian hóa Z-Score. 

$$

Z = \frac{x_i - \bar{x}}{\sigma}

$$

Áp dụng Z-score cho cả 2 vector. Bất kỳ giá trị nào có $|Z| > 4$ (Vượt quá 4 lần độ lệch chuẩn) sẽ bị gán cờ Outlier và dạt bỏ khỏi danh sách đo MI. 
Việc cắt tỉa dữ liệu thừa (Trimmed Data) này giúp đẩy MI từ một con số bị dìm do nhiễu $\to$ phục hồi lại điểm tương hỗ cốt lõi, phản biện lại nhược điểm của công thức histogram Manual.

---

## 3. Khoảng Cách Vị Trí Vs Tương Quan Thông Tin (Analysis & Results)

### 3.1. Tính Ma Trận Tương Hỗ Chéo Điểm (Pairwise Token MI Matrix)
Vì có 7 mục tiêu, ma trận phân tích sẽ có cấu trúc $7 \times 7$. Bỏ qua chéo chính và nửa dưới đối xứng, phần dữ liệu nửa trên chứa $MI$ giữa toàn bộ các cặp khoảng cách từ 1 đến 7. 

### 3.2. Ma Trận Khoảng Cách Cục Bộ (Inter-token Distances)
Khoảng cách vật lý giữa hai từ "coffee" được tính giản lược bằng số lượng Token nằm xen giữa chúng. Không phải Embedding Vector Distances. (Do đây là số nguyên bậc thứ tự, không phải biến thiên liên tục).

### 3.3. Phương Trình Tương Quan Xếp Hạng Kendall (Kendall's tau)
Vì biến quãng cách là một chuỗi mang tính định hạng (ordinal variable - số nguyên ngắt quãng), việc dùng Tương quan Pearson là sai nguyên lý thống kê. Ta phải chuyển qua hệ số **Kendall's Tau** (Tương tự Pearson, chạy từ $-1 \to 1$).

**Kết quả Scatter Plot kết nối:**
Biểu đồ trải hiển thị mối tương quan nghịch đảo rõ rệt $\to$ `Hệ số r Kendall = -0.5`. 
- **Giải thích:** Hai từ "coffee" đứng càng gần nhau trong một câu, chỉ số M.I giữa biểu diễn không gian $Attention$ của chúng càng mãnh liệt. Khi hai từ bị đẩy ra xa nhau chừng vài chục định vị, tiểu lớp ngữ cảnh bị vỡ vụn, khiến khả năng san sẻ tương đồng ý niệm rơi thẳng đứng.

---

## 4. Kết Luận
Bài toán Token Distance vén màn cơ chế "Nhớ gần" (Local Memory Context) của Multi-head Attention thông qua thấu kính Mutual Information. Bằng việc chắt lọc Z-score Outliers, ta có thể xây dựng các biểu đồ tương tự Pearson nhưng dành cho các đại lượng phi tuyến cực kỳ chính xác. Ở phần sau, nghiên cứu sẽ phát triển mô hình này mở rộng xuyên suốt 48 Blocks (Laminar Profile) để xem xét định kiến nội dung ở vùng biến đổi sâu nhất (Deep Layers).

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu lệnh code trích xuất từ thí nghiệm: `aero_LLM_11_CodeChallenge Attention to coffee MI and token distances (part 1).md` (Giới thiệu hàm tính Z-Score $>4$, Kendall tau Correlation và nguyên lý MI của Token cặp).
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
| 📌 **[Thử Thách Lập Trình (Code Challenge): MI Và Khoảng Cách Token (Phần 1)](aero_llm_11_codechallenge_attention_to_coffee_mi_and_token_distances_part_1_.md)** | [Xem bài viết →](aero_llm_11_codechallenge_attention_to_coffee_mi_and_token_distances_part_1_.md) |
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
