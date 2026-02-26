
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [11 investigating token embeddings](index.md)

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
# Ánh Xạ SVD Các Dải Điểm Nhúng Có Quan Hệ Chéo

## Tóm tắt

Một trong những giới hạn khi phân tích toàn bộ Không gian Embeddings bằng SVD (Singular Value Decomposition) là sự bão hòa nhiễu - những thành phần chính (Principal Components) thường đánh mất độ chi tiết do phải gánh đỡ một tỷ lệ phương sai khổng lồ của cả trăm ngàn cụm từ dị biệt. Giải pháp đối trọng là khoanh vùng tọa độ vi mô: Thay vì nén toàn thể mảng BERT, ta tạo ra các "Ma trận Con" (Submatrices) chứa duy nhất tổ hợp token mang đặc trưng nhóm (VD: Tên quốc gia, chữ số). Bài phân tích dưới đây minh chứng năng lực của SVD trong việc tìm ra những trục liên kết đồng dạng ẩn dật dưới các nhóm từ vựng cụ thể.

---

## 1. Kiến Tạo Ma Trận Con (Submatrices Embeddings)

Cấu hình thử nghiệm được thiết kế dựa trên 2 tập dữ liệu mẫu trích từ mô hình BERT:
1. **Tập 10 Chữ số đơn:** `["0", "1", "2", ..., "9"]`
2. **Tập 10 Quốc gia Liên Minh Châu Âu (EU):** `["France", "Germany", "Italy", "Spain", ...]` (Chọn lọc ưu tiên các quốc gia không bị băm ngang bởi tokenizer để đảm bảo luật *1 word = 1 token*).

Hai ma trận con nhận được (Matrix $M_{\text{digits}}$ và $M_{\text{EU}}$) có chung kích thước $10 \times 768$.
Tính độc lập phân phối (Orthogonality mapping) được khẳng định ngay từ bước thử nghiệm khi Ma trận Vectơ Trung Bình (Mean vectors) của tập Chữ số và tập EU trả về mức tương quan cực thấp ($r \approx 0.01$). Điều này chứng minh 2 đám mây tọa độ này bay xa nhau hoàn toàn trong cấu trúc dải ngân hà 768 chiều.

---

## 2. Loại Bỏ Đường Tiệm Cận Bằng Kỹ Thuật Dịch Tâm (Mean-Centering)

Trước khi tiến hành phân rã nhân ma trận $M$, mọi cấu trúc dữ liệu hình học tuyến tính đều phải tiến hành lùi tâm (Mean Centering).
Tính tịnh tiến này cưa bỏ khoảng cách dư thừa từ điểm $0$ đến lõi đám mây dữ liệu:

$$
\hat{M}_{i} = M_{i} - \mu
$$

*(Với $\mu$ là vector trung bình cực đại có độ dài bằng số cột kích thước D=768).*

Khi Mean-centering được thực thi chặt chẽ, đường quang phổ giá trị suy biến (Singular value spectrum / Scree plot) từ SVD sẽ có đặc tính rỗng dư tại giá trị cuối cùng. Nói cách khác, thuật toán cưa đi một *bậc tự do* (Rank minus 1), biểu diễn bằng việc singular value cuối cùng sẽ đâm thẳng về $0$. Nếu không lùi tâm, trục phân phối SVD sẽ dồn toàn bộ sự khác biệt vào Component-1 (Trục thứ 1), làm sai lệch khả năng đọc hiểu Component-2.

---

## 3. Khám Phá Ý Nghĩa SVD Bằng Phép Chiếu Nghịch Tập Hợp (Over-Projections)

### Khái Niệm Phép Chiếu Rộng Rãi:
Sau khi SVD thành công $\hat{M}_{\text{EU}} = U \Sigma V^T$, chúng ta thu được chùm Vector riêng biệt đặc tả tính "*Châu Âu*" nắm giữ tại hàng thứ tự đầu tiên của đa giác $V^T$ (Kí hiệu $V_{\text{top}}$).

Phép màu giải thích nằm ở bước sau: Thay vì giới hạn khảo sát trên 10 nước Châu Âu, ta lấy **toàn bộ 30.000 tokenizer còn lại của hệ BERT**, trừ đi $\mu_{\text{EU}}$, rồi nhân tích vô hướng đổ bóng toàn bộ 30.000 từ này lên trục $V_{\text{top}}$:

$$
\text{Projections} = (E_{\text{all\_tokens}} - \mu_{\text{EU}}) \cdot V_{\text{top}}
$$

### Diễn Dịch Chóp Đồ Thị (Extremes Projections):
Thống kê 30 token có tích vô hướng văng ra xa nhất trên Trục $V_{\text{top}}$ (Top positive / Top negative Projections) mở ra chân trời cơ chế máy học:
- Ở dải cực âm của Trục Châu Âu, chúng ta bắt gặp những từ vựng không hề nằm trong nhóm gốc đào tạo nhưng cùng một hệ trục địa lý ngôn ngữ như: *Latvian, Tallinn, Vilnius, Estonian*.
- Ở dải của ma trận Chữ Số, các cực đoan dự đoán kéo theo sự xuất hiện của các chuỗi text format số (VD: *Seven, Null, Zero, Divided*), chứng tỏ trục không gian toán học có khả năng nối kết hình dáng số ("7") với ký hiệu văn bản ("Seven").

---

## 4. Kết luận

Sự phân mảng Ma trận con (Submatrices Extracting) cung cấp một khung kính lúp mạnh mẽ giảm bớt nhiễu loạn ngẫu nhiên của toàn bộ thư viện ngôn ngữ tự nhiên. Phương pháp lấy SVD tạo ra ma trận V, rồi đem toàn bộ đại dương Embeddings phản kích dội ngược chiếu rọi lên $V$ chính là một chiếc kính rọi đèn soi sáng cấu trúc nội mạc (Mech Interpretability) cho thấy cách hàng tỷ ma trận thông số Neural Network móc nối khái niệm của con người thành mạng nhện tính toán.

---

## Tài liệu tham khảo

1. **Turian, J., et al. (2010).** *Word representations: A simple and general method for semi-supervised learning.* ACL.
2. **Deerwester, S., et al. (1990).** *Indexing by latent semantic analysis.* JASIS.
3. Tài liệu thực hành định lượng *SVD projections of related embeddings*.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [aero llm 01 codechallenge cosine similarity advanced part 1](aero_llm_01_codechallenge_cosine_similarity_advanced_part_1_.md) | [Xem bài viết →](aero_llm_01_codechallenge_cosine_similarity_advanced_part_1_.md) |
| [aero llm 02 codechallenge cosine similarity advanced part 2](aero_llm_02_codechallenge_cosine_similarity_advanced_part_2_.md) | [Xem bài viết →](aero_llm_02_codechallenge_cosine_similarity_advanced_part_2_.md) |
| [Theo Dõi Dòng Chảy Cosine Similarity Trên Trục Văn Bản Chuyên Tuần Tự (Word Sequences)](aero_llm_03_codechallenge_cosine_similarity_in_word_sequences.md) | [Xem bài viết →](aero_llm_03_codechallenge_cosine_similarity_in_word_sequences.md) |
| [Nghệ Thuật Vẽ Bản Đồ Nhiệt Ma Trận Nhúng Bằng Cường Độ Từ (Coloring Cosine Similarity)](aero_llm_04_codechallenge_coloring_cosine_similarity.md) | [Xem bài viết →](aero_llm_04_codechallenge_coloring_cosine_similarity.md) |
| [Ảo Ảnh Của Trí Tuệ Toán Học Trong Ngôn Ngữ: Sức Mạnh Của Random Embeddings](aero_llm_05_codechallenge_can_random_embeddings_be_interpreted.md) | [Xem bài viết →](aero_llm_05_codechallenge_can_random_embeddings_be_interpreted.md) |
| [Phương Pháp T-SNE Và Thuật Toán Phân Cụm DBSCAN: Chiếu Không Gian Đa Chiều Cho LLMs](aero_llm_06_t_sne_projection_and_dbscan_clustering_theory_.md) | [Xem bài viết →](aero_llm_06_t_sne_projection_and_dbscan_clustering_theory_.md) |
| [Phân Cụm Ngữ Nghĩa Qua Phép Chiếu t-SNE & Mật Độ DBSCAN (Python)](aero_llm_07_t_sne_projection_and_dbscan_clustering_python_.md) | [Xem bài viết →](aero_llm_07_t_sne_projection_and_dbscan_clustering_python_.md) |
| [Thách Thức Code: Tìm Lỗ Hổng Phân Cụm Bằng Bộ Lọc Bảng Chữ Cái Chữ X](aero_llm_08_codechallenge_cluster_the_x_terms.md) | [Xem bài viết →](aero_llm_08_codechallenge_cluster_the_x_terms.md) |
| [Phân Rã Token, Nhúng Và Phân Cụm Biểu Tượng Emojis Bằng Đồ Thị Mật Độ](aero_llm_09_codechallenge_tokenize_embed_and_cluster_happy_emojis.md) | [Xem bài viết →](aero_llm_09_codechallenge_tokenize_embed_and_cluster_happy_emojis.md) |
| [Phân Tích RSA (Representational Similarity Analysis) Giữa Các Mô Hình Ngôn Ngữ](aero_llm_10_rsa_representational_similarity_analysis_.md) | [Xem bài viết →](aero_llm_10_rsa_representational_similarity_analysis_.md) |
| [Phân Tích Độ Lệch RSA (Part 1): So Sánh Sự Bất Đồng Giữa Không Gian GloVe 50D và 300D](aero_llm_11_codechallenge_compare_embeddings_with_rsa_part_1_.md) | [Xem bài viết →](aero_llm_11_codechallenge_compare_embeddings_with_rsa_part_1_.md) |
| [Phân Tích Độ Lệch RSA (Part 2): Đối Chiếu Tương Quan Pearson Cho Khoảng Cách Cosine](aero_llm_12_codechallenge_compare_embeddings_with_rsa_part_2_.md) | [Xem bài viết →](aero_llm_12_codechallenge_compare_embeddings_with_rsa_part_2_.md) |
| [So Sánh Không Gian Nhúng: Word2Vec Và GPT-2 Qua Phân Tích RSA](aero_llm_13_codechallenge_word2vec_vs_gpt2.md) | [Xem bài viết →](aero_llm_13_codechallenge_word2vec_vs_gpt2.md) |
| [Bố Cục Đồ Thị Mạng (Network Graph) Thông Qua Ma Trận Cosine Similarity](aero_llm_14_codechallenge_graph_representation_of_cosine_similarities.md) | [Xem bài viết →](aero_llm_14_codechallenge_graph_representation_of_cosine_similarities.md) |
| [Số Học Tuyến Tính và Rút Trích Tương Đồng Giữa Các Từ Nhúng (Word Embeddings Analogies)](aero_llm_15_embeddings_arithmetic_and_analogies.md) | [Xem bài viết →](aero_llm_15_embeddings_arithmetic_and_analogies.md) |
| [Vỡ Mộng Về Số Học Vector Tương Đương (Soft-Coded Analogies) Trên Word2Vec](aero_llm_16_codechallenge_soft_coded_analogies_in_word2vec.md) | [Xem bài viết →](aero_llm_16_codechallenge_soft_coded_analogies_in_word2vec.md) |
| [Thiết Lập Và Diễn Giải Trục Ngữ Nghĩa Tuyến Tính (Linear Semantic Axes)](aero_llm_17_creating_and_interpreting_linear_semantic_axes.md) | [Xem bài viết →](aero_llm_17_creating_and_interpreting_linear_semantic_axes.md) |
| [Khai Thác Thuật Toán k-NN Cho Tìm Kiếm Từ Đồng Nghĩa Trên BERT](aero_llm_18_knn_for_synonym_searching_in_bert.md) | [Xem bài viết →](aero_llm_18_knn_for_synonym_searching_in_bert.md) |
| [Cạnh Tranh Tìm Từ Đồng Nghĩa BERT vs GPT: Cơ Chế Tokenization Đa Ký Tự](aero_llm_19_codechallenge_bert_v_gpt_knn_kompetition.md) | [Xem bài viết →](aero_llm_19_codechallenge_bert_v_gpt_knn_kompetition.md) |
| [Sự Dịch Chuyển Và Đồng Tồn Biểu Diễn Giữa Các Không Gian Nhúng](aero_llm_20_research_on_translating_embeddings_spaces.md) | [Xem bài viết →](aero_llm_20_research_on_translating_embeddings_spaces.md) |
| [Phân Tích Chùm Quang Phổ Suy Biến (Singular Value Spectrum) Của Không Gian Nhúng](aero_llm_21_singular_value_spectrum_of_embeddings_submatrices.md) | [Xem bài viết →](aero_llm_21_singular_value_spectrum_of_embeddings_submatrices.md) |
| 📌 **[Ánh Xạ SVD Các Dải Điểm Nhúng Có Quan Hệ Chéo](aero_llm_22_codechallenge_svd_projections_of_related_embeddings.md)** | [Xem bài viết →](aero_llm_22_codechallenge_svd_projections_of_related_embeddings.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
