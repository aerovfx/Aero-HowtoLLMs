
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
# Khai Thác Thuật Toán k-NN Cho Tìm Kiếm Từ Đồng Nghĩa Trên BERT

## Tóm tắt

K-Nearest Neighbors (k-NN - k Láng giềng gần nhất) là một thuật toán cốt lõi trong phân loại cụm học máy cổ điển (Machine Learning Classification). Tuy nhiên, khi kết hợp cùng Vector nhúng (Embeddings) của họ mô hình ngôn ngữ lớn (LLMs) như BERT, thuật toán này thể hiện khả năng tra cứu chéo từ đồng nghĩa (synonym search) ở mức độ đáng kinh ngạc. Bài viết dưới đây trình bày nguyên lý không gian của k-NN, phân biệt hai định chuẩn tính điểm Euclidean và Cosine Similarity, cũng như cách triển khai cho các chiều ẩn trong từ điển số hóa tự nhiên.

---

## 1. Nguyên Lý Của k-Nearest Neighbors (k-NN)

Trong mô hình k-NN, dữ liệu mới không hề được gắn nhãn trước theo một hệ số chặn hàm (Linear threshold). Cách tiếp cận này tuân theo một cơ chế định vị hình học đơn giản: **Dữ liệu thuộc về cộng đồng nào đang áp đảo xung quanh nó.**

1. Đầu vào là một Vector truy vấn vô danh (Unlabeled data point) $x$.
2. Hệ thống quét đếm khoảng cách từ $x$ đến toàn bộ những Vector $\vec{v}_i$ đang mang nhãn dữ liệu có sẵn trong bộ nhớ.
3. Tham số $k$ quy định sẽ lấy $k$ điểm có khoảng cách không gian sát với $x$ nhất (Nearest Neighbors). Lời khuyên là thiết lập $k$ thành số lẻ (VD: $k=3, 5, 7$) để chặn trường hợp hòa/cân bằng.
4. Lựa chọn ưu tiên theo nguyên lý Bầu chọn theo khuynh hướng (Majority Voting): nhãn hiệu chiếm đa số trong $k$ phần tử cận kề sẽ được gán cho $x$.

Trường hợp sử dụng để khám phá từ loại trong BERT, việc dự đoán theo lớp sẽ được thay thế bằng liệt kê $k$-vector gần nhất với từ gốc nhằm tìm ra các từ đồng nghĩa (vd: "Beauty" sẽ gọi ra "Gorgeous", "Elegance"). 

---

## 2. Số Học Khoảng Cách: Euclidean và Cosine Similarity

Không gian tọa độ của mảng Embeddings ma trận BERT sở hữu $D=768$ chiều. Bài toán tìm Láng giềng (Distance calculations) yêu cầu một thước đo chuẩn. Hai thước đo thông dụng đem lại hai góc nhìn dị biệt:

### 2.1. Chuẩn Khoảng Cách Hình Học (Euclidean Distance)
Lấy gốc từ định lý tam giác vuông trong không gian $N$-chiều, Euclidean đo đạc chiều dài thật sự của sợi dây nối giữa mũi tên vector token $\vec{v}$ và token mục tiêu $\vec{w}$:

$$
\delta(\vec{v}, \vec{w}) = \sqrt{\sum_{i=1}^{D} (v_i - w_i)^2}
$$

Chuẩn Euclidean thể hiện tính tách biệt tuyệt đối (absolute spatial magnitude) của thông tin.

### 2.2. Chuẩn Tương Quan Góc (Cosine Similarity)
Trọng tâm đo lường sự đồng dạng không nằm ở lực độ dài, mà vứt bỏ tất cả giới hạn véc-tơ để tìm độ chênh góc giữa hai ngọn vector:

$$
\text{CosineSim}(\vec{v}, \vec{w}) = \frac{\vec{v} \cdot \vec{w}}{\|\vec{v}\| \|\vec{w}\|} \in [-1, 1]
$$

**Sự Lệch Pha Đáng Lưu Ý:** Các vector có chung hướng nội hàm (Cosine Similarity hướng về 1) nhưng hoàn toàn có thể sở hữu Khoảng cách Euclidean kéo dãn ra khổng lồ nếu độ phủ vector (Norm of vector) bị đẩy cực xa gốc tọa độ. Do đó, việc tìm Láng giềng gần nhất k-NN trong cấu trúc BERT đòi hỏi nhà nghiên cứu phải xác định thuộc tính đang săn tìm là khoảng cách hay góc lệch nhạy cảm biểu diễn song song.

---

## 3. Khai Thác Tiền Xử Lý Giảm Chiều Bằng PCA/t-SNE

Khi ứng dụng tệp $k=5$ cho cụm từ "Beauty" xuyên thấu toàn bộ $30.000$ từ điển của BERT, gánh nặng toán học (Tính $30.000$ phép tính hàm mũ $L2-Norm$) có thể sẽ làm đình trệ bộ vi xử lý nếu hệ vector lớn như định dạng GPT hiện đại (với số token trên 1 triệu). 
Theo lý thuyết thông luật của Học Máy (Machine Learning), để tránh "Lời nguyền đa chiều" (Curse of Dimensionality), ma trận nên được phân rã bằng Principal Component Analysis (PCA) triệt tiêu quang phổ yếu (SVD variance noise) sinh ra một ma trận giảm chiều $D = 100$ trước khi hàm k-NN khởi chạy, đảm bảo chi phí thấp mà không đánh tụt độ nhạy tương quan ngữ nghĩa.

---

## 4. Kết luận

Mô hình k-Nearest Neighbors là khối hạt nhân trong mọi bộ truy vấn tìm điểm dữ liệu (Search Engines) ứng dụng vào Mạng Nơ-ron. Việc lạm dụng tính chất khoảng cách ở vùng Embeddings của BERT cho phép k-NN bứt phá khỏi cơ chế nhãn mác nhị phân, trở thành công cụ đắc lực giải phẫu hiện tượng đa nghĩa từ vựng cũng như khai thác vùng giao thoa khái niệm (Concept boundary overlapping).

---

## Tài liệu tham khảo

1. **Cover, T., & Hart, P. (1967).** *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory.
2. **Devlin, J., et al. (2018).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
3. Tài liệu đào tạo *Investigating token embeddings - kNN for synonym-searching in BERT*.
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
| 📌 **[Khai Thác Thuật Toán k-NN Cho Tìm Kiếm Từ Đồng Nghĩa Trên BERT](aero_llm_18_knn_for_synonym_searching_in_bert.md)** | [Xem bài viết →](aero_llm_18_knn_for_synonym_searching_in_bert.md) |
| [Cạnh Tranh Tìm Từ Đồng Nghĩa BERT vs GPT: Cơ Chế Tokenization Đa Ký Tự](aero_llm_19_codechallenge_bert_v_gpt_knn_kompetition.md) | [Xem bài viết →](aero_llm_19_codechallenge_bert_v_gpt_knn_kompetition.md) |
| [Sự Dịch Chuyển Và Đồng Tồn Biểu Diễn Giữa Các Không Gian Nhúng](aero_llm_20_research_on_translating_embeddings_spaces.md) | [Xem bài viết →](aero_llm_20_research_on_translating_embeddings_spaces.md) |
| [Phân Tích Chùm Quang Phổ Suy Biến (Singular Value Spectrum) Của Không Gian Nhúng](aero_llm_21_singular_value_spectrum_of_embeddings_submatrices.md) | [Xem bài viết →](aero_llm_21_singular_value_spectrum_of_embeddings_submatrices.md) |
| [Ánh Xạ SVD Các Dải Điểm Nhúng Có Quan Hệ Chéo](aero_llm_22_codechallenge_svd_projections_of_related_embeddings.md) | [Xem bài viết →](aero_llm_22_codechallenge_svd_projections_of_related_embeddings.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
