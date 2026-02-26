
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
# Phương Pháp T-SNE Và Thuật Toán Phân Cụm DBSCAN: Chiếu Không Gian Đa Chiều Cho LLMs

## Tóm tắt

Phân tích hình học vi mô trên các cụm không gian vector ngôn ngữ đang là mấu chốt của Machine Learning. Mọi nỗ lực tìm kiếm quy luật, khử nhiễu mã thông tin bên trong các LLMs đều vấp phải sự hạn chế quan sát đa chiều (>1000 chiều) của loài người. Bài báo này phân tích cơ sở toán vật lý cho hai kỹ thuật trụ cột: Làm phẳng không gian với thuật toán xác suất **t-SNE** (T-distributed Stochastic Neighbor Embedding) và Cắt lớp tổ hợp dữ liệu với phân cụm mật độ **DBSCAN** (Density-Based Spatial Clustering). Sự kết hợp này đưa ánh sáng đến cấu trúc "hộp đen" của Embeddings.

---

## 1. T-SNE: Nghệ Thuật Ép Không Gian Dựa Trên Xác Suất

Kỹ thuật t-SNE, được nghiên cứu và tiên phong bởi Geoffrey Hinton cùng cộng sự, chuyển đổi bài toán khoảng cách (Euclidean distance) thành bài toán tối ưu phân phối xác suất. Nếu hai vector nằm gần nhau theo luật hình học (nearest neighbors) tại gốc 1000 chiều đa ma trận, thì qua t-SNE, xác suất để chúng tiếp tục chạm nhau trên sàn 2 chiều (hoặc 3 chiều) là rất cao.

### 1.1 Tính Toán Phân Phối Ở Không Gian Điểm Ảnh Gốc
Đầu tiên, quy chuyển chuẩn hàm Softmax lên ma trận Euclidean. Tại lớp không gian bậc cao $X$, khả năng để vector $x_j$ nằm kề $x_i$ được biểu diễn bởi mật độ xác suất hàm mũ (Gaussian Gaussian Distribution):

$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
$$

Trong đó, $\sigma_i$ là phương sai (variance) chịu ảnh hưởng cấu hình phân tán kề lặp (Perplexity).

### 1.2 Chiếu Lên Chuẩn Bậc Thấp Và Tối Ưu Bằng Divergence
Hệ thống giả lập tiếp tục một chiều thấp $Y$ với cấu trúc Student t-Distribution nặng đuôi để ngăn cản hiện tượng đám đông nhồi nhét cực điểm (Crowding problem). Và mục đích vĩ đại của T-SNE là tinh chỉnh sao cho đồ thị phân phối khoảng cách cấu hình tại khối nhãn $Y$ mô phỏng chân xác nhất khối điểm $X$. Máy giải đạo hàm (Cost function gradient descent) thông qua việc kéo Min cho hàm chênh lệch **Kullback-Leibler (KL) Divergence**:

$$
C = \sum_{i} KL(P_i \parallel Q_i) = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

Sự trượt biến của Loss này khẳng định $Y$ đã tạo ra bóng ma 2 chiều sinh động của Mạng Nơ ron khổng lồ mà không phá hủy các quần tụ tương quan. Tính kết sinh của T-SNE là phi định chuẩn (Probabilistic/Non-deterministic). Mọi lần khởi động đều cho ra bản đồ khác trên nền tương đồng nhãn.

---

## 2. DBSCAN: Phân Lớp Không Gian Liên Kết Mật Độ Lân Cận

Khi t-SNE đã biến đám mây tham số ngẫu nhiên xuống còn mảnh đất phẳng trực quan, sự cần khát đi tìm các gia đình cấu trúc tiếp tục mở ra. DBSCAN (Density-Based Spatial Clustering of Applications with Noise) bỏ rơi tư duy tìm tâm cụm cổ thủ của K-Means, DBSCAN tiến hành gom mạng lân cận mật độ liên hành:

### 2.1 Định Quy Biến Số 
Thuật toán phóng tia quét tìm kiếm quanh các node vector dựa trên hai siêu tham số nền tảng:
- Cự ly biên độ $\epsilon$ (Epsilon distance threshold): Độ dài ngưỡng tia bán kính bao phủ một vùng.
- Ngưỡng giới hạn quân số (MinPts): Số điểm tối thiểu phải lọt vào lưới $\epsilon$ để tạo thành một khối cộng đồng liên đoàn.

### 2.2 Đọc Điểm Gây Loãng (Noise) và Điểm Kết Tinh (Core points)
Mọi quần đảo nối chuỗi lẫn nhau nhờ $\epsilon$ hợp thức hóa thành những nhánh phân chùm hữu cơ vĩ đại. Những Vector lạc loài với khoảng cách xa ngoài chùm $\epsilon$ được thải trừ thành phần bù (Noise points - Những biến dị nhiễu không gây ảnh hưởng đến trung tâm tổ chức cụm biểu diễn). Mức độ khắt khe biến động tỷ lệ thuận cùng sự tăng số MinPts hoặc bóp nghẹt $\epsilon$.

---

## 3. Hình Thành Đồ Thị Tương Quan Ma Trận Gram (Gram Matrix)
Ở lớp phân lớp toán học sâu hơn, cả t-SNE hay phân tập DBSCAN đều giải phẫu thông qua Ma trận Đồ Đồng Cấu Gram (Gram Matrix) của một bộ vi xử lý Vector nhúng:
$$
G_{E} = E \cdot E^T 
$$
Khi các vector được phân bổ đơn vị với lượng Vector-norm chuẫn L2, Gram Matrix lập tức hóa thân thành khối ảnh chiếu Cosine Similarity Matrix. Nó tiết lộ những kiến trúc lưới đồ thị sắc sảo đang giấu nhẻm ở đám mây khối $n$-nghiệm phức loạn. 

---

## 4. Kết luận
Bộ đôi Toán-Xác Suất t-SNE kết hợp DBSCAN cung phụng khả năng thám sát kỳ diệu, biến hệ thập nguyên ngàn chiều của Machine Learning thu gọn vào tầm tay hình học lớp đại cương. Thay vì bóp cong cấu trúc để ép vào chuẩn tâm (Centroids error), phép chiếu mật độ lân cận t-SNE giải trình nguyên vẹn sự kết nối thông qua đạo hàm KL và Epsilon threshold.

---

## Tài liệu tham khảo

1. **Laurens van der Maaten, L., & Hinton, G. (2008).** *Visualizing Data using t-SNE.* Journal of Machine Learning Research.
2. **Ester, M., et al. (1996).** *A density-based algorithm for discovering clusters in large spatial databases with noise (DBSCAN).* KDD.
3. **Schubert, E., et al. (2017).** *DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN.* 
4. Tài liệu bài giảng *Investigating token embeddings - T-SNE and DBSCAN (theory)*.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [aero llm 01 codechallenge cosine similarity advanced part 1](aero_llm_01_codechallenge_cosine_similarity_advanced_part_1_.md) | [Xem bài viết →](aero_llm_01_codechallenge_cosine_similarity_advanced_part_1_.md) |
| [aero llm 02 codechallenge cosine similarity advanced part 2](aero_llm_02_codechallenge_cosine_similarity_advanced_part_2_.md) | [Xem bài viết →](aero_llm_02_codechallenge_cosine_similarity_advanced_part_2_.md) |
| [Theo Dõi Dòng Chảy Cosine Similarity Trên Trục Văn Bản Chuyên Tuần Tự (Word Sequences)](aero_llm_03_codechallenge_cosine_similarity_in_word_sequences.md) | [Xem bài viết →](aero_llm_03_codechallenge_cosine_similarity_in_word_sequences.md) |
| [Nghệ Thuật Vẽ Bản Đồ Nhiệt Ma Trận Nhúng Bằng Cường Độ Từ (Coloring Cosine Similarity)](aero_llm_04_codechallenge_coloring_cosine_similarity.md) | [Xem bài viết →](aero_llm_04_codechallenge_coloring_cosine_similarity.md) |
| [Ảo Ảnh Của Trí Tuệ Toán Học Trong Ngôn Ngữ: Sức Mạnh Của Random Embeddings](aero_llm_05_codechallenge_can_random_embeddings_be_interpreted.md) | [Xem bài viết →](aero_llm_05_codechallenge_can_random_embeddings_be_interpreted.md) |
| 📌 **[Phương Pháp T-SNE Và Thuật Toán Phân Cụm DBSCAN: Chiếu Không Gian Đa Chiều Cho LLMs](aero_llm_06_t_sne_projection_and_dbscan_clustering_theory_.md)** | [Xem bài viết →](aero_llm_06_t_sne_projection_and_dbscan_clustering_theory_.md) |
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
| [Ánh Xạ SVD Các Dải Điểm Nhúng Có Quan Hệ Chéo](aero_llm_22_codechallenge_svd_projections_of_related_embeddings.md) | [Xem bài viết →](aero_llm_22_codechallenge_svd_projections_of_related_embeddings.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
