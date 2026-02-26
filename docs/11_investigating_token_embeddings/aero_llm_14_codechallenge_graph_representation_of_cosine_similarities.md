
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
# Bố Cục Đồ Thị Mạng (Network Graph) Thông Qua Ma Trận Cosine Similarity

## Tóm tắt

Phân tích Mạng Lưới Điểm (Graph Network Analysis) là nền móng của Khoa học dữ liệu nhằm tìm ra các chuỗi liên kết cộng sinh trong cụm mô hình văn bản (Clustering Tokens). Thay vì sử dụng một bảng Pixel vuông (Heatmap Matrix) rất phổ biến nhưng thiếu chiều sâu thị giác, bài báo khoa học này thiết lập một thuật toán ánh xạ các điểm token thành mạng lưới vũ trụ ly tâm hình tròn. Thông qua mặt nạ phân cực (Binary mask thresholds), chúng ta có thể trực quan hệ mô hình sự cộng hưởng tính chất ngữ nghĩa giữa nhiều điểm vector nhúng.

---

## 1. Cơ Sở Thiết Lập Mặt Nạ Lân Cận (Spatial Thresholding Mask)

Để thiết lập cấu trúc cạnh liên kết (Edge) giữa $N$ phân tử (Nodes - Tokens), chúng ta cần khởi tạo Ma trận khoảng cách Tương quan góc Cosine $N \times N$, biểu thị độ trùng lặp đặc trưng góc của từng bộ vector:
$$
S(i,j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}
$$

**Tính Ngưỡng Chặn Dòng (Cut-off Threshold):** 
Trong Mạng nơ-ron, sự tương quan của $S$ luôn dày đặc ở mức $\sim 0.2$, sinh ra vô vàn rác kết nối nhiễu. Ta cần thanh tẩy đồ thị bằng việc tính toán Ngưỡng Độ bão hòa dựa trên hàm Phương sai Bán chuẩn (Median + 1 Standard Deviation) chuyên bắt tín hiệu bất thường cường độ cao:
$$
\text{Threshold } (T) = \text{Median}(S_{\text{upper-triangular}}) + \sigma(S_{\text{upper-triangular}}) 
$$
Tất cả những điểm $S(i, j) < T$ hoàn toàn bị thay thế bằng mặt nạ nhị phân câm (Binary mask $= 0$). Chỉ những kết nối siêu cường ($S(i, j) \geq T$) được lộ diện trong mạng lý tưởng, chuyển hóa tập hợp vector rối rắm (Dense matrix) thành Cấu trúc thưa thớt logic (Sparse matrix). Đừng quên ép hàm đường chéo chính (Diagonal tự tương quan) bằng 0.

---

## 2. Tính Toán Không Gian Vệ Tinh Tròn (Circular Graph Coordinates)

Thay vì rải loạn ngẫu nhiên x-y, sơ đồ Tròn (Ring Plot) được chọn để san bằng tính phân cấp thứ bậc, đồng dạng mọi token cách đều tâm.

### 2.1 Ma Trận Vector Phân Cực:
Giả thiết số lượng $N$ tokens sẽ được phân chia đều nhau đính trên một bán kính $R=1$, chúng ta sử dụng hệ tọa độ cực để tìm góc pha $d\theta$ và tọa độ $\theta$ mỗi góc chèn:
$$
\Delta \theta = \frac{2\pi}{N} 
$$
Dải Vector Pha Góc (Phase Angles): $\theta \in \left[ 0, ~ 2\pi - \Delta \theta \right]$. *Tại sao lại kết thúc ở $2\pi - \Delta \theta$? Vì kết thúc đúng tại $2\pi$ tương ứng góc $360^\circ$ sẽ gây ra sự tự chèn lớp đè lên điểm đếm gốc số $0$.*

Từ đó, hoành độ vi phân hiển thị ra tọa độ 2D của mỗi Token Node:
$$
x_i = \cos(\theta_i) 
$$
$$
y_i = \sin(\theta_i)
$$

### 2.2 Quy Hoạch Bậc Kết Nối (Degree Size Scaling):
Trong Graph Theory, "Sức hút" của một đỉnh vòng (Node Size) được tính bằng Bậc (Degree) - Tức là số lượng cạnh liên đới dính vào nó. Ở bài toán này, Đám mây cỡ hạt được quy định thông qua việc Đếm tần số vượt ngưỡng $T$ (Suprathreshold counts) của một Vector hàng:

$$
\text{DotSize}_i \propto 3 \times \sqrt{\sum_{j=1}^{N} \mathbb{I}(S(i, j) \geq T)} 
$$

*(Chuyển biến tỷ lệ thu phóng căn bậc hai giúp phân tán hình ảnh hài hòa và êm mắt).*

### 2.3 Đường Mạch (Color Mapping Edges):
Những sợi chỉ đường ranh giới thẳng đứng sẽ nối tọa độ $(x_i, y_i)$ và $(x_j, y_j)$ với tham chiếu màu thay đổi trượt theo hệ thang nóng (Plasma colormap). Đường mạch màu tím có nghĩa Cosine dư ở mức thấp, đường màu vàng nóng thể hiện những dòng xoáy điểm tựa ngữ nghĩa mãnh liệt móc xích từ vựng lại với nhau.

---

## 3. Ứng Dụng Xuyên Mạng Graph

Đồ thị Cosine không chỉ đơn thuần là bộ màu lòe loẹt. Khi thả vào văn bản chứa kiến thức hạt nhân (Physics, Networking), sơ đồ vệ tinh sẽ rẽ nhánh các cộng đồng (Communities Detection). Tính hiệu lực sinh học tập trung vào sự trồi lên của một lượng ít Nút vệ tinh siêu đại diện (Hub hubs) với hệ mạng chằng chịt, kéo theo các Nút vệ tinh vệ quốc (vệ tinh nhược kết nối) quay quần xung quanh, minh họa sự đa pha phân mảng trong cơ học diễn giải (Mechanistic Interpretability).

---

## Tài liệu tham khảo

1. **Newman, M. E. J. (2003).** *The structure and function of complex networks.* SIAM Review.
2. **Bastian, M., et al. (2009).** *Gephi: An open source software for exploring and manipulating networks.* ICWSM.
3. Tài liệu đào tạo bài giảng *Investigating token embeddings - Graph representation of cosine similarities.*
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
| 📌 **[Bố Cục Đồ Thị Mạng (Network Graph) Thông Qua Ma Trận Cosine Similarity](aero_llm_14_codechallenge_graph_representation_of_cosine_similarities.md)** | [Xem bài viết →](aero_llm_14_codechallenge_graph_representation_of_cosine_similarities.md) |
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
