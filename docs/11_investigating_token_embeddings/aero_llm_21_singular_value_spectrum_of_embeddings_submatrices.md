
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
# Phân Tích Chùm Quang Phổ Suy Biến (Singular Value Spectrum) Của Không Gian Nhúng

## Tóm tắt

Singular Value Decomposition (SVD - Phân rã giá trị suy biến) là một trong những công cụ toán học tối cao trong học máy (Machine Learning) nhằm thực hiện kỹ thuật giảm chiều (Dimensionality reduction). Trong cấu trúc của Mạng Nơ-ron lớn và Mechanistic Interpretability, biểu đồ Quang phổ suy  biến (Singular value spectrum / Scree plot) dùng để chẩn đoán lượng trật tự tuyến tính định hướng ẩn trong đám mây vector từ vựng. Bài học này thảo luận cách thiết lập, tính toán toán học phương sai bằng SVD lên những ma trận nhúng từ (Embedding Matrix).

---

## 1. Cơ Sở Của Phân Rã Giá Trị Suy Biến (SVD)

Mô hình SVD phát biểu rằng bất kỳ ma trận chữ nhật nào cũng có thể được phân giải một cách hoàn chỉnh (Decomposition) vào một tổ hợp của ba ma trận đặc thù. Giả sử tập hợp embeddings của một cụm $N$ tokens ngôn ngữ tạo nên một ma trận hỗn hợp $E \in \mathbb{R}^{N \times D}$. Ma trận này được bóc tách:

$$
E = U \Sigma V^T
$$

### Cấu Trúc Ba Ma Trận
- **$U$ (Orthogonal row matrix - Dữ liệu hướng token):** Cung cấp các vector nền tảng trực giao trong không gian $N$, điều hướng dòng chú ý hàng tự do. Ở bài toán tìm trục không gian, $U$ không phải là đối tượng nghiên cứu.
- **$\Sigma$ (Diagonal matrix - Ma trận Giá trị Suy biến):** Là một ma trận đường chéo $\Sigma \in \mathbb{R}^{N \times D}$ mà các giá trị trên đường chéo $\sigma_i$ (Singular values) được gọi tắt là phổ tín hiệu, sắp xếp giảm dần $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k$. Những giá trị $\sigma$ mang năng lượng cao chứa phần lớn đại lượng phương sai mô hình.
- **$V^T$ (Tập Trục Cơ Sở Chuyển vị - Basis Vector Space):** Là ma trận chứa các vector độc lập tuyến tính, mỗi hàng là một chiều biến cố trực giao ở vùng nhúng chiều (embeddings dimensions) gốc của ma trận. Chúng cung cấp véc-tơ nền tối ưu (Optimal Basis Vectors).

---

## 2. Diễn Giải Phổ Tín Hiệu: Quang Phổ Suy Biến (Scree Plot)

Phổ giá trị suy biến có thể được đồ thị hóa thông qua đường biểu diễn $\sigma_i$ theo bậc hạng số.

### 2.1 Ma Trận Cấu Trúc Tuyến Tính Sắc Nét (Structured Data)
Nếu có một sự tương quan khăng khít theo xu hướng logic (Ví dụ: Một ma trận 20 tokens với toàn các từ vựng chỉ Phương tiện Giao thông), biểu đồ Scree plot sẽ có một hoặc hai hạt lõi (Component) $\sigma_1, \sigma_2$ cắm mốc rất cao và phần còn lại trượt sụt đổ dốc như sạt lở (Scree falling cliff).
Hiện tượng này báo hiện một đại lộ hướng phương sai chính cực lớn: "Đám mây từ vựng" không vô hướng mà đang bị căng giãn mạnh mẽ theo chỉ dẫn của **một trục ý nghĩa cốt lõi** (Semantic direction). 

### 2.2 Đám Mây Đẳng Hướng Nhiễu Loạn (Isotropic Cloud)
Ngược lại với ma trận chứa các tokens rời rạc (Ví dụ tập token chữ cái A-Z và các con số hỗn loạn), chùm quang phổ từ $\Sigma$ sẽ chạy trượt băng thoai thoải, không có sự cắt đứt giữa mỏ neo $\sigma_1$ với bầy $\sigma_{>1}$. Đó là dấu hiệu hình học biểu hiện rằng lượng dữ liệu đang trôi nổi trong không gian vi phân đa chiều với một đám mây khối cầu vô định (isotropic cloud scatter point). Thuật toán không định tuyến được một rãnh chiều sâu logic đủ tin cậy.

---

## 3. Quá Trình Mean-Centering Và Hiệu Sinh Zero-Rank

Trước khi đẩy khối embeddings $E$ vào buồng SVD, dữ liệu bắt buộc cần phải được trừ đi trung bình chung (mean-centered cross dimensions) để các đỉnh vector bắt rễ quanh tọa độ $0$:

$$
\bar{E} = E - \mu_E
$$

Do hệ quả của phép dịch tâm học máy tuyến tính, Rank (hạng) của ma trận sẽ giảm đi 1 bậc, dẫn đến điểm phần tử trị số suy biến cuối cùng của mảng phổ luôn luôn trượt bằng 0 ($\sigma_N = 0$).

---

## 4. Kết luận

Phân rã SVD không chỉ cho phép nén giảm ma trận ở hàng triệu tham số. Nó là một chiếc "X-Quang" chiếu soi lớp cơ bắp phương sai cho bộ não Mạng Nơ-ron. Một SVD quang phổ rơi tự do là hy vọng để giới nghiên cứu nắm bắt các sợi chỉ điều phối các chiều văn bản khổng lồ, là tiền đề cấu trúc hóa phép chiếu Principal Component Analysis (PCA) trên hệ hành vi học phân lớp của LLMs.

---

## Tài liệu tham khảo

1. **Golub, G. H., & Van Loan, C. F. (2013).** *Matrix Computations.* JHU Press. (Lý thuyết lõi về SVD).
2. **Deerwester, S., et al. (1990).** *Indexing by latent semantic analysis.* JASIS. (Thuật toán LSA cho NLP phân rã dựa trên SVD).
3. **Coenen, A., et al. (2019).** *Visualizing and measuring the geometry of BERT.* NeurIPS.
4. Tài liệu đào tạo bài giảng *Investigating token embeddings - SVD submatrices spectrum.*
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
| 📌 **[Phân Tích Chùm Quang Phổ Suy Biến (Singular Value Spectrum) Của Không Gian Nhúng](aero_llm_21_singular_value_spectrum_of_embeddings_submatrices.md)** | [Xem bài viết →](aero_llm_21_singular_value_spectrum_of_embeddings_submatrices.md) |
| [Ánh Xạ SVD Các Dải Điểm Nhúng Có Quan Hệ Chéo](aero_llm_22_codechallenge_svd_projections_of_related_embeddings.md) | [Xem bài viết →](aero_llm_22_codechallenge_svd_projections_of_related_embeddings.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
