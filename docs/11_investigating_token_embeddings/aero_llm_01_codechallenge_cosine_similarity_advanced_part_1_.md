
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
Phân tích Cosine Similarity nâng cao trong không gian embedding

Cơ sở toán học, hình học vector và ứng dụng trong mô hình ngôn ngữ lớn

⸻

Tóm tắt

Cosine Similarity là một trong những thước đo cốt lõi trong xử lý ngôn ngữ tự nhiên (NLP), đặc biệt khi làm việc với vector embedding có chiều cao. Bài viết này trình bày nền tảng toán học của Cosine Similarity, mở rộng sang các phân tích hình học trong không gian Hilbert, mối liên hệ với chuẩn hóa vector, phân phối xác suất trong embedding space, và ứng dụng trong retrieval, semantic search và đánh giá mô hình ngôn ngữ lớn (LLMs). Ngoài ra, bài viết bổ sung các công thức minh họa và liên hệ với lý thuyết thông tin.

⸻

1. Giới thiệu

$$
Trong NLP hiện đại, văn bản được ánh xạ sang vector trong không gian \mathbb{R}^d thông qua embedding models. Các tổ chức như:
$$

	•	OpenAI
	•	Google Research
	•	Meta AI

đã phát triển các hệ embedding cho:
	•	Semantic search
	•	Retrieval-augmented generation (RAG)
	•	Clustering
	•	Similarity detection

Trong các hệ này, Cosine Similarity là thước đo chuẩn để so sánh hai vector.

⸻

2. Định nghĩa Cosine Similarity

$$
Cho hai vector \mathbf{x}, \mathbf{y} \in \mathbb{R}^d:
$$

$$
\text{cosine\_sim}\mathbf{x}, \mathbf{y} =
$$

\frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}

Trong đó:
	•	Tích vô hướng:

$$

$$

\mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^{d} x_i y_i

$$

$$

	•	Chuẩn Euclid:

$$

$$

\|\mathbf{x}\| = \sqrt{\sum_{i=1}^{d} x_i^2}

$$

$$

⸻

3. Diễn giải hình học

Cosine similarity đo cos của góc giữa hai vector:

$$
\cos \theta = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}
$$

Giá trị:
	•	1 → cùng hướng
	•	0 → trực giao
	•	-1 → ngược hướng

Trong embedding NLP, vector thường được chuẩn hóa:

$$
\tilde{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|}
$$

Khi đó:

$$
\text{cosine\_sim}\mathbf{x}, \mathbf{y} =
$$

\tilde{\mathbf{x}} \cdot \tilde{\mathbf{y}}

⸻

4. Không gian chiều cao và hiện tượng tập trung

Trong không gian chiều cao d \gg 1:
	•	Các vector ngẫu nhiên có xu hướng gần trực giao
	•	Góc giữa hai vector ngẫu nhiên tiệm cận 90^\circ

Theo lý thuyết xác suất:

$$
Nếu x_i, y_i \sim \mathcal{N}(0,1)
$$

$$
\mathbb{E}[\mathbf{x} \cdot \mathbf{y}] = 0
$$

$$
Var\mathbf{x} \cdot \mathbf{y} = d
$$

Sau chuẩn hóa:

$$
\mathbb{E}[\cos \theta] \approx 0
$$

Hiện tượng này gọi là concentration of measure.

⸻

5. Quan hệ với khoảng cách Euclid

Khoảng cách Euclid:

$$
\|\mathbf{x} - \mathbf{y}\|^2 =
$$

\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}\cdot\mathbf{y}

Nếu chuẩn hóa:

$$
\|\tilde{\mathbf{x}} - \tilde{\mathbf{y}}\|^2 =
$$

2 - 2\cos \theta

Do đó:

$$
\cos \theta = 1 - \frac{1}{2}\|\tilde{\mathbf{x}} - \tilde{\mathbf{y}}\|^2
$$

→ Cosine similarity tương đương với Euclidean distance trong không gian chuẩn hóa.

⸻

6. Cosine Similarity trong embedding xác suất

Một embedding model ánh xạ văn bản t thành vector:

$$
f_\thetat \in \mathbb{R}^d
$$

Xác suất chọn tài liệu $d_i$ trong retrieval:

$P($d_i$\mid q)$ =
\frac{\exp$\alpha \cdot \cos(f(q$, f$d_i$))}

$$
{\sum_j \exp\alpha \cdot \cos(f(q, fd_j))}
$$

Trong đó:
	•	\alpha là temperature scaling

⸻

7. Liên hệ với Information Theory

Theo Elements of Information Theory:

Mutual information giữa hai vector embedding:

$$
I(X;Y) =
$$

$$
\mathbb{E}\left[
$$

$\log$ \frac{P(X,Y)}{$P(X)$$P(Y)$}
\right]

Cosine similarity có thể xem như xấp xỉ thô của sự phụ thuộc tuyến tính giữa hai biến.

⸻

8. Cosine Similarity và Loss Function

Trong contrastive learning (ví dụ SimCLR):

$$
\mathcal{L} =
$$

- $\log$
\frac{\exp$\cos(\mathbf{x}_i,\mathbf{x}_j$/\tau)}

$$
{\sum_k \exp\cos(\mathbf{x}_i,\mathbf{x}_k/\tau)}
$$

Trong đó:
	•	\tau là temperature
	•	$\mathbf{x}_i,\mathbf{x}_j$ là positive pair

⸻

9. Phân tích gradient

Giả sử:

$$
S = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}
$$

Gradient theo \mathbf{x}:

$$

$$

\frac{\partial S}{\partial \mathbf{x}} =

$$

$$

\frac{\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}
-
\frac{$\mathbf{x}\cdot\mathbf{y}$\mathbf{x}}
{\|\mathbf{x}\|^3\|\mathbf{y}\|}

Điều này cho thấy quá trình tối ưu sẽ:
	•	Kéo vector cùng hướng lại gần
	•	Đẩy vector khác hướng ra xa

⸻

10. Ứng dụng trong LLM

Các ứng dụng thực tế:
	•	Semantic Search
	•	Retrieval-Augmented Generation
	•	Clustering câu hỏi
	•	Detect duplicate content

Các tổ chức như Stanford University và MIT đã sử dụng cosine similarity trong các hệ thống IR và NLP hiện đại.

⸻

11. Hạn chế
	1.	Không nhạy với độ lớn vector
	2.	Không nắm bắt quan hệ phi tuyến
	3.	Bị ảnh hưởng bởi anisotropy trong embedding space

Một số nghiên cứu đề xuất:
	•	Whitening transformation
	•	Centering embeddings
	•	Angular margin loss

⸻

12. Kết luận

Cosine Similarity là thước đo hình học cơ bản nhưng cực kỳ hiệu quả trong NLP hiện đại. Trong không gian embedding chiều cao, nó:
	•	Ổn định
	•	Dễ tính toán
	•	Phù hợp cho retrieval

Tuy nhiên, cần kết hợp với chuẩn hóa và kỹ thuật regularization để đạt hiệu năng tối ưu.

⸻

Tài liệu tham khảo
	1.	Cover & Thomas (2006). Elements of Information Theory.
	2.	Bishop (2006). Pattern Recognition and Machine Learning.
	3.	Chen et al. (2020). SimCLR: A Simple Framework for Contrastive Learning.
	4.	Mikolov et al. (2013). Word2Vec.
	5.	Reimers & Gurevych (2019). Sentence-BERT.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[aero llm 01 codechallenge cosine similarity advanced part 1](aero_llm_01_codechallenge_cosine_similarity_advanced_part_1_.md)** | [Xem bài viết →](aero_llm_01_codechallenge_cosine_similarity_advanced_part_1_.md) |
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
| [Ánh Xạ SVD Các Dải Điểm Nhúng Có Quan Hệ Chéo](aero_llm_22_codechallenge_svd_projections_of_related_embeddings.md) | [Xem bài viết →](aero_llm_22_codechallenge_svd_projections_of_related_embeddings.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
