
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
# Nghệ Thuật Vẽ Bản Đồ Nhiệt Ma Trận Nhúng Bằng Cường Độ Từ (Coloring Cosine Similarity)

## Tóm tắt

Trí não con người là một cỗ máy nhận diện phổ thị giác (Visual pattern recognition), nó bế tắc hoàn toàn trước các bức tường ma trận số nguyên. Đóng vai trò làm cầu nối giữa hệ thống kỹ thuật tuyến tính và cảm thụ sinh lý của kỹ sư học máy, bài mô phỏng này dùng phương pháp Min-Max Scaling của Cường Độ Vector và Góc Tọa Độ để bóc tách một dạng Bản Đồ Nhiệt Văn Bản (Heatmap Overlays Text) trực tiếp trên các đoạn tài liệu Wikipedia (VD: Georgia/Algae Fuel/Purple). Kỹ thuật này giúp phát quang được sự lười biếng phân loại của hệ thống Tokenizer LLMs.

---

## 1. Công Cụ Khuếch Đại Khoảng Cách Mạng Bằng Độ Lớn Vector Hình Học

Khác với khoảng cách hai chiều, Độ Lớn Kích Thước (Vector Magnitude / L2-Norm) của một Embeddings vector (Khoảng cách điểm đó tính từ lõi $0$ của Không gian học) được tính bằng hàm Sum of Squares:

$$

$$

\|v\| = \sqrt{\sum_{i=1}^{D} v_i^2}

$$

$$

Với BERT, sự biến vi mô phân tử chỉ nằm tản mác từ dải $[0.8, 1.6]$.

Để dùng thước đo này gán vào thang Gradients Màu RGB (Heatmap Red color map), ta phải nén ép khoảng biến thiên dị biệt trên bằng hàm Cân Kế Tuyến Tính:

$$

$$

\text{Scaled } \|v\| = \frac{\|v\| - \text{Min}}{\text{Max} - \text{Min}}

$$

$$

Kỹ thuật này bảo lưu trọn vẹn điểm đồ thị tỉ lệ (Dữ liệu Scale tịnh tiến), nhưng đóng khung kết quả cứng vào $[0.0, 1.0]$. 
Khi nhuộm sắc lên văn bản, kết quả thị giác hóa mang lại điều kinh ngạc:
- **Pale Trắng Bạc (Min Length):** Toàn bộ giới từ ngữ pháp, dấu câu yếu như: *of, it's, comma (,), period (.), a, the, because, at*. Chúng chỉ nằm cách lõi 0 một quãng ngắn (chìm dưới đáy xã hội học sâu).
- **Red Sẫm Máu (Max Length):** Các từ ngữ mang tính khái niệm độc bảng dày đặc: *neoclassical, crossroads, contention, nouveau, various*. 

Các hạt từ vựng mang đặc tính tần suất học thấp (Rare vocab / High specialized), xuất hiện lẻ tẻ trên tập đào tạo bị mạng Lõi hệ thần kinh phóng đẩy văng mạnh thành những "tọa độ trôi dạt" ra xa Origin. 

---

## 2. Truy Vết Cosine Gốc Trực Tiếp Lên Phổ Vệ Tinh Trực Quan

Ứng dụng bản đồ nhiệt thứ hai được tiến hành qua cơ chế Bóc tách Cosine chuỗi: Nhuộm nền một Tokenized Document theo cường độ Cosine Similarity so với từ liền trước nó (Ngoại trừ phần tử thứ 0 trả kết quả `NaN`, buộc phải dùng hàm `np.nanmin` để triệt tiêu lỗi sập thuật toán Zero-division).

Với thuật toán gán Color Overlay lên đoạn văn *Algae fuel*, những từ ngữ bị chìm đỏ gắt bộc lộ ra các bộ đôi bài trùng cố hữu trong ngôn ngữ người như:
- `practical` + `significance` 
- `algae` + `fuel` 
- `fossil` + `fuels`

Máy học không hiểu sinh học hữu cơ, nó chỉ là một con chíp lặp chuỗi thống kê khi thấy Algae & Fuel cọ sát nhau lặp đi lặp lại tạo thành một vệt dính kết không gian. 

---

## 3. Khóa Target Tìm Điểm Gây Mòn Sự Tính Toán

Ngoài dạng tìm đồng bộ tiếp nối, Bản Đồ Nhiệt có khả năng ghim chết (Pinning) mục tiêu thành tâm đối tượng. Trong tài liệu Wikipedia nói về MÀU TÍM, chúng ta khóa Token `purple` làm tâm ($V_{\text{target}}$) .
Lệnh quét chổi tạo Heatmap quét toàn bộ đại lục văn bản, tất cả các từ trong văn bản đều bị làm Scale Cosine đối chiếu tới duy nhất tâm `purple`.
- Lúc này cường độ đỏ dâng lên ở các cụm từ liên kết địa đồ với sắc thái tím.
- Các vệ tinh mang tên `purple` nếu xuất hiện lặp lại trong văn bản, thuật toán ép điểm Normalize Cosine max $= 1.0 \to \text{Red}_{100\%}$. Hệ thống nhận dạng đây đích xác là hiện tượng Gương phản chiếu tự thân trong mạng Vector Space (Autocorrelation).

Phép toán màu hóa không dùng để vẽ đồ án mỹ thuật, mà trang bị cho các kỹ sư Explainable AI (XAI) khả năng đọc lướt nhanh cơ chế tập trung ngầm của Attention, phơi bày ra cách trí thông minh sinh học được định dạng lại dưới lớp mặt nạ Tensor thần kinh.

---

## Tài liệu tham khảo

1. **Karpathy, A., et al. (2015).** *Visualizing and Understanding Recurrent Networks.* ICLR (Phương trình đánh giá lớp vỏ nhiệt XAI).
2. Tài liệu thực hành lập trình số liệu XAI - *Coloring cosine similarity visualization.*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [aero llm 01 codechallenge cosine similarity advanced part 1](aero_llm_01_codechallenge_cosine_similarity_advanced_part_1_.md) | [Xem bài viết →](aero_llm_01_codechallenge_cosine_similarity_advanced_part_1_.md) |
| [aero llm 02 codechallenge cosine similarity advanced part 2](aero_llm_02_codechallenge_cosine_similarity_advanced_part_2_.md) | [Xem bài viết →](aero_llm_02_codechallenge_cosine_similarity_advanced_part_2_.md) |
| [Theo Dõi Dòng Chảy Cosine Similarity Trên Trục Văn Bản Chuyên Tuần Tự (Word Sequences)](aero_llm_03_codechallenge_cosine_similarity_in_word_sequences.md) | [Xem bài viết →](aero_llm_03_codechallenge_cosine_similarity_in_word_sequences.md) |
| 📌 **[Nghệ Thuật Vẽ Bản Đồ Nhiệt Ma Trận Nhúng Bằng Cường Độ Từ (Coloring Cosine Similarity)](aero_llm_04_codechallenge_coloring_cosine_similarity.md)** | [Xem bài viết →](aero_llm_04_codechallenge_coloring_cosine_similarity.md) |
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
