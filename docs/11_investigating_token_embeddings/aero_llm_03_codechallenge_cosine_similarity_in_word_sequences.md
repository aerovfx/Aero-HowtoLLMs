
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
# Theo Dõi Dòng Chảy Cosine Similarity Trên Trục Văn Bản Chuyên Tuần Tự (Word Sequences)

## Tóm tắt

Trên thực tế, ngôn ngữ giao tiếp không đơn thuần là những cụm từ đơn độc văng lảng vãng trong không gian Embeddings. Ngôn từ thực thụ chỉ có giá trị khi bị trói buộc vào một "Trị số Không thời gian" - Đó là Trình Tự Chữ Viết (Sequences). Báo cáo thực hành này đào cắt không gian Vector tĩnh của mô hình BERT, áp dụng liên hoàn kỹ thuật Vector hóa độ tương quan Cosine từng bước đệm (Sequential Pairs) thông qua hàm lân cận để giải mã cách bộ máy học tự động bóp méo ý nghĩa theo luồng di chuyển từ vựng.

---

## 1. Cơ Chế Kết Tinh Vector Cosine Nối Tiếp Bề Mặt (Sequential Pairs)
Mô tả cho câu lệnh: 
> *My phone is in the kitchen near the cold ice cream.*

Thuật toán không chạy điểm quy nạp tương tự cho toàn câu, mà nó cắt nhỏ từng chặng $t_i$:
$C(t_i, t_{i-1}) = \cos(\vec{v}_i, \vec{v}_{i-1}) = \frac{\vec{v}_i \cdot \vec{v}_{i-1}}{\\mid \vec{v}_i\\mid \\mid\vec{v}_{i-1}\\mid}$

Khi đặt lên thanh đồ thị Bar plot:
- Lực hút giữa `cold` và `ice` đẩy Cosine vọt lên ngưỡng $\sim 0.6$ (Mối quan hệ nhiệt đại đa cấu trúc).
- Lực hút giữa `ice` và `cream` duy trì $\sim 0.5$ (Cấu trúc danh từ ghép truyền thống).
- Nhưng lực hút giữa `phone` và `is` sụp đổ xuống mức $\sim 0.15$. Mạng lý lẽ của BERT đã học từ hàng triệu trang sách rằng `is` là động từ to be liên kết ngẫu nhiên với vạn vật. `Phone` chả có thuộc tính gì sinh ra lực hấp dẫn với `is`.

Do đó, đồ thị Sequential Cosine này chính là Biểu Đồ Điện Não Đồ (EEG) cho thấy mức độ gắn kết logic liền kề (Logical transition density) của từng chuỗi tư duy.

---

## 2. Đo Khảo Phân Nhánh Nghĩa Bằng Đường Tiệm Cận Biến Đổi (Diverging Sequences)

Lý do vì sao Cosine Cục bộ quan trọng được chứng minh qua hai câu Garden-Path:
A: *The conductor waved his hands as the train departed.*
B: *The conductor waved his hands as the orchestra began.*

Tại thời điểm bộ Tokenizer đi từ đầu đến chữ `The conductor waved his hands as...`: Trí tuệ của BERT lẫn não sinh học chúng ta chưa phân tích được từ "Conductor" này là "Người soát vé tàu tủy" hay "Nhạc trưởng giao hưởng" (Tính mập mờ ý niệm đa nghĩa Polysense). 

Toàn bộ biểu đồ đồ thị Cosine của hai câu văn đè lên nhau trùng khớp đến $\mathbf{100\%}$. Chỉ đến khi đâm sầm vào 2 Tokens biến hóa cuối cùng (`train departed` và `orchestra began`), biểu đồ mới rẽ nhánh đồ thị (Forking transition):
- Tại điểm rẽ $\to$ `train` với độ dốc Cosine cao hơn, kéo ngược tâm nhúng của mạng nội hàm lên một miền vận chuyển giao thông.
- Tại điểm rẽ $\to$ `orchestra`, một hàm phân bổ Vector khác được bẻ gãy kích hoạt. 
Đó chính là lúc sự tái định nghĩa được kiến tạo.

---

## 3. Bản Chất Của Tính Mập Mờ Giải Phẫu

Điều tiết lộ chua xót nhất từ thực nghiệm trên: Các ma trận tĩnh Embedded Matrix thuần túy (như BERT raw vector) **hoàn toàn câm điếc trong việc hiểu văn cảnh ngược**.
- Dù chữ `conductor` sau này đã được làm rạng tỏ là *Nhạc trưởng*. Thế nhưng, tọa độ điểm $\vec{v}_{\text{conductor}}$ khi rút thẳng từ Vocabulary Embeddings $E$, rồi được đối chiếu với $\vec{v}_{\text{waved}}$ là hoàn toàn tĩnh tại. Thống kê khoảng cách sẽ bị cứng ngắc (Frozen logic).

Tuy nhiên mạng ngôn ngữ học sâu BERT lại không chết bởi nguyên lý đó vì Embedded matrix này mới chỉ là "Tầng Trệt". Khi các giá trị này mớm dần qua nhiều Trụ Cột Attention Layers, một cơ chế truy ngược thời gian ngầm định (Backward context flow) sẽ ép cập nhật lại định dạng véc-tơ của từ `conductor` bằng cơ chế Self-attention có trọng số (Weighted dot matrix). Phân tích chuỗi tuần tự chính là tiền đề căn bản nhất để ta mở đường lên phân tích Context Vectors sau này.

---

## Tài liệu tham khảo

1. **Vaswani, A., et al. (2017).** *Attention is all you need.* NIPS. (Đặt ngòi nổ cho chuỗi thời gian phân đoạn ngữ đoạn).
2. **Peters, M. E., et al. (2018).** *Deep contextualized word representations.* NAACL (Mô hình hóa Context Dependency ELMo).
3. Tài liệu mô phỏng logic mạng học sâu *Cosine similarity in word sequences.*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [aero llm 01 codechallenge cosine similarity advanced part 1](aero_llm_01_codechallenge_cosine_similarity_advanced_part_1_.md) | [Xem bài viết →](aero_llm_01_codechallenge_cosine_similarity_advanced_part_1_.md) |
| [aero llm 02 codechallenge cosine similarity advanced part 2](aero_llm_02_codechallenge_cosine_similarity_advanced_part_2_.md) | [Xem bài viết →](aero_llm_02_codechallenge_cosine_similarity_advanced_part_2_.md) |
| 📌 **[Theo Dõi Dòng Chảy Cosine Similarity Trên Trục Văn Bản Chuyên Tuần Tự (Word Sequences)](aero_llm_03_codechallenge_cosine_similarity_in_word_sequences.md)** | [Xem bài viết →](aero_llm_03_codechallenge_cosine_similarity_in_word_sequences.md) |
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
