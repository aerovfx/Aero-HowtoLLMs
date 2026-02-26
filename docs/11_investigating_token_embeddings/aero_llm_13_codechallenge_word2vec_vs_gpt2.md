
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
# So Sánh Không Gian Nhúng: Word2Vec Và GPT-2 Qua Phân Tích RSA

## Tóm tắt

Một rào cản chí mạng trong Nghiên cứu Dữ liệu Văn bản (NLP) là việc xác nhận chất lượng tương quan giữa hai cỗ máy sở hữu chiều kích nhúng (Embeddings dimension size) đôi đũa lệch. Ma trận của mô hình đại cương Word2Vec có 300 chiều ẩn (300D), trong khi GPT-2 nặng nề sở hữu 768 chiều nhúng (768D). Làm thế nào để giải phẫu và chẩn đoán liệu GPT-2 và Word2Vec có chia sẻ chung một "triết học toán học" ngôn từ hay không? Bài báo này sẽ vận dụng một đường vòng bằng giải tích không gian thông qua thủ thuật kết chiếu Hệ số Tương Quan Pearson nội hàm, nền tảng của phương thức **RSA (Representational Similarity Analysis)**.

---

## 1. Thiết Lập Điểm Khớp Giao (Intersection Point Matching)

Sự so sánh hai đa hình học không gian bắt buộc phải được gắn kết trên một tập đối tượng con neo đậu duy nhất. Phép lọc được thiết lập thông qua phân tách danh sách từ khóa ở cả hai tệp từ điển (Vocab Arrays) của hai Tokenizer (Word2vec token list và GPT tokenizer vocab). Giả sử hệ thống thiết lập bộ đếm quét trích lọc (iteration filtering) 100 từ có số lượng ký tự chính xác bằng 6 ($length = 6$ letters).

Thuật toán dò ngược Try-Catch exception sẽ tạo được một tập ma trận trung gian gồm số lượng $N=100$ chữ khớp lệnh có mặt trong cả 2 từ điển bất chấp sự lệch pha của chỉ mục Index, triệt tiêu mọi biến dị các phần phụ kiện token lỗi hoặc khoảng trống ảo (Spaces issues).

---

## 2. Kiến Thiết Khối Phân Giải Cục Bộ

Bất lực hoàn toàn trước phép trừ hoặc cộng tuyến tính giữa một vector 300D và 768D, phương pháp lấy đạo chéo bắt đầu với tính độc lập từng phe không gian một.

$$
Trích lấy cụm thông tin vector của N=100 token trong hai hộp không gian, áp dụng ma trận tích vô hướng khoảng cách chéo Cosine Similarity:
$$

S_{W2V} = \text{CosineSim}(E_{\text{w2v-100}}) \in \mathbb{R}^{100 \times 100}

S_{GPT2} = \text{CosineSim}(E_{\text{gpt2-100}}) \in \mathbb{R}^{100 \times 100}

**Chắt Cất Đại Lượng (Upper Triangular Tiling):** 
Dọc theo chéo chính (Diagonal elements), tất cả các thông số đều vô nghĩa vì chúng luôn $\equiv 1.0$ (Tự soi chiếu gương). Tương tự, mặt đối xứng chéo dưới (Lower triangular) cũng là thông tin vi phạm lỗi dư thừa. Do đó, chỉ môt mảnh tam giác trên cùng (Upper components extract) có trị số vô hướng $\frac{100 \times 99}{2} = 4950$ điểm dữ liệu thô được dàn phẳng thành vector dây một chiều $v_{w2v}$ và $v_{gpt2}$.

---

## 3. Pearson Correlation Lên Ngôi Của Sự Phi Tuyến

Đây chính là điểm giao mùa của phân tích. Liệu chúng ta có nên làm một phép đo khoảng cách Cosine Similarity giữa $v_{w2v}$ và $v_{gpt2}$ để cho RSA Score không? Cấu trúc của mạng nơ-ron hồi đáp: **Không được sử dụng Cosine Similarity cho cấu hình RSA so sánh, điểm số này luôn luôn phải chạy bằng chuẩn Pearson Correlation.**

Điều này xảy ra do định dạng Không gian Dịch tâm Dị hướng (Distribution offsets deviation): 
Quang phổ Cosine của Word2Vec luôn được chuẩn hóa rộng rãi nằm giữa khu vực khoảng $[ -0.2 , 0.5 ]$. Trong khi đó, tính chất khối lượng đồ thị học mạng biến áp tự hồi quy (Autoregressive Transformers networks) như GPT mang đến hiệu ứng chùm điểm tụ lõi nón - tất cả mọi Cosine Similarities của GPT-2 lơ lửng ở đỉnh dư dương luôn lớn hơn $0$, loanh quanh khoảng $[ 0.3 , 0.8 ]$.

Nếu giả tưởng ta ép ma trận Word2vec tịnh tiến xuống trừ đi $-1$ trị số (Mean Offset subtract 1), chỉ số Cosine Similarity đột ngột nhảy vực thay đổi phương hướng đồ thị toàn tập. Nhưng tính chất **Hệ số Pearson ($\rho$) không bao giờ gãy đổ**:

\rho = \frac{\text{Cov}(v_{w2v}, v_{gpt2})}{\sigma_{\text{w2v}} \sigma_{\text{gpt2}}}

Luật tính hiệp phương sai chia chuẩn độ lệch $Cov(X,Y)$ tự động loại bỏ mọi độ lệch trung bình tâm (global mean offsets shift), khiến Pearson Correlation chỉ xét dựa trên tính chất "*Chúng nhảy nhót lên và xuống cùng một biên độ hay không*". 

Kết cục của điểm $\rho$ tính được RSA Score cung cấp một chỉ số cao ấn tượng, thừa nhận việc máy học dự đoán ngôn ngữ GPT-2 trên Transformer hay mô hình cửa sổ bối cảnh nhỏ Continuous Bag-of-Words như Word2vec, sự kiến thiết thông triệt của ngôn ngữ loài người ở mức sâu nhất trong AI là tương đồng đáng kinh ngạc.

---

## Tài liệu tham khảo

1. **Abnar, S., et al. (2019).** *Blackbox meets blackbox: Representational Similarity and Stability Analysis of Neural Language Models.* Proceedings of the 2019 ACL Workshop BlackboxNLP.
2. **Kriegeskorte, N., et al. (2008).** *Representational similarity analysis.* 
3. Tài liệu đào tạo nâng cao *Investigating embeddings - CodeChallenge Word2vec vs. GPT2*.
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
| 📌 **[So Sánh Không Gian Nhúng: Word2Vec Và GPT-2 Qua Phân Tích RSA](aero_llm_13_codechallenge_word2vec_vs_gpt2.md)** | [Xem bài viết →](aero_llm_13_codechallenge_word2vec_vs_gpt2.md) |
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
