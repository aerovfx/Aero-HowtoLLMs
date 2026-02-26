
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [05 embeddings spaces](index.md)

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
Khảo sát và Phân tích Toán học Embedding Tiền huấn luyện GloVe

Từ Ma trận Đồng xuất hiện đến Cấu trúc Hình học Không gian Từ vựng

⸻

Tóm tắt

Dựa trên tài liệu đính kèm “Exploring GloVe Pretrained Embeddings”, bài viết này trình bày một phân tích khoa học về embedding tiền huấn luyện GloVe, bao gồm cơ sở lý thuyết, hàm mục tiêu tối ưu, cấu trúc hình học của không gian vector, và các đặc tính ngữ nghĩa học được học từ thống kê đồng xuất hiện toàn cục.

Bài viết đồng thời mở rộng bằng các nguồn học thuật nền tảng (Pennington et al., 2014; Mikolov et al., 2013; Levy & Goldberg, 2014) và cung cấp các công thức toán học minh hoạ chi tiết.

⸻

1. Giới thiệu

Biểu diễn từ (word representation) là nền tảng của nhiều hệ thống xử lý ngôn ngữ tự nhiên (NLP).

Mục tiêu là xây dựng ánh xạ:

E: V \rightarrow \mathbb{R}^d

Trong đó:
	•	V: tập từ vựng
	•	d: số chiều embedding

Khác với Word2Vec (dựa trên ngữ cảnh cục bộ), GloVe khai thác thống kê toàn cục của ma trận đồng xuất hiện.

⸻

2. Ma trận Đồng xuất hiện

Giả sử một corpus có tổng số từ T.

Định nghĩa:

X_{ij} = \text{số lần từ } w_j \text{ xuất hiện trong cửa sổ ngữ cảnh của } w_i

Tổng số lần xuất hiện của w_i:

X_i = \sum_j X_{ij}

Xác suất đồng xuất hiện:

P_{ij} = \frac{X_{ij}}{X_i}

⸻

3. Trực giác Tỷ lệ Xác suất

Pennington et al. (2014) lập luận rằng tỷ lệ xác suất đồng xuất hiện mang thông tin ngữ nghĩa:

\frac{P_{ik}}{P_{jk}}

Ví dụ:
	•	i = ice
	•	j = steam
	•	k = solid

Ta kỳ vọng:

\frac{P$\text{solid}|\text{ice}$}{P$\text{solid}|\text{steam}$} \gg 1

Do đó, embedding nên mã hóa các tỷ lệ này.

⸻

4. Hàm Mục tiêu của GloVe

GloVe tìm vector w_i và \tilde{w}_j sao cho:

w_i^\top \tilde{w}_j + b_i + b_j \approx \log X_{ij}

Hàm mất mát:

J = \sum_{i,j} f$X_{ij}$
\left(
w_i^\top \tilde{w}_j + b_i + b_j - \log X_{ij}
\right)^2

Trong đó:

f$x$ =
\begin{cases}
$x/x_{max}$^\alpha & x < x_{max} \\
1 & \text{otherwise}
\end{cases}

Thường:

\alpha = 0.75

⸻

5. Liên hệ với PMI (Pointwise Mutual Information)

PMI được định nghĩa:

PMI(i,j) = \log \frac{P_{ij}}{P_i P_j}

Levy & Goldberg (2014) chỉ ra rằng Word2Vec với negative sampling xấp xỉ phân rã ma trận:

PMI(i,j) - \log k

GloVe gần tương đương với việc factorize ma trận log-count.

Do đó:

w_i^\top \tilde{w}_j \approx PMI(i,j)

⸻

6. Hình học của Không gian Embedding

Embedding sau huấn luyện nằm trong:

\mathbb{R}^d

Khoảng cách cosine:

\cos$\theta$ =
\frac{w_i^\top w_j}
{\|w_i\| \|w_j\|}

Phản ánh độ tương đồng ngữ nghĩa.

⸻

6.1 Quan hệ Tuyến tính

Một tính chất nổi bật:

w_{king} - w_{man} + w_{woman} \approx w_{queen}

Điều này có thể diễn giải:

(w_{king} - w_{man}) \approx (w_{queen} - w_{woman})

Cho thấy tồn tại các hướng ngữ nghĩa trong không gian vector.

⸻

7. Phân tích Phổ Trị riêng (Eigenvalue Spectrum)

Ma trận đồng xuất hiện:

X \in \mathbb{R}^{|V| \times |V|}

Phân rã SVD:

X = U \Sigma V^\top

Embedding tương đương với chọn:

W = U_d \Sigma_d^{1/2}

Phổ trị riêng thường tuân theo luật Zipf:

\lambda_r \propto \frac{1}{r^\beta}

Theo George Kingsley Zipf.

⸻

8. Entropy và Thông tin

Entropy của phân bố từ:

H$W$ = -\sum_i P$w_i$\log P$w_i$

Mutual information giữa hai từ:

I(i;j) = \sum_{i,j} P_{ij} \log \frac{P_{ij}}{P_i P_j}

GloVe học embedding sao cho:

w_i^\top w_j \approx I(i;j)

⸻

9. Độ phức tạp Tính toán

Giả sử số phần tử khác 0 của X là |X|.

Độ phức tạp:

O(|X|d)

So với Transformer như BERT:

O(n^2 d)

GloVe hiệu quả hơn cho embedding tĩnh.

⸻

10. Hạn chế của GloVe
	1.	Embedding tĩnh
	2.	Không phụ thuộc ngữ cảnh
	3.	Không mô hình hóa thứ tự từ

Biểu diễn cố định:

e$w$ = \text{hằng số}

Trong khi mô hình ngữ cảnh:

e_t = f$w_1,\dots,w_T$

⸻

11. Thực nghiệm Khám phá Embedding

Các phép phân tích thường dùng:
	•	PCA:

Z = XW
	•	t-SNE:

P_{ij} \propto \exp$-\|x_i-x_j\|^2$

Cho thấy các cụm ngữ nghĩa rõ ràng:
	•	Quốc gia
	•	Giới tính
	•	Số nhiều

⸻

12. Kết luận

GloVe dựa trên nguyên lý:

w_i^\top w_j \approx \log X_{ij}

Embedding học được:
	•	Cấu trúc tuyến tính
	•	Quan hệ ngữ nghĩa
	•	Thông tin toàn cục

Mặc dù đã bị thay thế trong nhiều ứng dụng bởi mô hình Transformer, GloVe vẫn là nền tảng lý thuyết quan trọng trong biểu diễn từ phân bố.

⸻

Tài liệu tham khảo
	1.	Pennington, Socher & Manning (2014). GloVe: Global Vectors for Word Representation.
	2.	Mikolov et al. (2013). Efficient Estimation of Word Representations.
	3.	Levy & Goldberg (2014). Neural Word Embedding as Implicit Matrix Factorization.
	4.	Shannon (1948). A Mathematical Theory of Communication.
	5.	Zipf (1935). The Psycho-Biology of Language.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [aero llm 01 word2vec vs glove vs gpt vs bert oh my](aero_llm_01_word2vec_vs_glove_vs_gpt_vs_bert_oh_my_.md) | [Xem bài viết →](aero_llm_01_word2vec_vs_glove_vs_gpt_vs_bert_oh_my_.md) |
| 📌 **[aero llm 02 exploring glove pretrained embeddings](aero_llm_02_exploring_glove_pretrained_embeddings.md)** | [Xem bài viết →](aero_llm_02_exploring_glove_pretrained_embeddings.md) |
| [aero llm 03 codechallenge wikipedia vs twitter embeddings part 1](aero_llm_03_codechallenge_wikipedia_vs_twitter_embeddings_part_1_.md) | [Xem bài viết →](aero_llm_03_codechallenge_wikipedia_vs_twitter_embeddings_part_1_.md) |
| [So sánh Biểu Diễn Từ Vựng giữa Wikipedia và Twitter bằng Phân Tích Tương Đồng Biểu Diễn (RSA)](aero_llm_04_codechallenge_wikipedia_vs_twitter_embeddings_part_2_.md) | [Xem bài viết →](aero_llm_04_codechallenge_wikipedia_vs_twitter_embeddings_part_2_.md) |
| [So sánh Biểu Diễn Ngữ Nghĩa của GPT-2 và BERT thông qua Phân Tích Embedding](aero_llm_05_exploring_gpt2_and_bert_embeddings.md) | [Xem bài viết →](aero_llm_05_exploring_gpt2_and_bert_embeddings.md) |
| [Toán học của Token và Embedding trong Mô hình Ngôn ngữ Lớn](aero_llm_06_codechallenge_math_with_tokens_and_embeddings.md) | [Xem bài viết →](aero_llm_06_codechallenge_math_with_tokens_and_embeddings.md) |
| [Cosine Similarity và Mối Quan Hệ với Hệ Số Tương Quan: Cơ Sở Toán Học và Ứng Dụng trong NLP](aero_llm_07_cosine_similarity_and_relation_to_correlation_.md) | [Xem bài viết →](aero_llm_07_cosine_similarity_and_relation_to_correlation_.md) |
| [Phân Tích Cosine Similarity trong Không Gian Embedding của GPT-2](aero_llm_08_codechallenge_gpt2_cosine_similarities.md) | [Xem bài viết →](aero_llm_08_codechallenge_gpt2_cosine_similarities.md) |
| [Unembedding trong Mô Hình Ngôn Ngữ Lớn: Từ Vector Ẩn Đến Token](aero_llm_09_codechallenge_unembeddings_vectors_to_tokens_.md) | [Xem bài viết →](aero_llm_09_codechallenge_unembeddings_vectors_to_tokens_.md) |
| [Position Embeddings trong Transformer: Cơ Sở Toán Học và Ứng Dụng trong Mô Hình Ngôn Ngữ Lớn](aero_llm_10_position_embeddings.md) | [Xem bài viết →](aero_llm_10_position_embeddings.md) |
| [Phân Tích Thực Nghiệm Embedding Vị Trí Trong Transformer: Từ Cấu Trúc Tuyến Tính Đến Không Gian Hình Học](aero_llm_11_codechallenge_exploring_position_embeddings.md) | [Xem bài viết →](aero_llm_11_codechallenge_exploring_position_embeddings.md) |
| [Huấn Luyện Embedding Từ Đầu: Cơ Sở Toán Học, Cơ Chế Tối Ưu và Ứng Dụng Trong Mô Hình Ngôn Ngữ](aero_llm_12_training_embeddings_from_scratch.md) | [Xem bài viết →](aero_llm_12_training_embeddings_from_scratch.md) |
| [Thiết Kế Data Loader Cho Huấn Luyện Mô Hình Ngôn Ngữ: Cơ Sở Toán Học, Kiến Trúc và Tối Ưu Hoá](aero_llm_13_create_a_data_loader_to_train_a_model.md) | [Xem bài viết →](aero_llm_13_create_a_data_loader_to_train_a_model.md) |
| [Xây Dựng Mô Hình Học Embedding Từ Đầu: Kiến Trúc, Tối Ưu Hoá và Phân Tích Toán Học](aero_llm_14_build_a_model_to_learn_the_embeddings.md) | [Xem bài viết →](aero_llm_14_build_a_model_to_learn_the_embeddings.md) |
| [Hàm Mất Mát Trong Huấn Luyện Embedding: Cơ Sở Lý Thuyết, Phân Tích Gradient và Ứng Dụng Trong Mô Hình Ngôn Ngữ](aero_llm_15_loss_function_to_train_the_embeddings.md) | [Xem bài viết →](aero_llm_15_loss_function_to_train_the_embeddings.md) |
| [Huấn luyện và Đánh giá Mô hình Học Máy: Cơ sở Lý thuyết và Thực tiễn](aero_llm_16_train_and_evaluate_the_model.md) | [Xem bài viết →](aero_llm_16_train_and_evaluate_the_model.md) |
| [Sự Thay Đổi của Embeddings Trong Quá Trình Huấn Luyện: Phân Tích Toán Học và Thực Nghiệm](aero_llm_17_codechallenge_how_the_embeddings_change.md) | [Xem bài viết →](aero_llm_17_codechallenge_how_the_embeddings_change.md) |
| [Độ Ổn Định của Embeddings trong Mô Hình Ngôn Ngữ: Phân Tích Toán Học và Thực Nghiệm](aero_llm_18_codechallenge_how_stable_are_embeddings.md) | [Xem bài viết →](aero_llm_18_codechallenge_how_stable_are_embeddings.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
