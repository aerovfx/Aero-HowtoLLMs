
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
# Cosine Similarity và Mối Quan Hệ với Hệ Số Tương Quan: Cơ Sở Toán Học và Ứng Dụng trong NLP

## Tóm tắt

Cosine similarity là một thước đo hình học phổ biến trong xử lý ngôn ngữ tự nhiên (NLP), đặc biệt khi so sánh các vector embedding trong các mô hình như [GPT-2](chatgpt://generic-entity?number=0) và [BERT](chatgpt://generic-entity?number=1). Bài viết này trình bày cơ sở toán học của cosine similarity, phân tích mối quan hệ của nó với hệ số tương quan Pearson, và làm rõ vai trò của chuẩn hóa vector trong không gian nhiều chiều.

---

## 1. Giới thiệu

Trong không gian vector $\mathbb{R}^d$, việc đo độ tương đồng giữa hai vector $\mathbf{x}, \mathbf{y}$ có thể thực hiện bằng nhiều cách:

- Khoảng cách Euclid
- Tích vô hướng
- Cosine similarity
- Hệ số tương quan

Trong các hệ embedding hiện đại, cosine similarity được ưu tiên do tính **bất biến theo độ lớn (scale-invariant)**.

---

## 2. Định nghĩa Cosine Similarity

Cho hai vector:

$$

\mathbf{x}, \mathbf{y} \in \mathbb{R}^d

$$

Cosine similarity được định nghĩa:

$$

\text{cosine}(\mathbf{x},\mathbf{y})
=
\frac{\mathbf{x} \cdot \mathbf{y}}
{\|\mathbf{x}\| \|\mathbf{y}\|}

$$

Trong đó:

$$

\mathbf{x} \cdot \mathbf{y}
=
\sum_{i=1}^{d} x_i y_i

$$

$$

\|\mathbf{x}\|
=
\sqrt{\sum_{i=1}^{d} x_i^2}

$$

### 2.1 Diễn giải hình học

Gọi $\theta$ là góc giữa hai vector:

$$

\mathbf{x} \cdot \mathbf{y}
=
\|\mathbf{x}\| \|\mathbf{y}\| \cos \theta

$$

Suy ra:

$$

\text{cosine}(\mathbf{x},\mathbf{y}) = \cos \theta

$$

Do đó:

- 1 → cùng hướng
- 0 → trực giao
- -1 → ngược hướng

---

## 3. Chuẩn hóa vector

Nếu ta chuẩn hóa:

$$

\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|}

$$

$$

\hat{\mathbf{y}} = \frac{\mathbf{y}}{\|\mathbf{y}\|}

$$

Khi đó:

$$

\text{cosine}(\mathbf{x},\mathbf{y})
=
\hat{\mathbf{x}} \cdot \hat{\mathbf{y}}

$$

Điều này cho thấy cosine similarity chính là tích vô hướng của các vector đơn vị.

---

## 4. Hệ số tương quan Pearson

Cho hai biến ngẫu nhiên $X, Y$, hệ số tương quan Pearson:

$$

\rho_{X,Y}
=
\frac{\text{Cov}(X,Y)}
{\sigma_X \sigma_Y}

$$

Trong đó:

$$

\text{Cov}(X,Y)
=
\frac{1}{n}
\sum_{i=1}^{n}
(x_i - \bar{x})(y_i - \bar{y})

$$

$$

\sigma_X
=
\sqrt{\frac{1}{n}
\sum_{i=1}^{n}
(x_i - \bar{x})^2}

$$

---

## 5. Mối quan hệ giữa Cosine và Pearson

Giả sử ta chuẩn hóa vector bằng cách trừ trung bình:

$$

\tilde{x}_i = x_i - \bar{x}

$$

$$

\tilde{y}_i = y_i - \bar{y}

$$

Khi đó:

$$

\rho_{X,Y}
=
\frac{\tilde{\mathbf{x}} \cdot \tilde{\mathbf{y}}}
{\|\tilde{\mathbf{x}}\|
\|\tilde{\mathbf{y}}\|}

$$

Như vậy:

> Pearson correlation chính là cosine similarity của hai vector đã được **centered (trừ trung bình)**.

### 5.1 So sánh bản chất

| Đặc điểm | Cosine | Pearson |
|----------|---------|----------|
| Trừ trung bình | Không | Có |
| Bất biến theo scale | Có | Có |
| Nhạy với offset | Có | Không |

---

## 6. Ứng dụng trong Embedding

Giả sử:

$$

E \in \mathbb{R}^{|V| \times d}

$$

với mỗi từ:

$$

\mathbf{v}_w \in \mathbb{R}^d

$$

Độ tương đồng ngữ nghĩa giữa hai từ:

$$

\text{sim}(w_i,w_j)
=
\frac{\mathbf{v}_i \cdot \mathbf{v}_j}
{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}

$$

---

## 7. Ma trận tương đồng

Cho tập $n$ từ:

$$

X \in \mathbb{R}^{n \times d}

$$

Ma trận cosine similarity:

$$

S_{ij}
=
\frac{\mathbf{v}_i \cdot \mathbf{v}_j}
{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}

$$

Nếu ta vector hóa phần tam giác trên của $S$ và tính tương quan giữa hai mô hình embedding khác nhau:

$$

r
=
\frac{\sum (x_i - \bar{x})(y_i - \bar{y})}
{\sqrt{\sum (x_i - \bar{x})^2}
\sqrt{\sum (y_i - \bar{y})^2}}

$$

Ta thu được mức độ tương đồng cấu trúc (Representational Similarity Analysis).

---

## 8. So sánh với Khoảng cách Euclid

Khoảng cách:

$$

d(\mathbf{x},\mathbf{y})
=
\|\mathbf{x}-\mathbf{y}\|

$$

Nếu vector đã chuẩn hóa:

$$

\|\mathbf{x}-\mathbf{y}\|^2
=
2 - 2\cos\theta

$$

Suy ra:

$$

\cos\theta
=
1 - \frac{1}{2}
\|\mathbf{x}-\mathbf{y}\|^2

$$

Điều này chứng minh cosine similarity và Euclid distance có quan hệ tuyến tính khi vector được chuẩn hóa.

---

## 9. Ý nghĩa hình học trong không gian cao chiều

Trong không gian cao chiều:

- Phần lớn vector ngẫu nhiên gần trực giao.
- Cosine similarity tập trung quanh 0.
- Embedding học được cấu trúc làm lệch phân bố này.

Giả sử:

$$

\mathbf{x},\mathbf{y}
\sim \mathcal{N}(0,I_d)

$$

Khi $d \to \infty$:

$$

\text{cosine}(\mathbf{x},\mathbf{y})
\to 0

$$

Đây là hiện tượng “curse of dimensionality”.

---

## 10. Kết luận

Cosine similarity là công cụ hình học cốt lõi trong NLP vì:

- Bất biến theo độ lớn vector
- Dễ tính toán
- Liên hệ trực tiếp với Pearson correlation
- Phù hợp với embedding đã chuẩn hóa

Về mặt toán học:

$$

\text{Pearson}
=
\text{Cosine}(\text{centered vectors})

$$

$$

\text{Euclid}
\leftrightarrow
\text{Cosine}
\quad (\text{khi chuẩn hóa})

$$

Hiểu rõ mối quan hệ này giúp ta phân tích chính xác cấu trúc không gian embedding và đánh giá sự tương đồng giữa các mô hình ngôn ngữ.

---

## Tài liệu tham khảo

1. Vaswani et al. (2017). Attention is All You Need.  
2. Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.  
3. Radford et al. (2019). Language Models are Unsupervised Multitask Learners.  
4. Jurafsky & Martin (2023). Speech and Language Processing.  
5. Kriegeskorte et al. (2008). Representational Similarity Analysis.

---
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [aero llm 01 word2vec vs glove vs gpt vs bert oh my](aero_llm_01_word2vec_vs_glove_vs_gpt_vs_bert_oh_my_.md) | [Xem bài viết →](aero_llm_01_word2vec_vs_glove_vs_gpt_vs_bert_oh_my_.md) |
| [aero llm 02 exploring glove pretrained embeddings](aero_llm_02_exploring_glove_pretrained_embeddings.md) | [Xem bài viết →](aero_llm_02_exploring_glove_pretrained_embeddings.md) |
| [aero llm 03 codechallenge wikipedia vs twitter embeddings part 1](aero_llm_03_codechallenge_wikipedia_vs_twitter_embeddings_part_1_.md) | [Xem bài viết →](aero_llm_03_codechallenge_wikipedia_vs_twitter_embeddings_part_1_.md) |
| [So sánh Biểu Diễn Từ Vựng giữa Wikipedia và Twitter bằng Phân Tích Tương Đồng Biểu Diễn (RSA)](aero_llm_04_codechallenge_wikipedia_vs_twitter_embeddings_part_2_.md) | [Xem bài viết →](aero_llm_04_codechallenge_wikipedia_vs_twitter_embeddings_part_2_.md) |
| [So sánh Biểu Diễn Ngữ Nghĩa của GPT-2 và BERT thông qua Phân Tích Embedding](aero_llm_05_exploring_gpt2_and_bert_embeddings.md) | [Xem bài viết →](aero_llm_05_exploring_gpt2_and_bert_embeddings.md) |
| [Toán học của Token và Embedding trong Mô hình Ngôn ngữ Lớn](aero_llm_06_codechallenge_math_with_tokens_and_embeddings.md) | [Xem bài viết →](aero_llm_06_codechallenge_math_with_tokens_and_embeddings.md) |
| 📌 **[Cosine Similarity và Mối Quan Hệ với Hệ Số Tương Quan: Cơ Sở Toán Học và Ứng Dụng trong NLP](aero_llm_07_cosine_similarity_and_relation_to_correlation_.md)** | [Xem bài viết →](aero_llm_07_cosine_similarity_and_relation_to_correlation_.md) |
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
