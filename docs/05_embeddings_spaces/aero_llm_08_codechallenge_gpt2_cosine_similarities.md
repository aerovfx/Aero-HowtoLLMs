
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
# Phân Tích Cosine Similarity trong Không Gian Embedding của GPT-2

## Tóm tắt

Cosine similarity là một công cụ trung tâm trong việc phân tích cấu trúc hình học của không gian embedding trong các mô hình ngôn ngữ lớn. Bài viết này trình bày cơ sở toán học và quy trình thực nghiệm để tính toán cosine similarity giữa các token embedding của [GPT-2](chatgpt://generic-entity?number=0), đồng thời phân tích ý nghĩa hình học và thống kê của các giá trị tương đồng thu được.

---

## 1. Giới thiệu

Trong các mô hình Transformer sinh văn bản, mỗi token được ánh xạ sang một vector trong không gian $\mathbb{R}^d$. Với GPT-2:

- Kích thước embedding: $d = 768$ (bản base)
- Từ vựng: khoảng 50.000 token

Mỗi token $t$ có vector embedding:

$$
\mathbf{v}_t \in $\mathbb${R}^{768}
$$

Phân tích cosine similarity giữa các vector này giúp hiểu cấu trúc ngữ nghĩa nội tại của mô hình.

---

## 2. Cơ sở toán học của Cosine Similarity

Cho hai vector:

$$
\mathbf{x}, \mathbf{y} \in $\mathbb${R}^d
$$

Định nghĩa:

$$
\text{cosine}(\mathbf{x},\mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}} {\|\mathbf{x}\| \|\mathbf{y}\|}
$$

Trong đó:

$$

$$

\mathbf{x} \cdot \mathbf{y} = $\sum$_{i=1}^{d} x_i y_i

$$

$$

$$

$$

\|\mathbf{x}\| = \sqrt{$\sum$_{i=1}^{d} x_i^2}

$$

$$

Giá trị nằm trong khoảng:

$$
-1 $\le$q \text{cosine} $\le$q 1
$$

---

## 3. Chuẩn hóa và tính toán hiệu quả

Trong thực tế, ta chuẩn hóa trước:

$$
\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|}
$$

Khi đó:

$$
\text{cosine}(\mathbf{x},\mathbf{y}) = \hat{\mathbf{x}} \cdot \hat{\mathbf{y}}
$$

Nếu ma trận embedding:

$$
E \in $\mathbb${R}^{|V| \times d}
$$

Sau khi chuẩn hóa từng hàng:

$$
\hat{E}
$$

Ma trận cosine similarity toàn bộ từ vựng:

$$
S = \hat{E} \hat{E}^T
$$

---

## 4. Phân tích thực nghiệm với GPT-2

### 4.1 Trích xuất embedding

Với token index $i$:

$$

$$

\mathbf{v}_i = E[i]

$$

$$

Trong GPT-2, embedding đầu vào và embedding đầu ra thường được chia sẻ trọng số (weight tying):

$$
W_{out} = E^T
$$

Điều này tạo liên hệ hình học trực tiếp giữa không gian embedding và không gian dự đoán xác suất.

---

### 4.2 Ví dụ: So sánh token

Giả sử ta chọn token:

- “cat”
- “dog”
- “banana”

Ta tính:

$$
\text{sim}(\text{cat},\text{dog})
$$

$$
\text{sim}(\text{cat},\text{banana})
$$

Kỳ vọng:

$$
\text{sim}(\text{cat},\text{dog}) > \text{sim}(\text{cat},\text{banana})
$$

Do cấu trúc ngữ nghĩa gần nhau.

---

## 5. Phân bố Cosine Similarity trong không gian cao chiều

Giả sử hai vector ngẫu nhiên:

$$
\mathbf{x},\mathbf{y} \sim $\mathcal${N}(0,I_d)
$$

Khi $d \to $\infty$$:

$$

$$

$\mathbb${E}[\text{cosine}] = 0

$$

$$

$$

$$

\text{Var}(\text{cosine}) $\approx$ \frac{1}{d}

$$

$$

Với $d = 768$:

$$

$$

\text{Var} $\approx$ \frac{1}{768}

$$

$$

Do đó:

- Vector ngẫu nhiên gần trực giao
- Cosine lớn biểu thị cấu trúc học được

---

## 6. Liên hệ với Softmax và xác suất dự đoán

Trong GPT-2, xác suất token tiếp theo:

$$

$$

P(w_t  \mid  h_t) = \text{softmax}(W_{out} h_t)

$$

$$

Nếu weight tying:

$$
W_{out} = E^T
$$

Khi đó:

$$

$$

z_i = \mathbf{v}_i \cdot h_t

$$

$$

Softmax:

$$

$$

P(w_i) = \frac{e^{\mathbf{v}_i \cdot h_t}} {$\sum$_j e^{\mathbf{v}_j \cdot h_t}}

$$

$$

Như vậy:

> Dự đoán xác suất thực chất dựa trên tích vô hướng giữa embedding và hidden state.

Nếu chuẩn hóa:

$$

$$

\mathbf{v}_i \cdot h_t = \|\mathbf{v}_i\| \|h_t\| \cos\theta

$$

$$

Do đó cosine similarity trực tiếp ảnh hưởng đến xác suất dự đoán.

---

## 7. Ma trận tương đồng cục bộ

Cho tập $n$ token:

$$
X \in $\mathbb${R}^{n \times d}
$$

Ma trận cosine:

$$
S_{ij} = \frac{\mathbf{v}_i \cdot \mathbf{v}_j} {\|\mathbf{v}_i\|\|\mathbf{v}_j\|}
$$

Ta có thể phân tích:

- Cụm từ (clustering)
- Phân tích trị riêng:

$$
S \mathbf{u} = \lambda \mathbf{u}
$$

Giá trị riêng lớn phản ánh cấu trúc ngữ nghĩa chiếm ưu thế.

---

## 8. Khoảng cách tương đương

Nếu vector đã chuẩn hóa:

$$
\|\mathbf{x}-\mathbf{y}\|^2 = 2 - 2\cos\theta
$$

Suy ra:

$$
\cos\theta = 1 - \frac{1}{2} \|\mathbf{x}-\mathbf{y}\|^2
$$

Điều này cho thấy cosine similarity và Euclid distance tương đương về mặt hình học khi chuẩn hóa.

---

## 9. Ý nghĩa lý thuyết

Cosine similarity trong GPT-2:

1. Định nghĩa cấu trúc hình học của từ vựng.
2. Liên hệ trực tiếp với xác suất dự đoán.
3. Phản ánh cấu trúc phân bố dữ liệu huấn luyện.
4. Giảm ảnh hưởng của độ lớn vector.

Về bản chất:

$$
\text{Prediction} $\propto$ \exp(\|\mathbf{v}\|\|h\|\cos\theta)
$$

Do đó góc giữa vector đóng vai trò quyết định.

---

## 10. Kết luận

Phân tích cosine similarity trong GPT-2 cho thấy:

- Không gian embedding có cấu trúc hình học rõ ràng.
- Các token liên quan có góc nhỏ (cosine lớn).
- Dự đoán xác suất phụ thuộc trực tiếp vào tích vô hướng.
- Trong không gian cao chiều, cấu trúc học được nổi bật hơn nền ngẫu nhiên.

Hiểu rõ nền tảng toán học này giúp ta:

- Phân tích embedding hiệu quả
- So sánh mô hình
- Thực hiện Representational Similarity Analysis (RSA)
- Tối ưu hóa hệ thống retrieval hoặc semantic search

---

## Tài liệu tham khảo

1. Vaswani et al. (2017). Attention is All You Need.  
2. Radford et al. (2019). Language Models are Unsupervised Multitask Learners.  
3. Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.  
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
| [Cosine Similarity và Mối Quan Hệ với Hệ Số Tương Quan: Cơ Sở Toán Học và Ứng Dụng trong NLP](aero_llm_07_cosine_similarity_and_relation_to_correlation_.md) | [Xem bài viết →](aero_llm_07_cosine_similarity_and_relation_to_correlation_.md) |
| 📌 **[Phân Tích Cosine Similarity trong Không Gian Embedding của GPT-2](aero_llm_08_codechallenge_gpt2_cosine_similarities.md)** | [Xem bài viết →](aero_llm_08_codechallenge_gpt2_cosine_similarities.md) |
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
