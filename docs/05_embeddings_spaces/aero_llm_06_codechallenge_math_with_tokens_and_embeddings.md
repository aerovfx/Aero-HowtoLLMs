
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
# Toán học của Token và Embedding trong Mô hình Ngôn ngữ Lớn

## Tóm tắt

Trong các mô hình ngôn ngữ hiện đại như [GPT-2](chatgpt://generic-entity?number=0) và [BERT](chatgpt://generic-entity?number=1), văn bản không được xử lý trực tiếp dưới dạng chữ mà được chuyển đổi thành **token** và sau đó ánh xạ sang không gian vector thông qua **embedding**. Bài viết này phân tích nền tảng toán học của quá trình token hóa, ánh xạ embedding, cũng như các phép toán đại số vector cho phép mô hình học được cấu trúc ngữ nghĩa.

---

## 1. Từ văn bản đến token

Giả sử ta có chuỗi văn bản:

$$
\mathcal{T} = (w_1, w_2, ..., w_n)
$$

Bộ tokenizer thực hiện ánh xạ:

$$
\tau: \mathcal{V}_{text} \rightarrow \mathcal{V}_{token}
$$

Trong đó:

- \( \mathcal{V}_{text} \): tập từ tự nhiên
- \( \mathcal{V}_{token} \): tập token rời rạc

Kết quả là dãy chỉ số:

$$
(t_1, t_2, ..., t_n), \quad t_i \in \{1,2,...,|V|\}
$$

---

## 2. One-hot Encoding

Mỗi token \( t_i \) được biểu diễn ban đầu dưới dạng vector one-hot:

$$
\mathbf{x}_i \in \mathbb{R}^{|V|}
$$

$$
x_{ij} =
\begin{cases}
1 & \text{nếu } j = t_i \\
0 & \text{ngược lại}
\end{cases}
$$

Đây là không gian rất cao chiều và không hiệu quả về mặt tính toán.

---

## 3. Ma trận Embedding

Ta định nghĩa ma trận embedding:

$$
E \in \mathbb{R}^{|V| \times d}
$$

Trong đó:

- \( |V| \): kích thước từ vựng
- \( d \): số chiều embedding

Vector embedding được tính:

$$
\mathbf{v}_i = \mathbf{x}_i E
$$

Do \( \mathbf{x}_i \) là one-hot, nên:

$$
\mathbf{v}_i = E_{t_i}
$$

Tức là lấy hàng thứ \( t_i \) của ma trận embedding.

---

## 4. Cộng embedding và vị trí (Positional Encoding)

Trong Transformer, embedding cuối cùng là tổng của:

$$
\mathbf{z}_i = \mathbf{v}_i + \mathbf{p}_i
$$

Trong đó \( \mathbf{p}_i \) là positional encoding:

$$
PE_{(pos,2k)} = \sin\left(\frac{pos}{10000^{2k/d}}\right)
$$

$$
PE_{(pos,2k+1)} = \cos\left(\frac{pos}{10000^{2k/d}}\right)
$$

Điều này giúp mô hình nhận biết thứ tự chuỗi.

---

## 5. Đại số vector trong không gian embedding

### 5.1 Độ tương đồng Cosine

$$
\text{cosine}(\mathbf{v}_i,\mathbf{v}_j)
=
\frac{\mathbf{v}_i \cdot \mathbf{v}_j}
{\|\mathbf{v}_i\|\|\mathbf{v}_j\|}
$$

Phản ánh mức độ tương đồng ngữ nghĩa.

---

### 5.2 Phép cộng ngữ nghĩa

Trong nhiều mô hình, quan hệ tuyến tính có thể xuất hiện:

$$
\mathbf{v}_{king} - \mathbf{v}_{man}
+ \mathbf{v}_{woman}
\approx
\mathbf{v}_{queen}
$$

Điều này cho thấy không gian embedding học được cấu trúc ngữ nghĩa tuyến tính.

---

## 6. Tối ưu hóa Embedding

Trong mô hình tự hồi quy như GPT-2, hàm mất mát là:

$$
\mathcal{L}
=
- \sum_{t=1}^{T}
\log P(w_t | w_{<t})
$$

Với:

$$
P(w_t | w_{<t})
=
\text{softmax}(W_o h_t)
$$

$$
\text{softmax}(z_i)
=
\frac{e^{z_i}}
{\sum_{j=1}^{|V|} e^{z_j}}
$$

Gradient lan truyền ngược để cập nhật ma trận embedding:

$$
E \leftarrow E - \eta \nabla_E \mathcal{L}
$$

Trong đó \( \eta \) là learning rate.

---

## 7. Hình học của không gian embedding

Giả sử:

$$
X \in \mathbb{R}^{n \times d}
$$

Ma trận hiệp phương sai:

$$
\Sigma = \frac{1}{n} X^T X
$$

Giải bài toán trị riêng:

$$
\Sigma \mathbf{u} = \lambda \mathbf{u}
$$

Các trị riêng lớn cho biết chiều chiếm ưu thế của không gian ngữ nghĩa.

---

## 8. Chuẩn hóa và ổn định số học

Thường áp dụng chuẩn hóa:

$$
\hat{\mathbf{v}} =
\frac{\mathbf{v}}{\|\mathbf{v}\|}
$$

Điều này làm:

$$
\|\hat{\mathbf{v}}\| = 1
$$

Giúp tăng ổn định khi tính attention và cosine similarity.

---

## 9. Từ token đến Attention

Self-attention tính:

$$
Q = XW_Q
$$
$$
K = XW_K
$$
$$
V = XW_V
$$

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Embedding ban đầu đóng vai trò nền tảng cho toàn bộ phép biến đổi này.

---

## 10. Kết luận

Quá trình từ văn bản đến embedding có thể tóm tắt:

$$
\text{Text}
\rightarrow
\text{Token}
\rightarrow
\text{One-hot}
\rightarrow
\text{Embedding}
\rightarrow
\text{Attention}
\rightarrow
\text{Contextual Representation}
$$

Về mặt toán học:

- Token là biến rời rạc.
- Embedding là ánh xạ tuyến tính sang không gian liên tục.
- Attention là phép biến đổi phi tuyến phụ thuộc ngữ cảnh.
- Toàn bộ hệ thống được tối ưu hóa thông qua gradient descent.

Hiểu rõ cấu trúc toán học này giúp giải thích vì sao các mô hình ngôn ngữ lớn có thể học được cấu trúc ngữ nghĩa phức tạp từ dữ liệu văn bản khổng lồ.

---

## Tài liệu tham khảo

1. Vaswani et al. (2017). Attention is All You Need.  
2. Radford et al. (2019). Language Models are Unsupervised Multitask Learners.  
3. Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.  
4. Jurafsky & Martin (2023). Speech and Language Processing.

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
| 📌 **[Toán học của Token và Embedding trong Mô hình Ngôn ngữ Lớn](aero_llm_06_codechallenge_math_with_tokens_and_embeddings.md)** | [Xem bài viết →](aero_llm_06_codechallenge_math_with_tokens_and_embeddings.md) |
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
