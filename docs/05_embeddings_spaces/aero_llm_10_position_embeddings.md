
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
# Position Embeddings trong Transformer: Cơ Sở Toán Học và Ứng Dụng trong Mô Hình Ngôn Ngữ Lớn

## Tóm tắt

Kiến trúc Transformer không có cơ chế tuần tự nội tại như RNN, do đó cần một phương pháp mã hóa thứ tự của token trong chuỗi. Position Embeddings (PE) được đề xuất trong bài báo gốc “Attention is All You Need” nhằm bổ sung thông tin vị trí vào biểu diễn embedding. Bài viết này phân tích cơ sở toán học của positional encoding, các biến thể học được (learned positional embeddings), và vai trò của chúng trong các mô hình như [GPT-2](chatgpt://generic-entity?number=0) và [BERT](chatgpt://generic-entity?number=1).

---

## 1. Giới thiệu

Trong Transformer, self-attention chỉ dựa trên:

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Cơ chế này không chứa thông tin về vị trí thứ tự của token.

Do đó, nếu chỉ dùng embedding từ vựng:

$$
\mathbf{v}_i
$$

thì hai chuỗi:

- “dog bites man”
- “man bites dog”

sẽ có tập embedding giống nhau (chỉ khác thứ tự).

---

## 2. Biểu diễn vị trí: Công thức Sinusoidal

Trong bài báo Transformer gốc (Vaswani et al., 2017), positional encoding được định nghĩa:

$$
PE_{(pos,2k)} =
\sin\left(
\frac{pos}{10000^{2k/d}}
\right)
$$

$$
PE_{(pos,2k+1)} =
\cos\left(
\frac{pos}{10000^{2k/d}}
\right)
$$

Trong đó:

- $pos$: vị trí trong chuỗi
- $k$: chỉ số chiều
- $d$: kích thước embedding

---

## 3. Đặc tính toán học

### 3.1 Tần số hình học

Ta có:

$$
\omega_k = \frac{1}{10000^{2k/d}}
$$

Do đó:

$$
PE(pos,k) =
\sin(\omega_k pos)
\quad \text{hoặc} \quad
\cos(\omega_k pos)
$$

Tần số thay đổi theo cấp số nhân → cho phép mô hình biểu diễn cả:

- Quan hệ gần (low frequency)
- Quan hệ xa (high frequency)

---

### 3.2 Biểu diễn dịch chuyển tuyến tính

Một đặc tính quan trọng:

$$
PE(pos + \Delta)
=
PE(pos)\cos(\omega\Delta)
+
PE_{\perp}(pos)\sin(\omega\Delta)
$$

Điều này cho phép mô hình học quan hệ khoảng cách tuyến tính giữa các vị trí.

---

## 4. Kết hợp Embedding và Position

Embedding cuối cùng:

$$
\mathbf{z}_i
=
\mathbf{v}_i
+
\mathbf{p}_i
$$

Trong đó:

- $\mathbf{v}_i$: token embedding
- $\mathbf{p}_i$: positional embedding

Khi đó:

$$
Z = V + P
$$

với:

$$
V, P \in \mathbb{R}^{n \times d}
$$

---

## 5. Learned Positional Embeddings

Trong [GPT-2](chatgpt://generic-entity?number=2) và [BERT](chatgpt://generic-entity?number=3), positional embeddings thường được học trực tiếp:

$$
P \in \mathbb{R}^{L_{max} \times d}
$$

với $L_{max}$ là độ dài tối đa.

Khi đó:

$$
\mathbf{p}_i = P[i]
$$

Ưu điểm:

- Linh hoạt hơn
- Học trực tiếp từ dữ liệu

Nhược điểm:

- Không tự nhiên mở rộng sang chuỗi dài hơn độ dài huấn luyện

---

## 6. Phân tích hình học

Sau khi cộng:

$$
\mathbf{z}_i
=
\mathbf{v}_i + \mathbf{p}_i
$$

Self-attention tính:

$$
Q = ZW_Q
$$

$$
K = ZW_K
$$

Tích vô hướng:

$$
QK^T
=
(V + P)W_Q
((V + P)W_K)^T
$$

Khai triển:

$$
=
VW_QW_K^TV^T
+
VW_QW_K^TP^T
+
PW_QW_K^TV^T
+
PW_QW_K^TP^T
$$

Cho thấy attention bao gồm:

- Quan hệ token–token
- Quan hệ token–position
- Quan hệ position–position

---

## 7. Relative Position Encoding

Một số mô hình hiện đại sử dụng vị trí tương đối:

$$
\text{Attention}_{ij}
=
\frac{
Q_i K_j^T + b_{i-j}
}{
\sqrt{d}
}
$$

Trong đó $b_{i-j}$ phụ thuộc vào khoảng cách giữa vị trí.

Điều này giúp mô hình tổng quát hóa tốt hơn.

---

## 8. Ảnh hưởng đến Cosine Similarity

Do:

$$
\mathbf{z}_i
=
\mathbf{v}_i + \mathbf{p}_i
$$

Cosine similarity giữa hai token tại vị trí khác nhau:

$$
\text{cosine}(\mathbf{z}_i,\mathbf{z}_j)
=
\frac{
(\mathbf{v}_i+\mathbf{p}_i)\cdot
(\mathbf{v}_j+\mathbf{p}_j)
}{
\|\mathbf{z}_i\|\|\mathbf{z}_j\|
}
$$

Mở rộng tử số:

$$
=
\mathbf{v}_i\cdot\mathbf{v}_j
+
\mathbf{v}_i\cdot\mathbf{p}_j
+
\mathbf{p}_i\cdot\mathbf{v}_j
+
\mathbf{p}_i\cdot\mathbf{p}_j
$$

Cho thấy vị trí ảnh hưởng trực tiếp đến hình học embedding.

---

## 9. Tính bất biến và giới hạn

### 9.1 Không bất biến dịch chuyển

Với learned positional embedding:

$$
\mathbf{p}_{i+1}
\neq
\mathbf{p}_i + c
$$

Do đó mô hình không tự động bất biến với dịch chuyển.

---

### 9.2 Độ dài chuỗi

Với sinusoidal:

$$
PE(pos)
\text{ có thể tính cho mọi } pos
$$

Với learned:

$$
pos > L_{max}
\Rightarrow
\text{không xác định}
$$

---

## 10. Kết luận

Position embeddings là thành phần thiết yếu giúp Transformer:

- Nhận biết thứ tự
- Học quan hệ khoảng cách
- Mô hình hóa cấu trúc cú pháp

Về mặt toán học:

$$
\text{Transformer}
=
\text{Attention}(V + P)
$$

Sự lựa chọn giữa sinusoidal và learned positional embeddings ảnh hưởng đến:

- Khả năng tổng quát hóa
- Ổn định huấn luyện
- Hình học của không gian biểu diễn

Hiểu rõ cơ chế này giúp:

- Phân tích hành vi mô hình
- Thiết kế kiến trúc mới
- Mở rộng mô hình sang chuỗi dài hơn

---

## Tài liệu tham khảo

1. Vaswani et al. (2017). Attention is All You Need.  
2. Radford et al. (2019). Language Models are Unsupervised Multitask Learners.  
3. Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.  
4. Press & Wolf (2017). Using the Output Embedding to Improve Language Models.  
5. Jurafsky & Martin (2023). Speech and Language Processing.

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
| [Phân Tích Cosine Similarity trong Không Gian Embedding của GPT-2](aero_llm_08_codechallenge_gpt2_cosine_similarities.md) | [Xem bài viết →](aero_llm_08_codechallenge_gpt2_cosine_similarities.md) |
| [Unembedding trong Mô Hình Ngôn Ngữ Lớn: Từ Vector Ẩn Đến Token](aero_llm_09_codechallenge_unembeddings_vectors_to_tokens_.md) | [Xem bài viết →](aero_llm_09_codechallenge_unembeddings_vectors_to_tokens_.md) |
| 📌 **[Position Embeddings trong Transformer: Cơ Sở Toán Học và Ứng Dụng trong Mô Hình Ngôn Ngữ Lớn](aero_llm_10_position_embeddings.md)** | [Xem bài viết →](aero_llm_10_position_embeddings.md) |
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
