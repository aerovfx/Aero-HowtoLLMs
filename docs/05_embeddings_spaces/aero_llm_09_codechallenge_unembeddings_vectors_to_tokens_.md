
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
# Unembedding trong Mô Hình Ngôn Ngữ Lớn: Từ Vector Ẩn Đến Token

## Tóm tắt

Trong các mô hình ngôn ngữ dựa trên Transformer, quá trình “embedding” ánh xạ token rời rạc sang không gian vector liên tục. Tuy nhiên, bước ngược lại – chuyển từ vector ẩn sang phân phối xác suất trên token – được gọi là **unembedding**. Bài viết này phân tích nền tảng toán học của unembedding trong [GPT-2](chatgpt://generic-entity?number=0), làm rõ vai trò của weight tying, tích vô hướng, softmax và cấu trúc hình học của không gian từ vựng.

---

## 1. Giới thiệu

Quá trình xử lý văn bản trong mô hình ngôn ngữ có thể tóm tắt:

$$

\text{Token} 
\rightarrow 
\text{Embedding} 
\rightarrow 
\text{Transformer layers} 
\rightarrow 
\text{Hidden state} 
\rightarrow 
\text{Unembedding} 
\rightarrow 
\text{Softmax}

$$

Nếu embedding là ánh xạ:

$$

f: \mathcal{V} \rightarrow \mathbb{R}^d

$$

thì unembedding là ánh xạ ngược:

$$

g: \mathbb{R}^d \rightarrow \mathbb{R}^{|\mathcal{V}|}

$$

---

## 2. Embedding: Từ token đến vector

Giả sử từ vựng có kích thước $|V|$, ma trận embedding:

$$

E \in \mathbb{R}^{|V| \times d}

$$

Với token chỉ số $i$:

$$

\mathbf{v}_i = E[i]

$$

Nếu biểu diễn one-hot $\mathbf{x}_i$:

$$

\mathbf{v}_i = \mathbf{x}_i E

$$

---

## 3. Unembedding: Từ vector đến token

Sau khi qua các lớp Transformer, ta thu được hidden state:

$$

\mathbf{h}_t \in \mathbb{R}^d

$$

Để chuyển sang logit:

$$

\mathbf{z} = W_U \mathbf{h}_t

$$

Trong đó:

$$

W_U \in \mathbb{R}^{|V| \times d}

$$

Vector logit:

$$

z_i = \mathbf{w}_i \cdot \mathbf{h}_t

$$

---

## 4. Weight Tying

Trong GPT-2, thường sử dụng weight tying:

$$

W_U = E

$$

hoặc:

$$

W_U = E^T

$$

Khi đó:

$$

z_i = \mathbf{v}_i \cdot \mathbf{h}_t

$$

Điều này có ý nghĩa hình học:

> Logit của token $i$ chính là tích vô hướng giữa embedding của token đó và hidden state.

---

## 5. Softmax và phân phối xác suất

Xác suất dự đoán token tiếp theo:

$$

P(w_i | h_t)
=
\frac{e^{z_i}}
{\sum_{j=1}^{|V|} e^{z_j}}

$$

Thay $z_i = \mathbf{v}_i \cdot \mathbf{h}_t$:

$$

P(w_i)
=
\frac{
\exp(\mathbf{v}_i \cdot \mathbf{h}_t)
}
{
\sum_j
\exp(\mathbf{v}_j \cdot \mathbf{h}_t)
}

$$

Nếu chuẩn hóa:

$$

\mathbf{v}_i \cdot \mathbf{h}_t
=
\|\mathbf{v}_i\|
\|\mathbf{h}_t\|
\cos \theta_i

$$

Suy ra:

$$

P(w_i)
\propto
\exp(
\|\mathbf{v}_i\|
\|\mathbf{h}_t\|
\cos \theta_i
)

$$

Góc giữa vector quyết định xác suất.

---

## 6. Diễn giải hình học

Hidden state $\mathbf{h}_t$ có thể xem như:

- Một “truy vấn ngữ nghĩa”
- Một điểm trong không gian embedding

Unembedding thực hiện phép chiếu:

$$

\mathbf{z} = E \mathbf{h}_t

$$

Nghĩa là ta đo mức độ “gần” giữa $\mathbf{h}_t$ và từng vector từ vựng.

Nếu hai token có embedding gần nhau:

$$

\mathbf{v}_i \approx \mathbf{v}_j

$$

thì:

$$

z_i \approx z_j

$$

Do đó phân phối xác suất sẽ tương tự.

---

## 7. Hàm mất mát và tối ưu hóa

Hàm mất mát cross-entropy:

$$

\mathcal{L}
=
- \log P(w_{true})

$$

Gradient theo $\mathbf{h}_t$:

$$

\nabla_{\mathbf{h}_t}
\mathcal{L}
=
\sum_i
P(w_i)\mathbf{v}_i
-
\mathbf{v}_{true}

$$

Điều này cho thấy:

- Hidden state được điều chỉnh về phía embedding đúng
- Và đẩy xa embedding sai

---

## 8. So sánh với phân loại tuyến tính

Unembedding tương đương một bộ phân loại tuyến tính:

$$

z_i = \mathbf{w}_i^T \mathbf{h}_t

$$

Khác biệt là:

- Số lớp rất lớn (~50k)
- Trọng số gắn trực tiếp với embedding

---

## 9. Quan hệ với Cosine Similarity

Nếu chuẩn hóa embedding:

$$

\hat{\mathbf{v}}_i
=
\frac{\mathbf{v}_i}{\|\mathbf{v}_i\|}

$$

Khi đó:

$$

z_i
=
\|\mathbf{v}_i\|
\|\mathbf{h}_t\|
\cos\theta_i

$$

Nếu bỏ qua độ lớn:

$$

z_i \propto \cos\theta_i

$$

Như vậy unembedding về bản chất dựa trên cosine similarity.

---

## 10. Phân tích phổ (Spectral Perspective)

Giả sử ma trận embedding:

$$

E = U \Sigma V^T

$$

(SVD decomposition)

Hidden state:

$$

\mathbf{h}_t
=
V \mathbf{c}

$$

Logit:

$$

\mathbf{z}
=
U \Sigma \mathbf{c}

$$

Các giá trị singular lớn chi phối phân phối xác suất.

---

## 11. Ý nghĩa lý thuyết

Unembedding:

1. Chuyển từ không gian liên tục sang rời rạc.
2. Là phép chiếu tuyến tính quy mô lớn.
3. Phụ thuộc trực tiếp vào cấu trúc hình học của embedding.
4. Tạo liên kết chặt chẽ giữa học biểu diễn và dự đoán xác suất.

Về mặt toán học:

$$

\text{Prediction}
=
\text{Softmax}(E \mathbf{h}_t)

$$

---

## 12. Kết luận

Unembedding là bước cuối nhưng cực kỳ quan trọng trong mô hình ngôn ngữ. Nó:

- Chuyển hidden state thành phân phối token
- Dựa trên tích vô hướng trong không gian embedding
- Thể hiện rõ mối quan hệ giữa hình học vector và xác suất

Hiểu rõ cơ chế này giúp:

- Phân tích hành vi mô hình
- Thực hiện interpretability
- Thiết kế kỹ thuật steering và logit lens
- So sánh không gian biểu diễn giữa các mô hình

---

## Tài liệu tham khảo

1. Vaswani et al. (2017). Attention is All You Need.  
2. Radford et al. (2019). Language Models are Unsupervised Multitask Learners.  
3. Press & Wolf (2017). Using the Output Embedding to Improve Language Models.  
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
| [Toán học của Token và Embedding trong Mô hình Ngôn ngữ Lớn](aero_llm_06_codechallenge_math_with_tokens_and_embeddings.md) | [Xem bài viết →](aero_llm_06_codechallenge_math_with_tokens_and_embeddings.md) |
| [Cosine Similarity và Mối Quan Hệ với Hệ Số Tương Quan: Cơ Sở Toán Học và Ứng Dụng trong NLP](aero_llm_07_cosine_similarity_and_relation_to_correlation_.md) | [Xem bài viết →](aero_llm_07_cosine_similarity_and_relation_to_correlation_.md) |
| [Phân Tích Cosine Similarity trong Không Gian Embedding của GPT-2](aero_llm_08_codechallenge_gpt2_cosine_similarities.md) | [Xem bài viết →](aero_llm_08_codechallenge_gpt2_cosine_similarities.md) |
| 📌 **[Unembedding trong Mô Hình Ngôn Ngữ Lớn: Từ Vector Ẩn Đến Token](aero_llm_09_codechallenge_unembeddings_vectors_to_tokens_.md)** | [Xem bài viết →](aero_llm_09_codechallenge_unembeddings_vectors_to_tokens_.md) |
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
