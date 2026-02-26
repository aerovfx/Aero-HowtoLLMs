
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
# Xây Dựng Mô Hình Học Embedding Từ Đầu: Kiến Trúc, Tối Ưu Hoá và Phân Tích Toán Học

Tóm tắt

Bài viết này trình bày quy trình xây dựng một mô hình học embedding từ đầu (build a model to learn the embeddings), bao gồm thiết kế kiến trúc tối thiểu cho bài toán dự đoán token tiếp theo, định nghĩa hàm mất mát, lan truyền ngược và phân tích động học tối ưu. Phân tích được đặt trong bối cảnh các mô hình ngôn ngữ tự hồi quy như GPT-2 của OpenAI, dựa trên nền tảng Transformer từ công trình Attention Is All You Need của Ashish Vaswani và cộng sự. Đồng thời, bài viết liên hệ với các mô hình embedding cổ điển như Word2Vec của Tomas Mikolov.

⸻

1. Giới thiệu

Embedding là ánh xạ từ không gian rời rạc sang không gian vector liên tục:

f: \{1,\dots,V\} \rightarrow \mathbb{R}^d

Trong đó:
	•	V: kích thước từ vựng
	•	d: số chiều embedding

Ma trận embedding:

\mathbf{E} \in \mathbb{R}^{V \times d}

Vector của token w:

\mathbf{e}_w = \mathbf{E}[w]

Mục tiêu huấn luyện là tìm \mathbf{E} sao cho embedding phản ánh cấu trúc ngữ nghĩa và ngữ cảnh.

⸻

2. Kiến trúc mô hình tối thiểu

Xét mô hình đơn giản cho bài toán next-token prediction.

2.1 Lớp Embedding

Token đầu vào:

\mathbf{x} \in \mathbb{R}^{B \times L}

Sau embedding:

\mathbf{H} =
\mathbf{E}[\mathbf{x}]
\in
\mathbb{R}^{B \times L \times d}

⸻

2.2 Lớp tuyến tính đầu ra

Logits:

\mathbf{Z}
=
\mathbf{H}
\mathbf{W}
+
\mathbf{b}

Với:
	•	\mathbf{W} \in \mathbb{R}^{d \times V}
	•	\mathbf{b} \in \mathbb{R}^{V}

⸻

2.3 Softmax

P(y=i \mid \mathbf{h})
=
\frac{
\exp(z_i)
}{
\sum_{j=1}^{V}
\exp(z_j)
}

⸻

3. Hàm mất mát và tối ưu

3.1 Cross-Entropy

\mathcal{L}
=
-
\sum_{t=1}^{L}
\log
P(y_t \mid x_{\lt t})

Trung bình trên batch:

\mathcal{L}_{batch}
=
\frac{1}{BL}
\sum_{b=1}^{B}
\sum_{t=1}^{L}
\mathcal{L}_{b,t}

⸻

3.2 Gradient theo embedding

Gọi:

\mathbf{p} = \text{softmax}(\mathbf{z})

Gradient theo logits:

\frac{\partial \mathcal{L}}{\partial \mathbf{z}}
=
\mathbf{p} - \mathbf{y}

Gradient theo embedding:

\frac{\partial \mathcal{L}}{\partial \mathbf{e}_w}
=
\mathbf{W}
(\mathbf{p} - \mathbf{y})

Cập nhật:

\mathbf{E}[w]
\leftarrow
\mathbf{E}[w]
-
\eta
\frac{\partial \mathcal{L}}{\partial \mathbf{E}[w]}

⸻

4. Trọng số buộc (Weight Tying)

Trong các mô hình như GPT-2, ta thường buộc:

\mathbf{W} = \mathbf{E}^T

Khi đó:

z_i
=
\mathbf{h}^T
\mathbf{e}_i

Ý nghĩa:
	•	Logit là tích vô hướng giữa hidden state và embedding token.
	•	Không gian embedding đóng vai trò vừa mã hoá vừa giải mã (unembedding).

⸻

5. Phân tích động học học embedding

5.1 Hướng cập nhật

Gradient embedding:

\Delta \mathbf{e}_w
=
-
\eta
\mathbf{W}
(\mathbf{p}-\mathbf{y})

Khi token dự đoán đúng:

\mathbf{p} \approx \mathbf{y}
\Rightarrow
\Delta \mathbf{e}_w \approx 0

Khi sai:
	•	Embedding dịch chuyển về phía vector đúng
	•	Tách xa vector sai

⸻

5.2 Phân tích hình học

Cosine similarity:

\cos(\theta)
=
\frac{
\mathbf{e}_a \cdot \mathbf{e}_b
}{
\|\mathbf{e}_a\|
\|\mathbf{e}_b\|
}

Qua huấn luyện:
	•	Token xuất hiện trong ngữ cảnh tương tự → vector gần nhau
	•	Hình thành cụm ngữ nghĩa

⸻

6. Liên hệ với Word2Vec

Trong Skip-gram:

P(c \mid w)
=
\frac{
\exp(\mathbf{u}_c^T \mathbf{v}_w)
}{
\sum_{j=1}^{V}
\exp(\mathbf{u}_j^T \mathbf{v}_w)
}

Tối ưu:

\max
\sum_{(w,c)}
\log P(c \mid w)

Negative Sampling:

\mathcal{L}
=
\log \sigma(\mathbf{u}_c^T \mathbf{v}_w)
+
\sum_{k=1}^{K}
\log \sigma(-\mathbf{u}_{n_k}^T \mathbf{v}_w)

Mô hình embedding hiện đại có thể xem như mở rộng của cơ chế này trong không gian sâu (deep contextual space).

⸻

7. Mở rộng sang Transformer

Trong Transformer:

\mathbf{z}_t
=
\mathbf{e}_t
+
\mathbf{p}_t

Self-attention:

\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V

Embedding ảnh hưởng trực tiếp đến attention scores:

QK^T
=
(\mathbf{E}+\mathbf{P})W_Q
W_K^T
(\mathbf{E}+\mathbf{P})^T

Do đó embedding không chỉ là bảng tra cứu mà là nền tảng cấu trúc toàn bộ mô hình.

⸻

8. Phân tích phổ và cấu trúc thấp chiều

Xét ma trận embedding:

\mathbf{E}
\in
\mathbb{R}^{V \times d}

Ma trận hiệp phương sai:

\mathbf{C}
=
\frac{1}{V}
\mathbf{E}^T
\mathbf{E}

Giải bài toán trị riêng:

\mathbf{C}\mathbf{v}_i
=
\lambda_i
\mathbf{v}_i

Thực nghiệm cho thấy:
	•	Phần lớn phương sai tập trung ở vài thành phần chính.
	•	Embedding có cấu trúc thấp chiều hiệu quả.

⸻

9. Hội tụ và ổn định

Với tối ưu Adam:

m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1) g_t

v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2) g_t^2

\theta_t
=
\theta_{t-1}
-
\eta
\frac{m_t}{\sqrt{v_t}+\epsilon}

Embedding thường hội tụ nhanh ở giai đoạn đầu do gradient lớn.

⸻

10. Kết luận

Xây dựng mô hình học embedding từ đầu bao gồm:
	1.	Thiết kế lớp embedding.
	2.	Định nghĩa bài toán dự đoán.
	3.	Tối ưu bằng cross-entropy.
	4.	Phân tích động học cập nhật.
	5.	Hiểu cấu trúc hình học của không gian embedding.

Embedding không chỉ là thành phần phụ trợ mà là không gian hình học trung tâm của mô hình ngôn ngữ.

⸻

Tài liệu tham khảo
	1.	Ashish Vaswani et al. (2017). Attention Is All You Need.
	2.	OpenAI (2019). GPT-2 Technical Report.
	3.	Tomas Mikolov et al. (2013). Efficient Estimation of Word Representations in Vector Space.
	4.	Kingma & Ba (2015). Adam: A Method for Stochastic Optimization.
	5.	Goodfellow et al. (2016). Deep Learning.
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
| [Position Embeddings trong Transformer: Cơ Sở Toán Học và Ứng Dụng trong Mô Hình Ngôn Ngữ Lớn](aero_llm_10_position_embeddings.md) | [Xem bài viết →](aero_llm_10_position_embeddings.md) |
| [Phân Tích Thực Nghiệm Embedding Vị Trí Trong Transformer: Từ Cấu Trúc Tuyến Tính Đến Không Gian Hình Học](aero_llm_11_codechallenge_exploring_position_embeddings.md) | [Xem bài viết →](aero_llm_11_codechallenge_exploring_position_embeddings.md) |
| [Huấn Luyện Embedding Từ Đầu: Cơ Sở Toán Học, Cơ Chế Tối Ưu và Ứng Dụng Trong Mô Hình Ngôn Ngữ](aero_llm_12_training_embeddings_from_scratch.md) | [Xem bài viết →](aero_llm_12_training_embeddings_from_scratch.md) |
| [Thiết Kế Data Loader Cho Huấn Luyện Mô Hình Ngôn Ngữ: Cơ Sở Toán Học, Kiến Trúc và Tối Ưu Hoá](aero_llm_13_create_a_data_loader_to_train_a_model.md) | [Xem bài viết →](aero_llm_13_create_a_data_loader_to_train_a_model.md) |
| 📌 **[Xây Dựng Mô Hình Học Embedding Từ Đầu: Kiến Trúc, Tối Ưu Hoá và Phân Tích Toán Học](aero_llm_14_build_a_model_to_learn_the_embeddings.md)** | [Xem bài viết →](aero_llm_14_build_a_model_to_learn_the_embeddings.md) |
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
