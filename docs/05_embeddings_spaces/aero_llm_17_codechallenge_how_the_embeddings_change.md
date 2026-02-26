
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
# Sự Thay Đổi của Embeddings Trong Quá Trình Huấn Luyện: Phân Tích Toán Học và Thực Nghiệm

Tóm tắt

Biểu diễn từ (word embeddings) là nền tảng của các mô hình xử lý ngôn ngữ tự nhiên hiện đại. Trong quá trình huấn luyện, các vector embedding thay đổi liên tục nhằm tối ưu hóa hàm mục tiêu. Bài viết này phân tích cơ chế cập nhật embeddings dựa trên gradient descent, mô hình hóa sự thay đổi của không gian vector, và giải thích ý nghĩa hình học của quá trình tối ưu. Nội dung được xây dựng dựa trên bài thực hành “How the Embeddings Change”, kết hợp các công trình của Tomas Mikolov (Word2Vec), Jeffrey Pennington (GloVe), và Ashish Vaswani (Transformer).

⸻

1. Giới thiệu

Embeddings ánh xạ mỗi từ w thành một vector trong không gian \mathbb{R}^d:

E: w \rightarrow \mathbf{v}_w \in \mathbb{R}^d

Mục tiêu của huấn luyện là điều chỉnh các vector này sao cho:
	•	Các từ có ngữ nghĩa tương tự nằm gần nhau
	•	Quan hệ ngữ nghĩa được bảo toàn tuyến tính

Ví dụ nổi tiếng:

\mathbf{v}_{king} - \mathbf{v}_{man} + \mathbf{v}_{woman} \approx \mathbf{v}_{queen}

⸻

2. Cơ chế Toán học của Cập nhật Embeddings

2.1 Hàm mục tiêu (Skip-gram)

Trong Word2Vec (Mikolov et al., 2013), mục tiêu là tối đa hóa xác suất từ ngữ cảnh c xuất hiện quanh từ trung tâm w:

\max \prod_{(w,c)\in D} P(c|w)

Với softmax:

P(c|w) = \frac{\exp(\mathbf{v}_c^\top \mathbf{v}_w)}{\sum_{c'} \exp(\mathbf{v}_{c'}^\top \mathbf{v}_w)}

Hàm mất mát:

\mathcal{L} = - \sum_{(w,c)} \log P(c|w)

⸻

2.2 Gradient cập nhật vector

Gradient theo vector trung tâm:

\frac{\partial \mathcal{L}}{\partial \mathbf{v}_w}
= \sum_{c'} P(c'|w)\mathbf{v}_{c'} - \mathbf{v}_c

Cập nhật:

\mathbf{v}_w^{(t+1)} = \mathbf{v}_w^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{v}_w}

Trong đó \eta là learning rate.

⸻

3. Hình học của Không gian Embedding

3.1 Khoảng cách Cosine

Độ tương tự thường dùng cosine similarity:

\cos(\theta) = \frac{\mathbf{v}_a \cdot \mathbf{v}_b}
{||\mathbf{v}_a|| \, ||\mathbf{v}_b||}

Khi huấn luyện:
	•	Từ xuất hiện cùng nhau → góc giảm
	•	Từ không liên quan → góc tăng

⸻

3.2 Di chuyển trong không gian vector

Giả sử tại bước t:

\Delta \mathbf{v} = -\eta \nabla \mathcal{L}

Vector dịch chuyển theo hướng giảm loss. Tổng quát:

\mathbf{v}^{(T)} = \mathbf{v}^{(0)} - \eta \sum_{t=0}^{T-1} \nabla \mathcal{L}^{(t)}

Điều này cho thấy embedding cuối cùng là tích lũy của toàn bộ lịch sử gradient.

⸻

4. Embeddings trong Transformer

Trong kiến trúc Transformer (Vaswani et al., 2017), embedding được cộng với positional encoding:

\mathbf{x}_i = \mathbf{e}_i + \mathbf{p}_i

Self-attention:

Attention(Q,K,V) =
\text{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V

Ở đây embedding không chỉ cập nhật từ loss cuối cùng mà còn qua cơ chế attention đa đầu.

⸻

5. Phân tích Thực nghiệm: Sự thay đổi Embeddings

Dựa trên bài Code Challenge:
	1.	Ban đầu embeddings gần như ngẫu nhiên
	2.	Sau vài epoch:
	•	Cluster hình thành
	•	Cosine similarity giữa từ đồng nghĩa tăng
	3.	Sau hội tụ:
	•	Không gian ổn định
	•	Gradient tiệm cận 0

Điều kiện hội tụ:

||\nabla \mathcal{L}|| \rightarrow 0

⸻

6. Regularization và Ổn định

Thêm L2 regularization:

\mathcal{L}_{reg} = \mathcal{L} + \lambda ||\mathbf{v}||^2

Giúp tránh:
	•	Vector phình to vô hạn
	•	Overfitting

⸻

7. Bias–Variance trong Embeddings

Sai số kỳ vọng:

\mathbb{E}[(y - \hat{f}(x))^2]
=
Bias^2 + Variance + \sigma^2

Embeddings dimension lớn:
	•	Giảm bias
	•	Tăng variance

Cần cân bằng số chiều d.

⸻

8. Thảo luận

Sự thay đổi của embeddings phản ánh:
	•	Cấu trúc phân bố xác suất ngôn ngữ
	•	Quan hệ đồng xuất hiện
	•	Tối ưu hóa trong không gian phi tuyến

Trong các mô hình lớn hiện nay (LLMs), embeddings còn được:
	•	Fine-tune theo domain
	•	Điều chỉnh bằng RLHF
	•	Áp dụng contrastive learning

⸻

9. Kết luận

Embeddings không phải là vector tĩnh mà là thực thể động, liên tục thay đổi trong quá trình tối ưu hóa. Về mặt toán học, chúng là nghiệm của một bài toán tối ưu phi lồi trong không gian nhiều chiều. Sự tiến hóa của embeddings chính là quá trình hình thành cấu trúc ngữ nghĩa trong không gian vector.

Hiểu rõ cơ chế cập nhật giúp:
	•	Thiết kế mô hình hiệu quả hơn
	•	Chọn hyperparameter hợp lý
	•	Tránh hiện tượng mất ổn định huấn luyện

⸻

Tài liệu tham khảo
	1.	Mikolov, T. et al. (2013). Efficient Estimation of Word Representations in Vector Space.
	2.	Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation.
	3.	Vaswani, A. et al. (2017). Attention Is All You Need.
	4.	Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
	5.	Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
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
| [Xây Dựng Mô Hình Học Embedding Từ Đầu: Kiến Trúc, Tối Ưu Hoá và Phân Tích Toán Học](aero_llm_14_build_a_model_to_learn_the_embeddings.md) | [Xem bài viết →](aero_llm_14_build_a_model_to_learn_the_embeddings.md) |
| [Hàm Mất Mát Trong Huấn Luyện Embedding: Cơ Sở Lý Thuyết, Phân Tích Gradient và Ứng Dụng Trong Mô Hình Ngôn Ngữ](aero_llm_15_loss_function_to_train_the_embeddings.md) | [Xem bài viết →](aero_llm_15_loss_function_to_train_the_embeddings.md) |
| [Huấn luyện và Đánh giá Mô hình Học Máy: Cơ sở Lý thuyết và Thực tiễn](aero_llm_16_train_and_evaluate_the_model.md) | [Xem bài viết →](aero_llm_16_train_and_evaluate_the_model.md) |
| 📌 **[Sự Thay Đổi của Embeddings Trong Quá Trình Huấn Luyện: Phân Tích Toán Học và Thực Nghiệm](aero_llm_17_codechallenge_how_the_embeddings_change.md)** | [Xem bài viết →](aero_llm_17_codechallenge_how_the_embeddings_change.md) |
| [Độ Ổn Định của Embeddings trong Mô Hình Ngôn Ngữ: Phân Tích Toán Học và Thực Nghiệm](aero_llm_18_codechallenge_how_stable_are_embeddings.md) | [Xem bài viết →](aero_llm_18_codechallenge_how_stable_are_embeddings.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
