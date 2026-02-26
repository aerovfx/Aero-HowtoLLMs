
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
# Độ Ổn Định của Embeddings trong Mô Hình Ngôn Ngữ: Phân Tích Toán Học và Thực Nghiệm

Tóm tắt

Embeddings là nền tảng của các mô hình xử lý ngôn ngữ tự nhiên hiện đại. Tuy nhiên, một câu hỏi quan trọng đặt ra: các embeddings có ổn định giữa các lần huấn luyện khác nhau hay không? Bài viết này phân tích tính ổn định của embeddings dưới góc nhìn toán học, thống kê và hình học không gian vector. Nội dung dựa trên bài thực hành “How Stable Are Embeddings?” và mở rộng từ các nghiên cứu của Tomas Mikolov (Word2Vec), Jeffrey Pennington (GloVe), và Yoshua Bengio.

⸻

1. Giới thiệu

Giả sử ta huấn luyện cùng một mô hình embedding nhiều lần với:
	•	Cùng tập dữ liệu
	•	Cùng kiến trúc
	•	Khác khởi tạo ngẫu nhiên

Ta thu được hai ma trận embedding:

E^{(1)}, \quad E^{(2)} \in \mathbb{R}^{V \times d}

Câu hỏi:

E^{(1)} \stackrel{?}{\approx} E^{(2)}

Trên thực tế, các embedding không trùng khớp từng phần tử, nhưng có thể tương đương về cấu trúc hình học.

⸻

2. Nguyên nhân Gây Bất Ổn Định

2.1 Tính bất định do khởi tạo ngẫu nhiên

Ban đầu:

\mathbf{v}_w^{(0)} \sim \mathcal{N}(0, \sigma^2 I)

Các điểm xuất phát khác nhau dẫn đến nghiệm tối ưu khác nhau trong không gian phi lồi.

⸻

2.2 Tính bất biến quay (Rotational Invariance)

Giả sử Q \in \mathbb{R}^{d \times d} là ma trận trực giao:

Q^\top Q = I

Nếu E là nghiệm tối ưu, thì:

E' = EQ

cũng là nghiệm tương đương vì:

(EQ)(EQ)^\top = EQQ^\top E^\top = EE^\top

Điều này giải thích vì sao embeddings giữa hai lần huấn luyện có thể khác nhau về tọa độ nhưng giống nhau về quan hệ tương đối.

⸻

3. Đo lường Độ Ổn Định

3.1 So sánh trực tiếp bằng chuẩn Frobenius

||E^{(1)} - E^{(2)}||_F

Tuy nhiên cách này không hiệu quả do vấn đề quay không gian.

⸻

3.2 Procrustes Alignment

Tìm ma trận quay tối ưu:

Q^{\ast} = \arg\min_Q ||E^{(1)}Q - E^{(2)}||_F

Sau căn chỉnh:

Stability = ||E^{(1)}Q^{\ast} - E^{(2)}||_F

Phương pháp này thường được dùng trong nghiên cứu ổn định embedding.

⸻

3.3 Tương tự cosine trung bình

Với mỗi từ w:

sim(w) =
\frac{
\mathbf{v}_w^{(1)} \cdot \mathbf{v}_w^{(2)}
}{
||\mathbf{v}_w^{(1)}||\,||\mathbf{v}_w^{(2)}||
}

Lấy trung bình trên toàn bộ từ vựng:

\overline{sim} =
\frac{1}{V} \sum_{w=1}^{V} sim(w)

⸻

4. Phân tích Lý thuyết

4.1 Hàm mục tiêu Skip-gram

\mathcal{L} =
- \sum_{(w,c)} \log
\frac{\exp(\mathbf{v}_w^\top \mathbf{v}_c)}
{\sum_{c'} \exp(\mathbf{v}_w^\top \mathbf{v}_{c'})}

Hàm mất mát này phụ thuộc vào tích vô hướng:

\mathbf{v}_w^\top \mathbf{v}_c

Do đó nếu:

\mathbf{v}'_w = Q\mathbf{v}_w

thì:

\mathbf{v}'_w{}^\top \mathbf{v}'_c
=
\mathbf{v}_w^\top Q^\top Q \mathbf{v}_c
=
\mathbf{v}_w^\top \mathbf{v}_c

→ Hàm mất mát không đổi.

⸻

5. Thực nghiệm: Kết quả điển hình

Từ bài Code Challenge:
	•	Embeddings thay đổi mạnh về giá trị tuyệt đối
	•	Sau căn chỉnh Procrustes → độ tương tự tăng đáng kể
	•	Quan hệ ngữ nghĩa (nearest neighbors) gần như giữ nguyên

Ví dụ:

NN^{(1)}(king) \approx NN^{(2)}(king)

⸻

6. Ảnh hưởng của Kích thước và Dữ liệu

6.1 Kích thước embedding

Khi d lớn:
	•	Không gian nghiệm rộng hơn
	•	Variance tăng

Theo phân tích bias–variance:

\mathbb{E}[(y - \hat{f}(x))^2]
=
Bias^2 + Variance + \sigma^2

⸻

6.2 Kích thước tập dữ liệu

Khi số mẫu N \rightarrow \infty:

\hat{\theta}_N \rightarrow \theta^{\ast}

Theo định lý hội tụ, embeddings trở nên ổn định hơn.

⸻

7. Embeddings trong Transformer

Trong kiến trúc của Ashish Vaswani:

\mathbf{x}_i = \mathbf{e}_i + \mathbf{p}_i

Các embedding được cập nhật qua nhiều lớp attention:

Attention(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V

Do sự lan truyền gradient qua nhiều tầng, embeddings thường ổn định hơn so với mô hình nông.

⸻

8. Thảo luận

Embeddings không ổn định tuyệt đối ở mức tọa độ, nhưng:
	•	Ổn định về cấu trúc hình học
	•	Ổn định về quan hệ ngữ nghĩa
	•	Bất biến theo phép quay

Do đó, tính ổn định nên được đánh giá bằng:
	•	Quan hệ láng giềng gần
	•	Cấu trúc khoảng cách
	•	Phổ trị riêng của ma trận tương quan

⸻

9. Kết luận

Độ ổn định của embeddings phụ thuộc vào:
	•	Khởi tạo ngẫu nhiên
	•	Kích thước không gian
	•	Lượng dữ liệu
	•	Thuật toán tối ưu

Về mặt toán học, embeddings là nghiệm của một bài toán tối ưu phi lồi có nhiều nghiệm tương đương theo phép quay. Do đó, sự khác biệt giữa các lần huấn luyện không đồng nghĩa với mất thông tin ngữ nghĩa.

Hiểu rõ bản chất này giúp:
	•	So sánh mô hình chính xác hơn
	•	Thiết kế thí nghiệm tái lập (reproducibility)
	•	Đánh giá embedding một cách có cơ sở khoa học

⸻

Tài liệu tham khảo
	1.	Mikolov, T. et al. (2013). Efficient Estimation of Word Representations in Vector Space.
	2.	Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation.
	3.	Bengio, Y. et al. (2003). A Neural Probabilistic Language Model.
	4.	Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
	5.	Hamilton, W. L. et al. (2016). Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change.
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
| [Sự Thay Đổi của Embeddings Trong Quá Trình Huấn Luyện: Phân Tích Toán Học và Thực Nghiệm](aero_llm_17_codechallenge_how_the_embeddings_change.md) | [Xem bài viết →](aero_llm_17_codechallenge_how_the_embeddings_change.md) |
| 📌 **[Độ Ổn Định của Embeddings trong Mô Hình Ngôn Ngữ: Phân Tích Toán Học và Thực Nghiệm](aero_llm_18_codechallenge_how_stable_are_embeddings.md)** | [Xem bài viết →](aero_llm_18_codechallenge_how_stable_are_embeddings.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
