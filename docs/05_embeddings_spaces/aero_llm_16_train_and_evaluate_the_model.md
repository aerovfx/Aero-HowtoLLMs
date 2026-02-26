
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
# Huấn luyện và Đánh giá Mô hình Học Máy: Cơ sở Lý thuyết và Thực tiễn

Tóm tắt

Huấn luyện và đánh giá mô hình là hai giai đoạn cốt lõi trong quy trình phát triển hệ thống học máy (Machine Learning – ML). Bài viết này trình bày cơ sở toán học của quá trình huấn luyện, các phương pháp tối ưu hóa, chiến lược chia dữ liệu, và các chỉ số đánh giá phổ biến. Đồng thời, bài viết bổ sung các công thức toán học nền tảng như hàm mất mát, gradient descent, entropy chéo và các thước đo hiệu suất. Nội dung được xây dựng theo chuẩn khoa học, kết hợp lý thuyết từ các tài liệu kinh điển trong lĩnh vực ML.

⸻

1. Giới thiệu

Trong học máy, mục tiêu của mô hình là xấp xỉ một hàm ánh xạ:

f_\theta: X \rightarrow Y

Trong đó:
	•	X là không gian đầu vào
	•	Y là không gian đầu ra
	•	\theta là tập tham số của mô hình

Quá trình huấn luyện nhằm tìm ra bộ tham số \theta^* sao cho hàm mất mát được tối thiểu hóa:

\theta^* = \arg\min_\theta \mathcal{L}$\theta$

⸻

2. Cơ sở Toán học của Huấn luyện Mô hình

2.1 Hàm mất mát (Loss Function)

Tùy theo loại bài toán, hàm mất mát được xác định khác nhau.

$a$ Hồi quy – Mean Squared Error (MSE)

\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} $y_i - \hat{y}_i$^2

Trong đó:
	•	y_i là giá trị thực
	•	\hat{y}_i là giá trị dự đoán

⸻

$b$ Phân loại – Cross Entropy Loss

\mathcal{L}_{CE} = - \sum_{i=1}^{n} y_i \log$\hat{y}_i$

Cross-entropy có nguồn gốc từ lý thuyết thông tin của Shannon (1948).

⸻

2.2 Tối ưu hóa bằng Gradient Descent

Thuật toán cập nhật tham số:

\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$\theta_t$

Trong đó:
	•	\eta là learning rate
	•	\nabla_\theta \mathcal{L} là gradient

Các biến thể:
	•	Batch Gradient Descent
	•	Stochastic Gradient Descent (SGD)
	•	Adam Optimizer:

m_t = \beta_1 m_{t-1} + $1-\beta_1$g_t
v_t = \beta_2 v_{t-1} + $1-\beta_2$g_t^2

Adam được đề xuất bởi Kingma & Ba (2015).

⸻

3. Quy trình Huấn luyện

3.1 Chia tập dữ liệu

Thông thường:
	•	Training set: 70–80%
	•	Validation set: 10–15%
	•	Test set: 10–15%

Mô hình được tối ưu trên training set, điều chỉnh siêu tham số trên validation set và đánh giá cuối cùng trên test set.

⸻

3.2 Overfitting và Underfitting

Overfitting

Mô hình học quá sát dữ liệu huấn luyện:

\mathcal{L}_{train} \ll \mathcal{L}_{test}

Giải pháp:
	•	Regularization:
\mathcal{L}_{reg} = \mathcal{L} + \lambda ||\theta||^2
	•	Dropout
	•	Early stopping

⸻

4. Đánh giá Mô hình

4.1 Bài toán Phân loại

$a$ Accuracy

Accuracy = \frac{TP + TN}{TP + TN + FP + FN}

⸻

$b$ Precision & Recall

Precision = \frac{TP}{TP + FP}

Recall = \frac{TP}{TP + FN}

⸻

$c$ F1-score

F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}

⸻

$d$ ROC-AUC

Diện tích dưới đường cong ROC đo khả năng phân biệt hai lớp.

⸻

4.2 Bài toán Hồi quy

$a$ Mean Absolute Error (MAE)

MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|

$b$ R² Score

R^2 = 1 - \frac{\sum $y_i - \hat{y}_i$^2}{\sum $y_i - \bar{y}$^2}

⸻

5. Đánh giá Thực nghiệm

Trong quá trình huấn luyện:
	•	Theo dõi loss curve
	•	So sánh train vs validation
	•	Sử dụng confusion matrix
	•	Cross-validation:

CV = \frac{1}{k} \sum_{i=1}^{k} \mathcal{L}_i

⸻

6. Thảo luận

Huấn luyện và đánh giá mô hình không chỉ là quá trình kỹ thuật mà còn là bài toán tối ưu hóa thống kê. Sai lệch (bias) và phương sai (variance) đóng vai trò quan trọng:

\mathbb{E}[$y - \hat{f}(x$)^2] = Bias^2 + Variance + \sigma^2

Cân bằng bias-variance là chìa khóa xây dựng mô hình tổng quát hóa tốt.

⸻

7. Kết luận

Quá trình huấn luyện và đánh giá mô hình dựa trên nền tảng toán học vững chắc của:
	•	Tối ưu hóa
	•	Xác suất thống kê
	•	Lý thuyết thông tin

Việc lựa chọn hàm mất mát, thuật toán tối ưu và chỉ số đánh giá phù hợp quyết định trực tiếp đến hiệu năng hệ thống. Trong bối cảnh AI hiện đại, đặc biệt với các mô hình lớn (Large Language Models), quy trình huấn luyện còn mở rộng sang:
	•	Fine-tuning
	•	Transfer learning
	•	Reinforcement Learning from Human Feedback (RLHF)

⸻

Tài liệu tham khảo
	1.	Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
	2.	Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
	3.	Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization.
	4.	Vapnik, V. (1998). Statistical Learning Theory. Wiley.
	5.	Shannon, C. E. (1948). A Mathematical Theory of Communication.
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
| 📌 **[Huấn luyện và Đánh giá Mô hình Học Máy: Cơ sở Lý thuyết và Thực tiễn](aero_llm_16_train_and_evaluate_the_model.md)** | [Xem bài viết →](aero_llm_16_train_and_evaluate_the_model.md) |
| [Sự Thay Đổi của Embeddings Trong Quá Trình Huấn Luyện: Phân Tích Toán Học và Thực Nghiệm](aero_llm_17_codechallenge_how_the_embeddings_change.md) | [Xem bài viết →](aero_llm_17_codechallenge_how_the_embeddings_change.md) |
| [Độ Ổn Định của Embeddings trong Mô Hình Ngôn Ngữ: Phân Tích Toán Học và Thực Nghiệm](aero_llm_18_codechallenge_how_stable_are_embeddings.md) | [Xem bài viết →](aero_llm_18_codechallenge_how_stable_are_embeddings.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
