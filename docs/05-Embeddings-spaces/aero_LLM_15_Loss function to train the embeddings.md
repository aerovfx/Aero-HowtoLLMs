
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [05 Embeddings spaces](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Hàm Mất Mát Trong Huấn Luyện Embedding: Cơ Sở Lý Thuyết, Phân Tích Gradient và Ứng Dụng Trong Mô Hình Ngôn Ngữ

Tóm tắt

Hàm mất mát (loss function) đóng vai trò trung tâm trong quá trình huấn luyện embedding cho mô hình ngôn ngữ. Bài viết này trình bày chi tiết các dạng hàm mất mát phổ biến dùng để huấn luyện embedding, bao gồm Cross-Entropy, Negative Sampling và các biến thể chuẩn hoá xác suất. Đồng thời, chúng tôi phân tích đạo hàm, động học cập nhật gradient và cấu trúc hình học của không gian embedding được hình thành. Bối cảnh nghiên cứu được đặt trong các mô hình tự hồi quy như GPT-2 của OpenAI, dựa trên kiến trúc Transformer từ công trình Attention Is All You Need của Ashish Vaswani và liên hệ với Word2Vec của Tomas Mikolov.

⸻

1. Giới thiệu

Embedding ánh xạ token rời rạc sang không gian liên tục:

\mathbf{E} \in \mathbb{R}^{V \times d}

Với:
	•	V: kích thước từ vựng
	•	d: số chiều embedding

Vector của token w:

\mathbf{e}_w = \mathbf{E}[w]

Để embedding học được cấu trúc ngữ nghĩa, cần định nghĩa một hàm mất mát phản ánh mục tiêu dự đoán.

⸻

2. Hàm mất mát Cross-Entropy cho bài toán dự đoán token

2.1 Xác suất Softmax

Logits:

z_i = \mathbf{h}^T \mathbf{w}_i

Xác suất:

P(y=i) =
\frac{\exp(z_i)}
{\sum_{j=1}^{V} \exp(z_j)}

⸻

2.2 Hàm mất mát

\mathcal{L}
=
-
\sum_{i=1}^{V}
y_i \log P(y=i)

Vì y là one-hot:

\mathcal{L}
=
-
\log P(y = y_{true})

Mục tiêu tối ưu:

\min_\theta \mathcal{L}

⸻

3. Phân tích gradient

3.1 Gradient theo logits

\frac{\partial \mathcal{L}}{\partial z_i}
=
P(y=i) - y_i

⸻

3.2 Gradient theo embedding

Với weight tying \mathbf{W} = \mathbf{E}^T:

z_i = \mathbf{h}^T \mathbf{e}_i

Gradient theo embedding token đúng y:

\frac{\partial \mathcal{L}}{\partial \mathbf{e}_y}
=
(P(y) - 1)\mathbf{h}

Với token sai:

\frac{\partial \mathcal{L}}{\partial \mathbf{e}_i}
=
P(i)\mathbf{h}

Diễn giải hình học:
	•	Embedding đúng được kéo gần \mathbf{h}
	•	Embedding sai bị đẩy xa

⸻

4. Negative Sampling

Trong Word2Vec:

\mathcal{L}
=
\log \sigma(\mathbf{u}_c^T \mathbf{v}_w)
+
\sum_{k=1}^{K}
\log \sigma(-\mathbf{u}_{n_k}^T \mathbf{v}_w)

Trong đó:

\sigma(x)
=
\frac{1}{1+e^{-x}}

Gradient theo tích vô hướng:

\frac{d}{dx}
\log \sigma(x)
=
1 - \sigma(x)

Phương pháp này giảm chi phí tính toán từ:

O(V)
\rightarrow
O(K)

⸻

5. Phân tích độ lồi và ổn định

Cross-Entropy với softmax là hàm lồi theo logits:

\frac{\partial^2 \mathcal{L}}{\partial z_i^2}
=
P(i)(1-P(i))

Ma trận Hessian:

H = \text{diag}(P) - PP^T

H là bán xác định dương (positive semi-definite).

Tuy nhiên, theo tham số embedding, bài toán không còn lồi do tính chất phi tuyến của mạng sâu.

⸻

6. Entropy và tối đa hoá khả năng

Cross-Entropy:

H(p,q)
=
-
\sum p(x)\log q(x)

Tối thiểu hoá Cross-Entropy tương đương với:

\min H(p,q)
\iff
\min D_{KL}(p||q)

Vì:

H(p,q)
=
H(p)
+
D_{KL}(p||q)

Trong đó:

D_{KL}(p||q)
=
\sum p(x)\log\frac{p(x)}{q(x)}

⸻

7. Vai trò trong Transformer

Trong mô hình như GPT-2:

\mathbf{z}_t
=
\mathbf{e}_t
+
\mathbf{p}_t

Loss toàn chuỗi:

\mathcal{L}
=
-
\sum_{t=1}^{T}
\log
P(x_t \mid x_{<t})

Gradient truyền ngược qua:
	•	Unembedding
	•	Self-attention
	•	Embedding

Embedding được cập nhật gián tiếp thông qua toàn bộ kiến trúc.

⸻

8. Phân tích động học học embedding

Giả sử:

\Delta \mathbf{e}
=
-\eta \nabla_{\mathbf{e}}\mathcal{L}

Sau nhiều bước:

\mathbf{e}_w^{(t)}
=
\mathbf{e}_w^{(0)}
-
\eta
\sum_{k=1}^{t}
\nabla_{\mathbf{e}_w}
\mathcal{L}_k

Token xuất hiện thường xuyên:

\|\mathbf{e}_w\|
\uparrow

Do tích lũy gradient nhiều hơn.

⸻

9. Phân tích hình học

Cosine similarity:

\cos(\theta)
=
\frac{\mathbf{e}_a \cdot \mathbf{e}_b}
{\|\mathbf{e}_a\|\|\mathbf{e}_b\|}

Huấn luyện làm tăng:

\mathbf{e}_w^T \mathbf{e}_c
\quad \text{khi } w,c \text{ xuất hiện cùng nhau}

Embedding hình thành các cụm ngữ nghĩa trong không gian cao chiều.

⸻

10. Kết luận

Hàm mất mát là cơ chế điều khiển quá trình hình thành không gian embedding.

Các điểm chính:
	1.	Cross-Entropy tối ưu xác suất dự đoán.
	2.	Gradient điều chỉnh embedding theo hướng hình học rõ ràng.
	3.	Negative Sampling giảm chi phí tính toán.
	4.	Loss ảnh hưởng trực tiếp đến cấu trúc hình học embedding.
	5.	Trong Transformer, embedding học thông qua toàn bộ pipeline attention.

Embedding không chỉ học thông qua tần suất mà thông qua cấu trúc phân phối xác suất toàn cục.

⸻

Tài liệu tham khảo
	1.	Ashish Vaswani et al. (2017). Attention Is All You Need.
	2.	OpenAI (2019). GPT-2 Technical Report.
	3.	Tomas Mikolov et al. (2013). Efficient Estimation of Word Representations in Vector Space.
	4.	Goodfellow et al. (2016). Deep Learning.
	5.	Bishop (2006). Pattern Recognition and Machine Learning.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [aero_LLM_01_Word2Vec vs. GloVe vs. GPT vs. BERT... oh my!.md](aero_LLM_01_Word2Vec vs. GloVe vs. GPT vs. BERT... oh my!.md) | [Xem bài viết →](aero_LLM_01_Word2Vec vs. GloVe vs. GPT vs. BERT... oh my!.md) |
| [aero_LLM_02_Exploring GloVe pretrained embeddings.md](aero_LLM_02_Exploring GloVe pretrained embeddings.md) | [Xem bài viết →](aero_LLM_02_Exploring GloVe pretrained embeddings.md) |
| [aero_LLM_03_CodeChallenge Wikipedia vs. Twitter embeddings (part 1).md](aero_LLM_03_CodeChallenge Wikipedia vs. Twitter embeddings (part 1).md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Wikipedia vs. Twitter embeddings (part 1).md) |
| [So sánh Biểu Diễn Từ Vựng giữa Wikipedia và Twitter bằng Phân Tích Tương Đồng Biểu Diễn (RSA)](aero_LLM_04_CodeChallenge Wikipedia vs. Twitter embeddings (part 2).md) | [Xem bài viết →](aero_LLM_04_CodeChallenge Wikipedia vs. Twitter embeddings (part 2).md) |
| [So sánh Biểu Diễn Ngữ Nghĩa của GPT-2 và BERT thông qua Phân Tích Embedding](aero_LLM_05_Exploring GPT2 and BERT embeddings.md) | [Xem bài viết →](aero_LLM_05_Exploring GPT2 and BERT embeddings.md) |
| [Toán học của Token và Embedding trong Mô hình Ngôn ngữ Lớn](aero_LLM_06_CodeChallenge Math with tokens and embeddings.md) | [Xem bài viết →](aero_LLM_06_CodeChallenge Math with tokens and embeddings.md) |
| [Cosine Similarity và Mối Quan Hệ với Hệ Số Tương Quan: Cơ Sở Toán Học và Ứng Dụng trong NLP](aero_LLM_07_Cosine similarity (and relation to correlation).md) | [Xem bài viết →](aero_LLM_07_Cosine similarity (and relation to correlation).md) |
| [Phân Tích Cosine Similarity trong Không Gian Embedding của GPT-2](aero_LLM_08_CodeChallenge GPT2 cosine similarities.md) | [Xem bài viết →](aero_LLM_08_CodeChallenge GPT2 cosine similarities.md) |
| [Unembedding trong Mô Hình Ngôn Ngữ Lớn: Từ Vector Ẩn Đến Token](aero_LLM_09_CodeChallenge Unembeddings (vectors to tokens).md) | [Xem bài viết →](aero_LLM_09_CodeChallenge Unembeddings (vectors to tokens).md) |
| [Position Embeddings trong Transformer: Cơ Sở Toán Học và Ứng Dụng trong Mô Hình Ngôn Ngữ Lớn](aero_LLM_10_Position embeddings.md) | [Xem bài viết →](aero_LLM_10_Position embeddings.md) |
| [Phân Tích Thực Nghiệm Embedding Vị Trí Trong Transformer: Từ Cấu Trúc Tuyến Tính Đến Không Gian Hình Học](aero_LLM_11_CodeChallenge Exploring position embeddings.md) | [Xem bài viết →](aero_LLM_11_CodeChallenge Exploring position embeddings.md) |
| [Huấn Luyện Embedding Từ Đầu: Cơ Sở Toán Học, Cơ Chế Tối Ưu và Ứng Dụng Trong Mô Hình Ngôn Ngữ](aero_LLM_12_Training embeddings from scratch.md) | [Xem bài viết →](aero_LLM_12_Training embeddings from scratch.md) |
| [Thiết Kế Data Loader Cho Huấn Luyện Mô Hình Ngôn Ngữ: Cơ Sở Toán Học, Kiến Trúc và Tối Ưu Hoá](aero_LLM_13_Create a data loader to train a model.md) | [Xem bài viết →](aero_LLM_13_Create a data loader to train a model.md) |
| [Xây Dựng Mô Hình Học Embedding Từ Đầu: Kiến Trúc, Tối Ưu Hoá và Phân Tích Toán Học](aero_LLM_14_Build a model to learn the embeddings.md) | [Xem bài viết →](aero_LLM_14_Build a model to learn the embeddings.md) |
| 📌 **[Hàm Mất Mát Trong Huấn Luyện Embedding: Cơ Sở Lý Thuyết, Phân Tích Gradient và Ứng Dụng Trong Mô Hình Ngôn Ngữ](aero_LLM_15_Loss function to train the embeddings.md)** | [Xem bài viết →](aero_LLM_15_Loss function to train the embeddings.md) |
| [Huấn luyện và Đánh giá Mô hình Học Máy: Cơ sở Lý thuyết và Thực tiễn](aero_LLM_16_Train and evaluate the model.md) | [Xem bài viết →](aero_LLM_16_Train and evaluate the model.md) |
| [Sự Thay Đổi của Embeddings Trong Quá Trình Huấn Luyện: Phân Tích Toán Học và Thực Nghiệm](aero_LLM_17_CodeChallenge How the embeddings change.md) | [Xem bài viết →](aero_LLM_17_CodeChallenge How the embeddings change.md) |
| [Độ Ổn Định của Embeddings trong Mô Hình Ngôn Ngữ: Phân Tích Toán Học và Thực Nghiệm](aero_LLM_18_CodeChallenge How stable are embeddings.md) | [Xem bài viết →](aero_LLM_18_CodeChallenge How stable are embeddings.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
