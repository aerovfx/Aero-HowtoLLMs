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
