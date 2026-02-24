# Huấn Luyện Embedding Từ Đầu: Cơ Sở Toán Học, Cơ Chế Tối Ưu và Ứng Dụng Trong Mô Hình Ngôn Ngữ

Tóm tắt

Embedding là nền tảng của các mô hình ngôn ngữ hiện đại, cho phép ánh xạ token rời rạc sang không gian vector liên tục. Bài viết này trình bày quy trình huấn luyện embedding từ đầu (training embeddings from scratch), phân tích cơ sở toán học của hàm mất mát, lan truyền ngược (backpropagation), tối ưu hoá, và mối liên hệ với các mô hình như Word2Vec và Transformer. Đồng thời, bài viết mở rộng thảo luận sang embedding trong các mô hình như GPT-2 của OpenAI và nền tảng self-attention từ công trình Attention Is All You Need của Ashish Vaswani và cộng sự.

⸻

1. Giới thiệu

Trong xử lý ngôn ngữ tự nhiên, mỗi token ban đầu được biểu diễn dưới dạng chỉ số rời rạc:

w \in \{1,2,\dots,V\}

Trong đó:
	•	V: kích thước từ vựng (vocabulary size)

Embedding ánh xạ token sang không gian liên tục:

f: \{1,\dots,V\} \rightarrow \mathbb{R}^d

Ma trận embedding:

\mathbf{E} \in \mathbb{R}^{V \times d}

Vector embedding của token w:

\mathbf{e}_w = \mathbf{E}[w]

⸻

2. Huấn luyện embedding như một lớp tuyến tính

2.1 Biểu diễn one-hot

Token w có thể biểu diễn bằng vector one-hot:

\mathbf{x} \in \mathbb{R}^V

với:

x_i =
\begin{cases}
1 & \text{nếu } i = w \\
0 & \text{ngược lại}
\end{cases}

Embedding thực chất là phép nhân ma trận:

\mathbf{e}_w = \mathbf{x}^T \mathbf{E}

Vì \mathbf{x} là one-hot nên phép nhân này tương đương với chọn một hàng trong ma trận.

⸻

3. Huấn luyện embedding trong bài toán dự đoán từ

Giả sử bài toán dự đoán từ tiếp theo (next-token prediction).

3.1 Xác suất Softmax

Logits:

\mathbf{z} = \mathbf{W}\mathbf{e}_w + \mathbf{b}

Xác suất:

P(y=i \mid w) =
\frac{\exp(z_i)}
{\sum_{j=1}^{V} \exp(z_j)}

⸻

3.2 Hàm mất mát Cross-Entropy

\mathcal{L} = -\sum_{i=1}^{V} y_i \log P(y=i)

Với y_i là vector nhãn one-hot.

Do đó:

\mathcal{L} = -\log P(y = y_{\text{true}})

⸻

4. Lan truyền ngược và cập nhật embedding

Gradient theo logits:

\frac{\partial \mathcal{L}}{\partial z_i}
= P(y=i) - y_i

Gradient theo embedding:

\frac{\partial \mathcal{L}}{\partial \mathbf{e}_w}
= \mathbf{W}^T ( \mathbf{p} - \mathbf{y})

Gradient theo ma trận embedding:

\frac{\partial \mathcal{L}}{\partial \mathbf{E}[w]}
= \frac{\partial \mathcal{L}}{\partial \mathbf{e}_w}

Cập nhật bằng gradient descent:

\mathbf{E}[w] \leftarrow
\mathbf{E}[w]
- \eta
\frac{\partial \mathcal{L}}{\partial \mathbf{E}[w]}

Trong đó:
	•	\eta: learning rate

Chỉ hàng tương ứng với token xuất hiện trong batch được cập nhật.

⸻

5. Embedding trong Word2Vec

Trong mô hình Skip-gram của Tomas Mikolov:

Mục tiêu:

\max \sum_{(w,c)}
\log P(c \mid w)

Với:

P(c \mid w)
=
\frac{\exp(\mathbf{u}_c^T \mathbf{v}_w)}
{\sum_{j=1}^{V} \exp(\mathbf{u}_j^T \mathbf{v}_w)}

Trong đó:
	•	\mathbf{v}_w: embedding trung tâm
	•	\mathbf{u}_c: embedding ngữ cảnh

Để giảm chi phí tính toán, Negative Sampling được sử dụng:

\mathcal{L} =
\log \sigma(\mathbf{u}_c^T \mathbf{v}_w)
+
\sum_{k=1}^{K}
\log \sigma(-\mathbf{u}_{n_k}^T \mathbf{v}_w)

⸻

6. Embedding trong Transformer

Trong Transformer:

\mathbf{z}_t =
\mathbf{e}_t + \mathbf{p}_t

Self-attention:

\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V

Với:

Q = ZW_Q, \quad
K = ZW_K, \quad
V = ZW_V

Embedding ảnh hưởng trực tiếp đến attention scores.

⸻

7. Tính chất hình học của embedding

7.1 Chuẩn vector

\|\mathbf{e}_w\|
=
\sqrt{
\sum_{i=1}^{d}
e_{w,i}^2
}

Token phổ biến thường có norm lớn hơn do được cập nhật nhiều lần.

⸻

7.2 Độ tương đồng cosine

\cos(\theta)
=
\frac{
\mathbf{e}_a \cdot \mathbf{e}_b
}{
\|\mathbf{e}_a\|\|\mathbf{e}_b\|
}

Cho phép đo mức độ tương đồng ngữ nghĩa.

Ví dụ quan hệ tuyến tính nổi tiếng:

\mathbf{e}_{\text{king}}
-
\mathbf{e}_{\text{man}}
+
\mathbf{e}_{\text{woman}}
\approx
\mathbf{e}_{\text{queen}}

⸻

8. Phân tích phổ giá trị riêng (Spectral Analysis)

Xét ma trận embedding:

\mathbf{E} \in \mathbb{R}^{V \times d}

Ma trận hiệp phương sai:

\mathbf{C}
=
\frac{1}{V}
\mathbf{E}^T
\mathbf{E}

Giải:

\mathbf{C}\mathbf{v}_i
=
\lambda_i \mathbf{v}_i

Kết quả thực nghiệm:
	•	Phương sai tập trung vào số ít thành phần chính
	•	Embedding có cấu trúc thấp chiều hiệu quả

⸻

9. Vai trò của tối ưu hoá

Các thuật toán tối ưu phổ biến:
	•	SGD
	•	Adam
	•	AdamW

Ví dụ Adam cập nhật:

m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t

v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2

\theta_t =
\theta_{t-1}
-
\eta
\frac{m_t}{\sqrt{v_t}+\epsilon}

Embedding được cập nhật đồng thời với toàn bộ mô hình.

⸻

10. Kết luận

Huấn luyện embedding từ đầu là quá trình:
	1.	Ánh xạ token rời rạc sang không gian liên tục
	2.	Tối ưu thông qua dự đoán ngữ cảnh hoặc token tiếp theo
	3.	Hình thành cấu trúc hình học phản ánh ngữ nghĩa
	4.	Tích hợp trực tiếp vào cơ chế attention trong Transformer

Embedding không chỉ là bảng tra cứu (lookup table), mà là một không gian hình học có cấu trúc, được hình thành thông qua tối ưu hoá thống kê quy mô lớn.

⸻

Tài liệu tham khảo
	1.	Ashish Vaswani et al. (2017). Attention Is All You Need.
	2.	OpenAI (2019). GPT-2 Technical Report.
	3.	Tomas Mikolov et al. (2013). Efficient Estimation of Word Representations in Vector Space.
	4.	Kingma & Ba (2015). Adam: A Method for Stochastic Optimization.

