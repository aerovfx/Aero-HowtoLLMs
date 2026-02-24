# Thiết Kế Data Loader Cho Huấn Luyện Mô Hình Ngôn Ngữ: Cơ Sở Toán Học, Kiến Trúc và Tối Ưu Hoá

Tóm tắt

Trong huấn luyện mô hình ngôn ngữ hiện đại, đặc biệt là các kiến trúc Transformer, data loader đóng vai trò trung gian quan trọng giữa dữ liệu thô và quá trình tối ưu hoá tham số. Bài viết này trình bày cơ sở lý thuyết và thực nghiệm của việc xây dựng data loader cho bài toán dự đoán token tiếp theo (next-token prediction), bao gồm tokenization, batching, tạo cặp (input, target), xử lý chuỗi dài và tối ưu hiệu suất. Phân tích được đặt trong bối cảnh kiến trúc Transformer từ công trình Attention Is All You Need của Ashish Vaswani và ứng dụng trong các mô hình như GPT-2 của OpenAI.

⸻

1. Giới thiệu

Huấn luyện mô hình ngôn ngữ tự hồi quy (autoregressive language model) yêu cầu tối ưu xác suất:

P(x_1, x_2, \dots, x_T)
=
\prod_{t=1}^{T}
P(x_t \mid x_{<t})

Trong đó:
	•	x_t: token tại vị trí t
	•	x_{<t}: toàn bộ ngữ cảnh trước đó

Data loader có nhiệm vụ:
	1.	Chuyển văn bản thành chuỗi token.
	2.	Chia thành các đoạn có độ dài cố định.
	3.	Tạo cặp (input, target) cho huấn luyện.
	4.	Cung cấp batch tối ưu cho GPU/TPU.

⸻

2. Biểu diễn dữ liệu cho huấn luyện

2.1 Tokenization

Giả sử văn bản sau khi token hóa:

\mathbf{s} = (t_1, t_2, \dots, t_N)

với:

t_i \in \{1,2,\dots,V\}
	•	V: kích thước từ vựng
	•	N: tổng số token

⸻

2.2 Tạo cặp (Input, Target)

Với độ dài ngữ cảnh cố định L, ta tạo:

\mathbf{x}^{(i)} =
(t_i, t_{i+1}, \dots, t_{i+L-1})

\mathbf{y}^{(i)} =
(t_{i+1}, t_{i+2}, \dots, t_{i+L})

Tức là target là phiên bản dịch trái của input.

Mục tiêu tối ưu:

\mathcal{L}
=
-
\sum_{t=1}^{L}
\log
P(t_{i+t} \mid t_i,\dots,t_{i+t-1})

⸻

3. Batch và Tối Ưu Tính Toán

3.1 Mini-batch

Với batch size B, ta có tensor:

X \in \mathbb{R}^{B \times L}

Y \in \mathbb{R}^{B \times L}

Loss trung bình:

\mathcal{L}_{batch}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathcal{L}^{(b)}

⸻

3.2 Phân tích độ phức tạp

Giả sử:
	•	Vocabulary size: V
	•	Embedding dimension: d
	•	Context length: L
	•	Batch size: B

Chi phí embedding:

O(BLd)

Chi phí attention:

O(BL^2 d)

Do đó, data loader phải đảm bảo cung cấp batch đủ lớn nhưng không vượt quá bộ nhớ GPU.

⸻

4. Chiến lược chia dữ liệu

4.1 Sliding Window

Tạo các mẫu huấn luyện với bước trượt 1:

(t_1,\dots,t_L),
(t_2,\dots,t_{L+1}),
\dots

Ưu điểm:
	•	Tận dụng tối đa dữ liệu

Nhược điểm:
	•	Tính toán trùng lặp

⸻

4.2 Chunking (chia đoạn không chồng lấn)

Chia thành các đoạn độc lập:

(t_1,\dots,t_L),
(t_{L+1},\dots,t_{2L})

Ưu điểm:
	•	Nhanh
	•	Giảm trùng lặp

Nhược điểm:
	•	Giảm số lượng mẫu

⸻

5. Tối ưu hoá bộ nhớ

5.1 Memory Mapping

Với tập dữ liệu lớn (hàng tỷ token), ta lưu dưới dạng mảng nhị phân:

\mathbf{D} \in \mathbb{N}^{N}

Sử dụng memory-mapped file:

\text{mmap}: \mathbb{N}^{N} \rightarrow \text{RAM (lazy loading)}

Điều này cho phép:
	•	Không load toàn bộ vào RAM
	•	Truy cập ngẫu nhiên hiệu quả

⸻

5.2 Shuffling

Trong huấn luyện SGD:

\theta \leftarrow
\theta - \eta \nabla_\theta \mathcal{L}(x_i)

Để đảm bảo ước lượng không chệch:

\mathbb{E}[\nabla_\theta \mathcal{L}_{batch}]
=
\nabla_\theta \mathcal{L}_{true}

Cần xáo trộn dữ liệu ngẫu nhiên.

⸻

6. Data Loader trong Huấn Luyện Transformer

Pipeline tổng quát:
	1.	Raw text
	2.	Tokenization
	3.	Lưu thành mảng số nguyên
	4.	Random sampling các đoạn dài L
	5.	Ghép batch
	6.	Chuyển sang GPU

Trong mô hình như GPT-2:

\mathbf{z}_t
=
\mathbf{e}_t
+
\mathbf{p}_t

Sau đó đi vào self-attention:

\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V

⸻

7. Ảnh hưởng của Data Loader đến Hội Tụ

Giả sử gradient ước lượng:

g_t = \nabla_\theta \mathcal{L}_{batch}

Phương sai:

\text{Var}(g_t)
=
\frac{\sigma^2}{B}

Batch lớn:
	•	Giảm phương sai
	•	Hội tụ ổn định

Batch nhỏ:
	•	Nhiễu cao
	•	Có thể tổng quát tốt hơn

⸻

8. Các vấn đề nâng cao

8.1 Curriculum Learning

Sắp xếp dữ liệu theo độ khó:

\mathcal{D}_1 \subset \mathcal{D}_2 \subset \dots

Giúp hội tụ nhanh hơn.

⸻

8.2 Packing Sequences

Khi chuỗi ngắn hơn L, có thể ghép nhiều chuỗi vào một block để tăng hiệu suất GPU.

⸻

8.3 Distributed Data Loading

Với K GPU:

\mathcal{D}
=
\bigcup_{k=1}^{K}
\mathcal{D}_k

Mỗi GPU xử lý phần riêng, đảm bảo không trùng lặp.

⸻

9. Kết luận

Data loader không chỉ là thành phần phụ trợ mà là yếu tố quyết định hiệu suất và độ ổn định của huấn luyện mô hình ngôn ngữ.

Các điểm chính:
	1.	Phải xây dựng cặp (input, target) chính xác cho bài toán autoregressive.
	2.	Tối ưu batch để cân bằng bộ nhớ và tốc độ.
	3.	Xử lý dữ liệu quy mô lớn bằng memory mapping.
	4.	Shuffling đảm bảo gradient không chệch.
	5.	Thiết kế data loader ảnh hưởng trực tiếp đến hội tụ.

Trong bối cảnh mô hình Transformer hiện đại, tối ưu hoá data pipeline quan trọng không kém tối ưu kiến trúc.

⸻

Tài liệu tham khảo
	1.	Ashish Vaswani et al. (2017). Attention Is All You Need.
	2.	OpenAI (2019). GPT-2 Technical Report.
	3.	Goodfellow et al. (2016). Deep Learning.
	4.	Bottou (2010). Large-Scale Machine Learning with SGD.

