
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
