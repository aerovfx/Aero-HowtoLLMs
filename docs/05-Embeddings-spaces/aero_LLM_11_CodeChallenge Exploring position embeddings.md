# Phân Tích Thực Nghiệm Embedding Vị Trí Trong Transformer: Từ Cấu Trúc Tuyến Tính Đến Không Gian Hình Học

Tóm tắt

Embedding vị trí (positional embeddings) là thành phần cốt lõi giúp mô hình Transformer xử lý thông tin thứ tự trong chuỗi. Bài viết này phân tích sâu embedding vị trí học được (learned positional embeddings), tập trung vào cấu trúc hình học, tính tuyến tính, độ tương đồng cosine và phân tích thành phần chính (PCA). Thực nghiệm được đặt trong bối cảnh mô hình GPT-2 do OpenAI phát triển, dựa trên kiến trúc Transformer từ công trình Attention Is All You Need của Ashish Vaswani và cộng sự.

⸻

1. Giới thiệu

Trong Transformer, embedding của một token tại vị trí t được biểu diễn bởi:

\mathbf{z}_t = \mathbf{e}_t + \mathbf{p}_t

Trong đó:
	•	\mathbf{e}_t \in \mathbb{R}^d: embedding ngữ nghĩa của token
	•	\mathbf{p}_t \in \mathbb{R}^d: embedding vị trí
	•	d: số chiều embedding

Vấn đề cốt lõi: self-attention là bất biến theo hoán vị (permutation invariant). Nếu không có embedding vị trí, mô hình không phân biệt được:
	•	“A B C”
	•	“C B A”

Do đó, embedding vị trí cung cấp cấu trúc thứ tự cho mô hình.

⸻

2. Embedding vị trí trong Transformer

2.1 Embedding vị trí hình sin–cosin (gốc)

Trong Transformer ban đầu:

\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)

\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)

Tính chất quan trọng:
	•	Tạo ra phổ tần số đa dạng
	•	Cho phép biểu diễn quan hệ tuyến tính giữa các vị trí
	•	Không cần tham số học thêm

⸻

2.2 Embedding vị trí học được (GPT-2)

Trong GPT-2, embedding vị trí được học như một ma trận tham số:

\mathbf{P} \in \mathbb{R}^{L \times d}

Với:
	•	L: chiều dài tối đa chuỗi
	•	d: số chiều embedding

Vector vị trí tại t:

\mathbf{p}_t = \mathbf{P}[t]

Các vector này được tối ưu thông qua gradient descent:

\mathbf{P} \leftarrow \mathbf{P} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{P}}

Trong đó:
	•	\eta: learning rate
	•	\mathcal{L}: hàm mất mát

⸻

3. Phân tích hình học của embedding vị trí

3.1 Chuẩn vector (Vector Norm)

Chuẩn L2 của embedding vị trí:

\|\mathbf{p}_t\|_2 = \sqrt{\sum_{i=1}^{d} p_{t,i}^2}

Quan sát thực nghiệm:
	•	Chuẩn tương đối ổn định theo vị trí
	•	Không có sự bùng nổ norm ở cuối chuỗi

Điều này giúp đảm bảo embedding vị trí không lấn át embedding token.

⸻

3.2 Độ tương đồng cosine

Độ tương đồng cosine giữa hai vị trí:

\cos(\theta) =
\frac{\mathbf{p}_t \cdot \mathbf{p}_s}
{\|\mathbf{p}_t\| \|\mathbf{p}_s\|}

Tính chất thực nghiệm:
	•	\cos(\mathbf{p}_t, \mathbf{p}_{t+1}) cao
	•	Giảm dần khi khoảng cách |t-s| tăng
	•	Tạo cấu trúc liên tục (smooth manifold)

Có thể mô hình hoá xấp xỉ:

\cos(\mathbf{p}_t, \mathbf{p}_{t+k}) \approx e^{-\alpha k}

với \alpha > 0.

⸻

4. Phân tích sai phân (Difference Vectors)

Xét vector sai phân:

\Delta_t = \mathbf{p}_{t+1} - \mathbf{p}_t

Nếu embedding có cấu trúc tuyến tính, ta kỳ vọng:

\Delta_t \approx \Delta_{t+1}

Thực nghiệm cho thấy:
	•	Các \Delta_t gần song song nhau
	•	Embedding vị trí gần như nằm trên một quỹ đạo tuyến tính trong không gian \mathbb{R}^d

Điều này gợi ý:

\mathbf{p}_t \approx \mathbf{p}_0 + t\mathbf{v}

với \mathbf{v} là vector hướng chính.

⸻

5. Phân tích thành phần chính (PCA)

5.1 Ma trận hiệp phương sai

\mathbf{C} =
\frac{1}{L} \sum_{t=1}^{L}
(\mathbf{p}_t - \bar{\mathbf{p}})
(\mathbf{p}_t - \bar{\mathbf{p}})^T

Trong đó:

\bar{\mathbf{p}} = \frac{1}{L} \sum_{t=1}^{L} \mathbf{p}_t

Giải bài toán trị riêng:

\mathbf{C}\mathbf{v}_i = \lambda_i \mathbf{v}_i

5.2 Kết quả thực nghiệm
	•	Thành phần chính thứ nhất (PC1) tương quan mạnh với chỉ số vị trí.
	•	Hơn 80% phương sai có thể nằm trong vài thành phần đầu.
	•	Cho thấy embedding vị trí có cấu trúc thấp chiều hiệu quả.

⸻

6. Ảnh hưởng đến Self-Attention

Self-attention:

\text{Attention}(Q,K,V) =
\text{softmax}\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V

Với:

Q = ZW_Q, \quad
K = ZW_K, \quad
Z = E + P

Suy ra:

QK^T =
(E + P)W_QW_K^T(E + P)^T

Khai triển:

= EWE^T + PWP^T + EWP^T + PWE^T

Embedding vị trí đóng góp trực tiếp vào ma trận attention scores.

⸻

7. So sánh với các phương pháp khác

Ngoài absolute positional embeddings, các phương pháp khác:
	•	Relative positional encoding
	•	RoPE (Rotary Positional Embedding)
	•	ALiBi

Các phương pháp này mã hoá khoảng cách tương đối thay vì vị trí tuyệt đối, giúp cải thiện khả năng ngoại suy sang chuỗi dài.

⸻

8. Thảo luận

8.1 Cấu trúc gần tuyến tính

Embedding vị trí học được trong GPT-2 có đặc tính:

\mathbf{p}_t \approx \mathbf{a} + t\mathbf{b} + \epsilon_t

với nhiễu nhỏ \epsilon_t.

Điều này giải thích vì sao PCA thu được trục chính gần tương ứng với chỉ số vị trí.

⸻

8.2 Không gian hình học mượt

Embedding vị trí không phân bố ngẫu nhiên mà tạo thành một đường cong mượt trong không gian cao chiều.

Điều này cho phép:
	•	Attention học được quan hệ khoảng cách
	•	Tăng tính ổn định khi huấn luyện

⸻

9. Kết luận

Qua phân tích lý thuyết và thực nghiệm, có thể rút ra:
	1.	Embedding vị trí học được có cấu trúc gần tuyến tính.
	2.	Độ tương đồng cosine giảm dần theo khoảng cách.
	3.	Không gian embedding có cấu trúc thấp chiều hiệu quả.
	4.	Embedding vị trí ảnh hưởng trực tiếp đến ma trận attention.

Việc hiểu rõ cấu trúc hình học của embedding vị trí có thể mở đường cho:
	•	Thiết kế kiến trúc hiệu quả hơn
	•	Cải thiện khả năng ngoại suy
	•	Tối ưu hoá bộ nhớ cho mô hình ngôn ngữ lớn

⸻

Tài liệu tham khảo
	1.	Ashish Vaswani et al. (2017). Attention Is All You Need.
	2.	OpenAI (2019). GPT-2 Technical Report.
	3.	Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	4.	Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
	5.	Press et al. (2022). ALiBi: Linear Biases for Transformers.
