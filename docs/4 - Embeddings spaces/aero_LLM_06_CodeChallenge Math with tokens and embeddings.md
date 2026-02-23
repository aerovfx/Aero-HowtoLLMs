# Toán học của Token và Embedding trong Mô hình Ngôn ngữ Lớn

## Tóm tắt

Trong các mô hình ngôn ngữ hiện đại như [GPT-2](chatgpt://generic-entity?number=0) và [BERT](chatgpt://generic-entity?number=1), văn bản không được xử lý trực tiếp dưới dạng chữ mà được chuyển đổi thành **token** và sau đó ánh xạ sang không gian vector thông qua **embedding**. Bài viết này phân tích nền tảng toán học của quá trình token hóa, ánh xạ embedding, cũng như các phép toán đại số vector cho phép mô hình học được cấu trúc ngữ nghĩa.

---

## 1. Từ văn bản đến token

Giả sử ta có chuỗi văn bản:

\[
\mathcal{T} = (w_1, w_2, ..., w_n)
\]

Bộ tokenizer thực hiện ánh xạ:

\[
\tau: \mathcal{V}_{text} \rightarrow \mathcal{V}_{token}
\]

Trong đó:

- \( \mathcal{V}_{text} \): tập từ tự nhiên
- \( \mathcal{V}_{token} \): tập token rời rạc

Kết quả là dãy chỉ số:

\[
(t_1, t_2, ..., t_n), \quad t_i \in \{1,2,...,|V|\}
\]

---

## 2. One-hot Encoding

Mỗi token \( t_i \) được biểu diễn ban đầu dưới dạng vector one-hot:

\[
\mathbf{x}_i \in \mathbb{R}^{|V|}
\]

\[
x_{ij} =
\begin{cases}
1 & \text{nếu } j = t_i \\
0 & \text{ngược lại}
\end{cases}
\]

Đây là không gian rất cao chiều và không hiệu quả về mặt tính toán.

---

## 3. Ma trận Embedding

Ta định nghĩa ma trận embedding:

\[
E \in \mathbb{R}^{|V| \times d}
\]

Trong đó:

- \( |V| \): kích thước từ vựng
- \( d \): số chiều embedding

Vector embedding được tính:

\[
\mathbf{v}_i = \mathbf{x}_i E
\]

Do \( \mathbf{x}_i \) là one-hot, nên:

\[
\mathbf{v}_i = E_{t_i}
\]

Tức là lấy hàng thứ \( t_i \) của ma trận embedding.

---

## 4. Cộng embedding và vị trí (Positional Encoding)

Trong Transformer, embedding cuối cùng là tổng của:

\[
\mathbf{z}_i = \mathbf{v}_i + \mathbf{p}_i
\]

Trong đó \( \mathbf{p}_i \) là positional encoding:

\[
PE_{(pos,2k)} = \sin\left(\frac{pos}{10000^{2k/d}}\right)
\]

\[
PE_{(pos,2k+1)} = \cos\left(\frac{pos}{10000^{2k/d}}\right)
\]

Điều này giúp mô hình nhận biết thứ tự chuỗi.

---

## 5. Đại số vector trong không gian embedding

### 5.1 Độ tương đồng Cosine

\[
\text{cosine}(\mathbf{v}_i,\mathbf{v}_j)
=
\frac{\mathbf{v}_i \cdot \mathbf{v}_j}
{\|\mathbf{v}_i\|\|\mathbf{v}_j\|}
\]

Phản ánh mức độ tương đồng ngữ nghĩa.

---

### 5.2 Phép cộng ngữ nghĩa

Trong nhiều mô hình, quan hệ tuyến tính có thể xuất hiện:

\[
\mathbf{v}_{king} - \mathbf{v}_{man}
+ \mathbf{v}_{woman}
\approx
\mathbf{v}_{queen}
\]

Điều này cho thấy không gian embedding học được cấu trúc ngữ nghĩa tuyến tính.

---

## 6. Tối ưu hóa Embedding

Trong mô hình tự hồi quy như GPT-2, hàm mất mát là:

\[
\mathcal{L}
=
- \sum_{t=1}^{T}
\log P(w_t | w_{<t})
\]

Với:

\[
P(w_t | w_{<t})
=
\text{softmax}(W_o h_t)
\]

\[
\text{softmax}(z_i)
=
\frac{e^{z_i}}
{\sum_{j=1}^{|V|} e^{z_j}}
\]

Gradient lan truyền ngược để cập nhật ma trận embedding:

\[
E \leftarrow E - \eta \nabla_E \mathcal{L}
\]

Trong đó \( \eta \) là learning rate.

---

## 7. Hình học của không gian embedding

Giả sử:

\[
X \in \mathbb{R}^{n \times d}
\]

Ma trận hiệp phương sai:

\[
\Sigma = \frac{1}{n} X^T X
\]

Giải bài toán trị riêng:

\[
\Sigma \mathbf{u} = \lambda \mathbf{u}
\]

Các trị riêng lớn cho biết chiều chiếm ưu thế của không gian ngữ nghĩa.

---

## 8. Chuẩn hóa và ổn định số học

Thường áp dụng chuẩn hóa:

\[
\hat{\mathbf{v}} =
\frac{\mathbf{v}}{\|\mathbf{v}\|}
\]

Điều này làm:

\[
\|\hat{\mathbf{v}}\| = 1
\]

Giúp tăng ổn định khi tính attention và cosine similarity.

---

## 9. Từ token đến Attention

Self-attention tính:

\[
Q = XW_Q
\]
\[
K = XW_K
\]
\[
V = XW_V
\]

\[
\text{Attention}(Q,K,V)
=
\text{softmax}\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

Embedding ban đầu đóng vai trò nền tảng cho toàn bộ phép biến đổi này.

---

## 10. Kết luận

Quá trình từ văn bản đến embedding có thể tóm tắt:

\[
\text{Text}
\rightarrow
\text{Token}
\rightarrow
\text{One-hot}
\rightarrow
\text{Embedding}
\rightarrow
\text{Attention}
\rightarrow
\text{Contextual Representation}
\]

Về mặt toán học:

- Token là biến rời rạc.
- Embedding là ánh xạ tuyến tính sang không gian liên tục.
- Attention là phép biến đổi phi tuyến phụ thuộc ngữ cảnh.
- Toàn bộ hệ thống được tối ưu hóa thông qua gradient descent.

Hiểu rõ cấu trúc toán học này giúp giải thích vì sao các mô hình ngôn ngữ lớn có thể học được cấu trúc ngữ nghĩa phức tạp từ dữ liệu văn bản khổng lồ.

---

## Tài liệu tham khảo

1. Vaswani et al. (2017). Attention is All You Need.  
2. Radford et al. (2019). Language Models are Unsupervised Multitask Learners.  
3. Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.  
4. Jurafsky & Martin (2023). Speech and Language Processing.

---