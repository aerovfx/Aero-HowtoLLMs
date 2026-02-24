# Phân Tích Cosine Similarity trong Không Gian Embedding của GPT-2

## Tóm tắt

Cosine similarity là một công cụ trung tâm trong việc phân tích cấu trúc hình học của không gian embedding trong các mô hình ngôn ngữ lớn. Bài viết này trình bày cơ sở toán học và quy trình thực nghiệm để tính toán cosine similarity giữa các token embedding của [GPT-2](chatgpt://generic-entity?number=0), đồng thời phân tích ý nghĩa hình học và thống kê của các giá trị tương đồng thu được.

---

## 1. Giới thiệu

Trong các mô hình Transformer sinh văn bản, mỗi token được ánh xạ sang một vector trong không gian \( \mathbb{R}^d \). Với GPT-2:

- Kích thước embedding: \( d = 768 \) (bản base)
- Từ vựng: khoảng 50.000 token

Mỗi token \( t \) có vector embedding:

\[
\mathbf{v}_t \in \mathbb{R}^{768}
\]

Phân tích cosine similarity giữa các vector này giúp hiểu cấu trúc ngữ nghĩa nội tại của mô hình.

---

## 2. Cơ sở toán học của Cosine Similarity

Cho hai vector:

\[
\mathbf{x}, \mathbf{y} \in \mathbb{R}^d
\]

Định nghĩa:

\[
\text{cosine}(\mathbf{x},\mathbf{y})
=
\frac{\mathbf{x} \cdot \mathbf{y}}
{\|\mathbf{x}\| \|\mathbf{y}\|}
\]

Trong đó:

\[
\mathbf{x} \cdot \mathbf{y}
=
\sum_{i=1}^{d} x_i y_i
\]

\[
\|\mathbf{x}\|
=
\sqrt{\sum_{i=1}^{d} x_i^2}
\]

Giá trị nằm trong khoảng:

\[
-1 \leq \text{cosine} \leq 1
\]

---

## 3. Chuẩn hóa và tính toán hiệu quả

Trong thực tế, ta chuẩn hóa trước:

\[
\hat{\mathbf{x}} =
\frac{\mathbf{x}}{\|\mathbf{x}\|}
\]

Khi đó:

\[
\text{cosine}(\mathbf{x},\mathbf{y})
=
\hat{\mathbf{x}} \cdot \hat{\mathbf{y}}
\]

Nếu ma trận embedding:

\[
E \in \mathbb{R}^{|V| \times d}
\]

Sau khi chuẩn hóa từng hàng:

\[
\hat{E}
\]

Ma trận cosine similarity toàn bộ từ vựng:

\[
S = \hat{E} \hat{E}^T
\]

---

## 4. Phân tích thực nghiệm với GPT-2

### 4.1 Trích xuất embedding

Với token index \( i \):

\[
\mathbf{v}_i = E[i]
\]

Trong GPT-2, embedding đầu vào và embedding đầu ra thường được chia sẻ trọng số (weight tying):

\[
W_{out} = E^T
\]

Điều này tạo liên hệ hình học trực tiếp giữa không gian embedding và không gian dự đoán xác suất.

---

### 4.2 Ví dụ: So sánh token

Giả sử ta chọn token:

- “cat”
- “dog”
- “banana”

Ta tính:

\[
\text{sim}(\text{cat},\text{dog})
\]

\[
\text{sim}(\text{cat},\text{banana})
\]

Kỳ vọng:

\[
\text{sim}(\text{cat},\text{dog})
>
\text{sim}(\text{cat},\text{banana})
\]

Do cấu trúc ngữ nghĩa gần nhau.

---

## 5. Phân bố Cosine Similarity trong không gian cao chiều

Giả sử hai vector ngẫu nhiên:

\[
\mathbf{x},\mathbf{y}
\sim \mathcal{N}(0,I_d)
\]

Khi \( d \to \infty \):

\[
\mathbb{E}[\text{cosine}] = 0
\]

\[
\text{Var}(\text{cosine}) \approx \frac{1}{d}
\]

Với \( d = 768 \):

\[
\text{Var} \approx \frac{1}{768}
\]

Do đó:

- Vector ngẫu nhiên gần trực giao
- Cosine lớn biểu thị cấu trúc học được

---

## 6. Liên hệ với Softmax và xác suất dự đoán

Trong GPT-2, xác suất token tiếp theo:

\[
P(w_t | h_t)
=
\text{softmax}(W_{out} h_t)
\]

Nếu weight tying:

\[
W_{out} = E^T
\]

Khi đó:

\[
z_i = \mathbf{v}_i \cdot h_t
\]

Softmax:

\[
P(w_i)
=
\frac{e^{\mathbf{v}_i \cdot h_t}}
{\sum_j e^{\mathbf{v}_j \cdot h_t}}
\]

Như vậy:

> Dự đoán xác suất thực chất dựa trên tích vô hướng giữa embedding và hidden state.

Nếu chuẩn hóa:

\[
\mathbf{v}_i \cdot h_t
=
\|\mathbf{v}_i\|
\|h_t\|
\cos\theta
\]

Do đó cosine similarity trực tiếp ảnh hưởng đến xác suất dự đoán.

---

## 7. Ma trận tương đồng cục bộ

Cho tập \( n \) token:

\[
X \in \mathbb{R}^{n \times d}
\]

Ma trận cosine:

\[
S_{ij}
=
\frac{\mathbf{v}_i \cdot \mathbf{v}_j}
{\|\mathbf{v}_i\|\|\mathbf{v}_j\|}
\]

Ta có thể phân tích:

- Cụm từ (clustering)
- Phân tích trị riêng:

\[
S \mathbf{u} = \lambda \mathbf{u}
\]

Giá trị riêng lớn phản ánh cấu trúc ngữ nghĩa chiếm ưu thế.

---

## 8. Khoảng cách tương đương

Nếu vector đã chuẩn hóa:

\[
\|\mathbf{x}-\mathbf{y}\|^2
=
2 - 2\cos\theta
\]

Suy ra:

\[
\cos\theta
=
1 - \frac{1}{2}
\|\mathbf{x}-\mathbf{y}\|^2
\]

Điều này cho thấy cosine similarity và Euclid distance tương đương về mặt hình học khi chuẩn hóa.

---

## 9. Ý nghĩa lý thuyết

Cosine similarity trong GPT-2:

1. Định nghĩa cấu trúc hình học của từ vựng.
2. Liên hệ trực tiếp với xác suất dự đoán.
3. Phản ánh cấu trúc phân bố dữ liệu huấn luyện.
4. Giảm ảnh hưởng của độ lớn vector.

Về bản chất:

\[
\text{Prediction}
\propto
\exp(\|\mathbf{v}\|\|h\|\cos\theta)
\]

Do đó góc giữa vector đóng vai trò quyết định.

---

## 10. Kết luận

Phân tích cosine similarity trong GPT-2 cho thấy:

- Không gian embedding có cấu trúc hình học rõ ràng.
- Các token liên quan có góc nhỏ (cosine lớn).
- Dự đoán xác suất phụ thuộc trực tiếp vào tích vô hướng.
- Trong không gian cao chiều, cấu trúc học được nổi bật hơn nền ngẫu nhiên.

Hiểu rõ nền tảng toán học này giúp ta:

- Phân tích embedding hiệu quả
- So sánh mô hình
- Thực hiện Representational Similarity Analysis (RSA)
- Tối ưu hóa hệ thống retrieval hoặc semantic search

---

## Tài liệu tham khảo

1. Vaswani et al. (2017). Attention is All You Need.  
2. Radford et al. (2019). Language Models are Unsupervised Multitask Learners.  
3. Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.  
4. Jurafsky & Martin (2023). Speech and Language Processing.  
5. Kriegeskorte et al. (2008). Representational Similarity Analysis.

---