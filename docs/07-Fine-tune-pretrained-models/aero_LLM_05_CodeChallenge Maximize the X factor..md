
# Tinh Chỉnh Mô Hình GPT-2 Bằng Hàm Mất Mát KL Divergence Để Tối Ưu Hóa Việc Sinh Token Chứa Ký Tự “X”

---

## Tóm tắt

Tinh chỉnh mô hình ngôn ngữ lớn (Large Language Models – LLMs) là một hướng tiếp cận quan trọng nhằm điều chỉnh hành vi sinh văn bản theo mục tiêu cụ thể. Bài báo này trình bày phương pháp tinh chỉnh mô hình GPT-2 Medium thông qua việc xây dựng hàm mất mát tùy chỉnh dựa trên độ đo KL Divergence nhằm gia tăng xác suất sinh các token chứa ký tự “X”. Nghiên cứu tập trung vào việc phân tích kiến trúc mô hình, đặc trưng đầu ra, chuyển đổi logit sang phân phối xác suất, thiết kế hàm mất mát và đánh giá tác động của siêu tham số học. Kết quả cho thấy việc lựa chọn tốc độ học có ảnh hưởng quyết định đến hiện tượng quá khớp và chất lượng sinh văn bản. 

---

## Từ khóa

GPT-2, Fine-tuning, KL Divergence, Language Modeling, Custom Loss Function, Token Optimization

---

## 1. Giới thiệu

Các mô hình ngôn ngữ dựa trên Transformer đã đạt được nhiều thành tựu trong lĩnh vực xử lý ngôn ngữ tự nhiên. Trong đó, GPT-2 là một mô hình sinh văn bản tự hồi quy nổi bật.

Bên cạnh việc huấn luyện chuẩn trên dữ liệu lớn, tinh chỉnh mô hình với mục tiêu đặc biệt là một hướng nghiên cứu quan trọng. Bài toán trong nghiên cứu này nhằm huấn luyện GPT-2 sinh ra nhiều token chứa ký tự “X” thông qua một hàm mất mát được thiết kế riêng. 

---

## 2. Cơ sở lý thuyết

### 2.1 Mô hình GPT-2

GPT-2 là mô hình Transformer một chiều với kiến trúc tự hồi quy. Xác suất sinh chuỗi từ (x_1, x_2, ..., x_T) được mô hình hóa bởi:

[
P(x_1, ..., x_T)=\prod_{t=1}^{T} P(x_t|x_1,...,x_{t-1})
]

Mỗi bước sinh token phụ thuộc vào toàn bộ ngữ cảnh trước đó.

---

### 2.2 Biểu diễn Logit và Softmax

Đầu ra của mô hình tại thời điểm (t) là vector logit:

[
\mathbf{z}_t = (z_1, z_2, ..., z_V)
]

với (V) là kích thước từ vựng.

Xác suất được tính bằng hàm Softmax:

[
P(i|t)=\frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
]

Log-probability:

[
\log P(i|t)= z_i - \log\left(\sum_{j=1}^{V} e^{z_j}\right)
]

---

### 2.3 Độ đo KL Divergence

KL Divergence đo khoảng cách giữa hai phân phối xác suất (P) và (Q):

[
D_{KL}(P||Q)=\sum_{i} P(i)\log\frac{P(i)}{Q(i)}
]

Trong nghiên cứu này:

* (P): phân phối mục tiêu (ưu tiên token chứa “X”)
* (Q): phân phối dự đoán của mô hình

---

## 3. Phương pháp nghiên cứu

### 3.1 Kiến trúc mô hình

Mô hình được sử dụng là GPT-2 Medium với:

* Số block Transformer: 24
* Embedding dimension: 1024
* Vocabulary size: 50,257

Cấu trúc mỗi block gồm:

* Multi-head Attention
* MLP (4× expansion)
* Layer Normalization



---

### 3.2 Phân tích đầu ra mô hình

Đầu ra của mô hình có dạng tensor:

[
O \in \mathbb{R}^{B \times T \times V}
]

Trong đó:

* (B): Batch size
* (T): Sequence length
* (V): Vocabulary size

Ví dụ:

[
O \in \mathbb{R}^{4 \times 64 \times 50257}
]

---

### 3.3 Kiểm tra phân phối đầu ra

Tổng xác suất:

[
\sum_{i=1}^{V} P_i \neq 1
]

Suy ra đầu ra ban đầu là logit thô.

Sau khi áp dụng:

[
\text{LogSoftmax}(z_i)=\log\frac{e^{z_i}}{\sum_j e^{z_j}}
]

Mới thu được phân phối hợp lệ.



---

### 3.4 Biến đổi dữ liệu

Tensor 3 chiều được reshape thành:

[
\mathbb{R}^{(B \times T) \times V}
]

Cụ thể:

[
4 \times 64 \times 50257 \rightarrow 256 \times 50257
]

Nhằm phù hợp với hàm mất mát KL.

---

### 3.5 Hàm mất mát tùy chỉnh

Hàm mất mát được thiết kế như sau:

[
\mathcal{L} = D_{KL}(P_{target}||Q_{model})
]

Trong đó:

[
P_{target}(i)=
\begin{cases}
\alpha & \text{nếu token chứa "X"} \
\beta & \text{ngược lại}
\end{cases}
]

với (\alpha > \beta).

Mục tiêu là tăng xác suất token chứa “X”.



---

### 3.6 Quy trình huấn luyện

Mỗi vòng huấn luyện gồm:

1. Sinh token ngẫu nhiên
2. Forward pass
3. LogSoftmax
4. Tính KL loss
5. Backpropagation
6. Cập nhật tham số

Công thức cập nhật:

[
\theta_{t+1}=\theta_t - \eta\nabla_\theta \mathcal{L}
]

với (\eta) là learning rate.

---

## 4. Thực nghiệm

### 4.1 Thiết lập

| Tham số         | Giá trị           |
| --------------- | ----------------- |
| Batch size      | 4                 |
| Sequence length | 64                |
| Epochs          | 300               |
| Optimizer       | Adam              |
| Learning rate   | (10^{-6},10^{-4}) |



---

### 4.2 Ảnh hưởng của Learning Rate

#### Trường hợp (\eta = 10^{-6})

* Loss giảm: 6 → 2
* Ít token chứa “X”
* Văn bản còn tự nhiên

#### Trường hợp (\eta = 10^{-4})

* Loss → 0.001
* 100% token chứa “X”
* Văn bản vô nghĩa

Hiện tượng overfitting rõ rệt.

---

### 4.3 Đánh giá kết quả

Chỉ số đánh giá:

[
R = \frac{Số\ token\ chứa\ X}{Tổng\ token}
]

Khi (\eta=10^{-4}):

[
R \approx 1
]

Cho thấy mô hình bị chi phối hoàn toàn bởi mục tiêu phụ.



---

## 5. Thảo luận

### 5.1 Ưu điểm

* Linh hoạt điều chỉnh hành vi mô hình
* Không cần huấn luyện lại từ đầu
* Dễ mở rộng sang mục tiêu khác

---

### 5.2 Hạn chế

* Dễ quá khớp
* Mất tính tự nhiên
* Nhạy cảm với siêu tham số
* Khó cân bằng nhiều mục tiêu

Fine-tuning đòi hỏi nhiều thử nghiệm thực tế. 

---

## 6. Kết luận

Nghiên cứu đã trình bày phương pháp tinh chỉnh GPT-2 bằng hàm mất mát KL nhằm tối ưu hóa việc sinh token chứa “X”. Kết quả cho thấy learning rate là yếu tố quyết định đến hiệu quả và độ ổn định của mô hình.

Hướng phát triển tương lai:

* Multi-objective fine-tuning
* Reinforcement Learning from Human Feedback
* Regularization nâng cao
* Human-in-the-loop training

---

## Tài liệu tham khảo

1. Code Challenge: *Maximize the X Factor*, “5 - CodeChallenge Maximize the X factor.txt”. 
2. Vaswani et al. (2017). *Attention Is All You Need*.
3. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners*.
4. Goodfellow et al. (2016). *Deep Learning*.

---
