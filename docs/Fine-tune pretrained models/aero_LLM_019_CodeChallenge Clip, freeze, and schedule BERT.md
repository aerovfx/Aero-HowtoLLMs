
# 📘 Kết Hợp Gradient Clipping, Freezing và Learning Rate Scheduler Trong Fine-Tuning Mô Hình BERT

## Tóm tắt (Abstract)

Fine-tuning các mô hình ngôn ngữ lớn như BERT cho bài toán phân loại văn bản thường gặp các vấn đề về tính ổn định và hội tụ. Ba kỹ thuật quan trọng gồm đóng băng tham số (freezing), cắt gradient (gradient clipping) và điều chỉnh tốc độ học (learning rate scheduler) được đề xuất nhằm cải thiện hiệu suất huấn luyện. Bài viết phân tích cơ sở lý thuyết, mô hình toán học và kết quả thực nghiệm của việc kết hợp ba phương pháp này trong bài toán phân tích cảm xúc đánh giá phim.

---

## 1. Giới thiệu

Các mô hình Transformer tiền huấn luyện như BERT đã trở thành nền tảng trong xử lý ngôn ngữ tự nhiên. Tuy nhiên, quá trình fine-tuning đòi hỏi:

* Kiểm soát số lượng tham số học
* Ổn định gradient
* Điều chỉnh tốc độ hội tụ

Theo tài liệu thực hành , việc kết hợp freezing, clipping và scheduling giúp tăng tính ổn định và hiệu quả huấn luyện.

Mục tiêu nghiên cứu:

* Phân tích vai trò từng kỹ thuật
* Xây dựng mô hình toán học tổng hợp
* Đánh giá tác động lên BERT
* Đề xuất hướng tối ưu

---

## 2. Cơ sở lý thuyết

### 2.1 Fine-tuning mô hình tiền huấn luyện

Cho mô hình tiền huấn luyện với tham số (\theta_0). Fine-tuning nhằm tìm:

[
\theta^*=\arg\min_{\theta}L(\theta;D_{task})
]

Trong đó (D_{task}) là tập dữ liệu mục tiêu.

---

### 2.2 Freezing tham số

Giả sử tập tham số được huấn luyện là (T\subset\theta):

[
\theta=\theta_{freeze}\cup\theta_{train},\quad
\theta_{freeze}\cap\theta_{train}=\emptyset
]

Với:

[
\nabla_{\theta_{freeze}}L=0
]

⇒ các tham số bị đóng băng không cập nhật.

---

### 2.3 Gradient Descent

Quy trình cập nhật:

[
\theta_{t+1}=\theta_t-\eta_t\mathbf{g}_t
]

[
\mathbf{g}*t=\nabla*\theta L(\theta_t)
]

---

## 3. Phương pháp nghiên cứu

### 3.1 Chiến lược Freezing trong BERT

Theo , mô hình được cấu hình:

* Đóng băng: Embedding + Attention
* Huấn luyện: MLP + Pooler + Classifier

Tỷ lệ tham số:

[
R=\frac{|\theta_{train}|}{|\theta_{total}|}\approx 0.5
]

---

### 3.2 Gradient Clipping

#### 3.2.1 Chuẩn hóa gradient

Với ngưỡng (c=1):

[
\mathbf{g}'=
\frac{c}{\max(|\mathbf{g}|,c)}\mathbf{g}
]

Đảm bảo:

[
|\mathbf{g}'|\le c
]

---

#### 3.2.2 Ảnh hưởng tới cập nhật

[
\theta_{t+1}=\theta_t-\eta_t\mathbf{g}'
]

Giúp hạn chế gradient explosion.

---

### 3.3 Learning Rate Scheduler

#### 3.3.1 Warm-up

[
\eta_t=\eta_{max}\frac{t}{T_{warm}},\quad t\le T_{warm}
]

---

#### 3.3.2 Linear Decay

[
\eta_t=\eta_{max}\left(1-\frac{t}{T_{sched}}\right)
]

Trong đó:

[
T_{sched}>T_{train}
]

để tránh (\eta_t=0).

---

### 3.4 Quy trình tổng hợp

Quy trình huấn luyện:

1. Forward
2. Backprop
3. Ghi nhận gradient norm
4. Clipping
5. Scheduler
6. Update

Phương trình tổng quát:

[
\theta_{t+1}=
\theta_t-
\eta_t
\frac{c}{\max(|\mathbf{g}_t|,c)}\mathbf{g}_t
]

---

## 4. Thực nghiệm

### 4.1 Thiết lập

Theo :

* 300 batch huấn luyện
* Warm-up 5%
* Linear scheduler (450 steps)
* Clipping: (c=1)

Theo dõi:

* Loss
* Accuracy
* Gradient norm

---

### 4.2 Phân tích hàm mất mát

Cross-Entropy:

[
L=-\sum_{i=1}^{N}y_i\log(p_i)
]

Quan sát:

[
Var(L_{clip+sch})<Var(L_{baseline})
]

⇒ học ổn định hơn.

---

### 4.3 Độ chính xác

Accuracy:

[
Acc=\frac{TP+TN}{TP+TN+FP+FN}
]

Kết quả:

| Giai đoạn    | Accuracy |
| ------------ | -------- |
| Trước tối ưu | ~85%     |
| Sau tối ưu   | ~90%     |

---

### 4.4 Phân tích Gradient Norm

Hai lớp được theo dõi:

* MLP layer (pre-trained)
* Classifier layer (random)

Chuẩn gradient:

[
G_t=|\nabla W_t|
]

Quan sát:

[
G_{MLP}<1 \quad (\text{đa số})
]

[
G_{CLS}>1 \quad (\text{nhiều giai đoạn đầu})
]

⇒ Clipping ảnh hưởng mạnh đến classifier.

---

### 4.5 Hiện tượng mất thông tin Gradient

Lượng thông tin bị mất:

[
\Delta g=
|\mathbf{g}|-|\mathbf{g}'|
]

Với:

[
|\mathbf{g}|>1
]

⇒ (\Delta g>0)

Đặc biệt lớn ở giai đoạn đầu.

---

## 5. Thảo luận

### 5.1 Đánh giá tính phù hợp của Clipping

Theo , clipping sớm có thể:

* Giảm tốc độ học
* Làm chậm classifier

Giải pháp:

[
c(t)=
\begin{cases}
\infty & t<T_0\
1 & t\ge T_0
\end{cases}
]

(Delayed clipping)

---

### 5.2 Tương tác giữa các kỹ thuật

Ba kỹ thuật phối hợp:

| Kỹ thuật  | Vai trò      |
| --------- | ------------ |
| Freezing  | Giảm tham số |
| Clipping  | Ổn định      |
| Scheduler | Hội tụ       |

Tác động tổng hợp:

[
Stability\propto f(F,C,S)
]

---

### 5.3 Ứng dụng cho LLM

Kết quả cho thấy:

* Cần thiết cho mô hình >1B tham số
* Giảm rủi ro divergence
* Tăng khả năng tái lập

---

## 6. Kết luận

Nghiên cứu đã phân tích việc kết hợp freezing, gradient clipping và learning rate scheduler trong fine-tuning BERT.

Kết quả chính:

* Loss ổn định hơn
* Accuracy tăng
* Gradient được kiểm soát
* Hội tụ nhanh hơn

Phương pháp phù hợp cho huấn luyện mô hình ngôn ngữ lớn trong điều kiện tài nguyên hạn chế.

---

## Tài liệu tham khảo

1. BERT Fine-Tuning Code Challenge 
2. Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
3. Goodfellow, I. et al. (2016). Deep Learning. MIT Press.
4. Loshchilov, I., Hutter, F. (2017). SGDR.

---
