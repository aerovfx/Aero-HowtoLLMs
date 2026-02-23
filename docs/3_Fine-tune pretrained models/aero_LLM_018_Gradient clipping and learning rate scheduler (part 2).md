
# 📘 Phân Tích Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu Quy Mô Lớn

## Tóm tắt (Abstract)

Trong huấn luyện mô hình học sâu hiện đại, đặc biệt là các mô hình ngôn ngữ lớn, việc kiểm soát tốc độ học và độ ổn định số học đóng vai trò then chốt. Learning Rate Scheduler là một kỹ thuật giúp điều chỉnh learning rate theo thời gian nhằm cải thiện khả năng hội tụ và hạn chế dao động. Bài viết trình bày cơ sở lý thuyết, mô hình toán học và kết quả thực nghiệm về các bộ điều chỉnh learning rate phổ biến như Cosine Scheduler và Linear Scheduler dựa trên tài liệu thực hành.

---

## 1. Giới thiệu

Tối ưu hóa trong học sâu chủ yếu dựa trên các thuật toán gradient-based. Tuy nhiên, việc sử dụng learning rate cố định thường gây ra các vấn đề như:

* Hội tụ chậm
* Dao động mạnh
* Dễ mắc kẹt tại điểm tối ưu cục bộ

Theo tài liệu thực nghiệm , Learning Rate Scheduler giúp khắc phục các hạn chế trên thông qua điều chỉnh learning rate động.

Mục tiêu nghiên cứu:

* Phân tích cơ chế hoạt động của scheduler
* Xây dựng mô hình toán học
* Đánh giá tác động đến quá trình học
* So sánh các phương pháp điều chỉnh

---

## 2. Cơ sở lý thuyết

### 2.1 Cập nhật tham số trong học sâu

Quy trình cập nhật tham số:

[
\theta_{t+1}=\theta_t-\eta_t \nabla_\theta L(\theta_t)
]

Trong đó:

* (\eta_t): learning rate tại thời điểm (t)
* (\nabla_\theta L): gradient hàm mất mát

Learning rate biến thiên theo thời gian giúp điều chỉnh độ lớn bước học.

---

### 2.2 Vai trò của Learning Rate

Learning rate ảnh hưởng trực tiếp tới:

* Tốc độ hội tụ
* Độ ổn định
* Khả năng tối ưu toàn cục

Khi:

[
\eta_t \to 0 \Rightarrow \theta_{t+1}\approx \theta_t
]

⇒ quá trình học gần như dừng lại.

---

## 3. Phương pháp nghiên cứu

### 3.1 Warm-up Phase

#### 3.1.1 Khái niệm

Warm-up giúp tránh cập nhật quá mạnh ở giai đoạn đầu huấn luyện.

Theo , learning rate tăng dần trong giai đoạn đầu.

---

#### 3.1.2 Mô hình toán học

Warm-up tuyến tính:

[
\eta_t=\eta_{max}\cdot\frac{t}{T_{warm}},\quad t\le T_{warm}
]

Trong đó:

* (T_{warm}): số bước warm-up

---

### 3.2 Cosine Learning Rate Scheduler

#### 3.2.1 Nguyên lý

Cosine scheduler làm giảm learning rate theo hàm cosin.

---

#### 3.2.2 Công thức

Với (C) chu kỳ:

[
\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})
\left(1+\cos\frac{2\pi Ct}{T}\right)
]

Trường hợp (C=\frac{1}{2}):

[
\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})
\left(1+\cos\frac{\pi t}{T}\right)
]

---

#### 3.2.3 Đặc điểm

* Giảm learning rate mượt
* Tránh giảm đột ngột
* Phù hợp Transformer, LLM

---

### 3.3 Linear Learning Rate Scheduler

#### 3.3.1 Nguyên lý

Giảm learning rate tuyến tính sau warm-up.

---

#### 3.3.2 Công thức

[
\eta_t=
\begin{cases}
\eta_{max}\frac{t}{T_{warm}} & t\le T_{warm}\
\eta_{max}\left(1-\frac{t-T_{warm}}{T-T_{warm}}\right) & t>T_{warm}
\end{cases}
]

---

#### 3.3.3 Điều chỉnh số bước huấn luyện

Theo , việc khai báo số bước khác với thực tế giúp:

[
T_{sched}>T_{train}
\Rightarrow \eta_t>0
]

trong suốt quá trình huấn luyện.

---

### 3.4 Kết hợp với Gradient Clipping

Cập nhật tham số tổng quát:

[
\theta_{t+1}=\theta_t-\eta_t\cdot
\frac{c}{\max(|\mathbf{g}|,c)}\mathbf{g}
]

Trong đó:

* (c): ngưỡng clipping

---

## 4. Thực nghiệm

### 4.1 Mô hình minh họa

Theo tài liệu , mô hình gồm:

* Vector trọng số (w=(w_1,w_2))
* Mục tiêu: (w_1>w_2)
* SGD + Scheduler

Hàm mất mát:

[
L=-\log\frac{e^{w_1}}{e^{w_1}+e^{w_2}}
]

---

### 4.2 Cosine Scheduler

Quan sát thực nghiệm:

* Học theo từng pha
* Xuất hiện giai đoạn "đóng băng"
* Học mạnh khi (\eta_t) lớn

Đồ thị:

[
w(t)\propto \int_0^t \eta_s ds
]

---

### 4.3 Linear Scheduler

Đặc điểm:

* Học đều
* Ít dao động
* Dễ kiểm soát

Trường hợp (\eta_t=0):

[
\theta_{t+1}=\theta_t
]

⇒ không học.

---

### 4.4 So sánh thực nghiệm

| Phương pháp      | Độ mượt    | Hội tụ  | Ổn định |
| ---------------- | ---------- | ------- | ------- |
| Không scheduler  | Thấp       | Kém     | Thấp    |
| Cosine           | Cao        | Tốt     | Tốt     |
| Linear           | Trung bình | Tốt     | Cao     |
| Warm-up + Cosine | Rất cao    | Rất tốt | Rất tốt |

---

## 5. Thảo luận

### 5.1 Kiểm soát phạm vi giá trị

Theo , hệ thống học sâu cần giữ giá trị trong miền ổn định:

[
|\theta_i|<M,\quad |g_i|<K
]

Các kỹ thuật hỗ trợ:

* Weight initialization
* LayerNorm
* Weight decay
* Clipping
* Scheduler

---

### 5.2 Ứng dụng trong LLM

Scheduler giúp:

* Ổn định huấn luyện Transformer
* Giảm gradient noise
* Hạn chế overfitting

Đặc biệt quan trọng với mô hình trên 1B tham số.

---

### 5.3 Hạn chế

* Phụ thuộc siêu tham số
* Khó tối ưu thủ công
* Tăng độ phức tạp huấn luyện

Cần thử nghiệm nhiều cấu hình.

---

## 6. Kết luận

Bài viết đã trình bày Learning Rate Scheduler trong huấn luyện mô hình học sâu, tập trung vào Cosine và Linear Scheduler.

Kết quả cho thấy:

* Scheduler cải thiện hội tụ
* Warm-up tăng ổn định
* Kết hợp clipping cho hiệu quả cao

Các phương pháp này là thành phần không thể thiếu trong huấn luyện mô hình AI hiện đại.

---

## Tài liệu tham khảo

1. Learning Rate Scheduler Tutorial (Part 2) 
2. Loshchilov, I., Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts.
3. Kingma, D., Ba, J. (2015). Adam: A Method for Stochastic Optimization.
4. Vaswani, A. et al. (2017). Attention Is All You Need.

---
