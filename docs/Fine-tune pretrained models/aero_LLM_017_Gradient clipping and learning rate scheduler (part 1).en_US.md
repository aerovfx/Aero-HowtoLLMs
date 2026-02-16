

# 📘 Ứng Dụng Gradient Clipping và Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu

## Tóm tắt (Abstract)

Trong quá trình huấn luyện các mô hình học sâu quy mô lớn, hiện tượng mất ổn định số học và hội tụ kém thường xuyên xảy ra. Hai kỹ thuật phổ biến nhằm khắc phục vấn đề này là Gradient Clipping và Learning Rate Scheduler. Bài viết trình bày nguyên lý, cơ sở toán học và ứng dụng thực nghiệm của hai phương pháp trên, dựa trên tài liệu huấn luyện thực tế. Kết quả cho thấy việc áp dụng hợp lý các kỹ thuật này giúp tăng tính ổn định và độ tin cậy của quá trình tối ưu.

---

## 1. Giới thiệu

Huấn luyện mạng nơ-ron sâu thường dựa trên phương pháp tối ưu gradient descent. Tuy nhiên, với các mô hình lớn, gradient có thể trở nên rất lớn (gradient explosion), dẫn đến:

* Mất ổn định số học
* Sai lệch quá trình cập nhật
* Mô hình không hội tụ

Theo tài liệu hướng dẫn , hai kỹ thuật thường được sử dụng để giải quyết vấn đề này là:

* Gradient Clipping
* Learning Rate Scheduler

Mục tiêu nghiên cứu gồm:

* Phân tích cơ chế hoạt động của hai kỹ thuật
* Trình bày công thức toán học liên quan
* Đánh giá ảnh hưởng tới quá trình học
* Đề xuất hướng áp dụng thực tế

---

## 2. Cơ sở lý thuyết

### 2.1 Gradient Descent

Quá trình cập nhật tham số trong học sâu được mô tả bởi:

[
\theta_{t+1}=\theta_t-\eta \nabla_\theta L(\theta_t)
]

Trong đó:

* (\theta_t): tham số tại bước (t)
* (\eta): learning rate
* (L): hàm mất mát
* (\nabla_\theta L): gradient

Khi (|\nabla_\theta L|) quá lớn, cập nhật tham số trở nên không ổn định.

---

### 2.2 Chuẩn của Gradient

Chuẩn Euclid của gradient:

[
|\mathbf{g}|*2=\sqrt{\sum*{i=1}^{n}g_i^2}
]

Trong đó:

* (\mathbf{g}): vector gradient
* (g_i): phần tử thứ (i)

Gradient explosion xảy ra khi:

[
|\mathbf{g}|_2 \gg 1
]

---

## 3. Phương pháp nghiên cứu

### 3.1 Gradient Clipping

#### 3.1.1 Khái niệm

Gradient clipping là kỹ thuật giới hạn độ lớn của gradient nhằm tránh cập nhật quá mức.

Theo tài liệu , thay vì cắt từng phần tử riêng lẻ, toàn bộ vector gradient được chuẩn hóa.

---

#### 3.1.2 Công thức toán học

Với ngưỡng (c), gradient sau clipping:

[
\mathbf{g}_{clip}=
\begin{cases}
\mathbf{g} & \text{nếu } |\mathbf{g}|\le c\
\frac{c}{|\mathbf{g}|}\mathbf{g} & \text{nếu } |\mathbf{g}|>c
\end{cases}
]

Điều này đảm bảo:

[
|\mathbf{g}_{clip}|\le c
]

---

#### 3.1.3 Cập nhật tham số

Sau clipping:

[
\theta_{t+1}=\theta_t-\eta \mathbf{g}_{clip}
]

Việc này giúp giới hạn bước nhảy của tham số.

---

### 3.2 Learning Rate Scheduler

#### 3.2.1 Khái niệm

Learning rate scheduler là kỹ thuật thay đổi learning rate theo thời gian huấn luyện.

Theo , việc duy trì learning rate cố định có thể làm giảm hiệu quả học với mô hình lớn.

---

#### 3.2.2 Warm-up

Trong giai đoạn khởi động:

[
\eta_t=\eta_{max}\cdot\frac{t}{T_{warm}}
]

Trong đó:

* (T_{warm}): số epoch warm-up
* (\eta_{max}): learning rate cực đại

---

#### 3.2.3 Cosine Scheduler

Hàm cosine decay:

[
\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})\left(1+\cos\frac{\pi t}{T}\right)
]

Trong đó:

* (T): tổng số epoch
* (\eta_{min}): learning rate tối thiểu

---

#### 3.2.4 Linear Scheduler

Giảm tuyến tính:

[
\eta_t=\eta_{max}\left(1-\frac{t}{T}\right)
]

---

### 3.3 Kết hợp Clipping và Scheduler

Quy trình huấn luyện:

1. Tính gradient
2. Áp dụng clipping
3. Cập nhật learning rate
4. Cập nhật tham số

[
\theta_{t+1}=\theta_t-\eta_t\cdot \mathbf{g}_{clip}
]

---

## 4. Thực nghiệm

### 4.1 Mô hình minh họa

Theo mô tả trong tài liệu , mô hình gồm:

* Hai tham số trọng số
* Hàm mất mát L2
* SGD optimizer

Loss function:

[
L=\sum_{i=1}^{n}w_i^2
]

---

### 4.2 Ảnh hưởng của Gradient Clipping

| Trạng thái     | Chuẩn Gradient | Tốc độ học                |
| -------------- | -------------- | ------------------------- |
| Không clipping | > 10           | Nhanh nhưng không ổn định |
| Có clipping    | = 1            | Chậm, ổn định             |

Clipping giúp giảm hiện tượng gradient explosion nhưng làm chậm tốc độ hội tụ.

---

### 4.3 Ảnh hưởng của Scheduler

Kết quả cho thấy:

* Giai đoạn đầu: học ổn định
* Giai đoạn sau: giảm dao động
* Tránh overfitting

Learning curve mượt hơn khi dùng scheduler.

---

### 4.4 So sánh tổng hợp

| Phương pháp   | Ổn định    | Hội tụ     | Hiệu quả   |
| ------------- | ---------- | ---------- | ---------- |
| Không dùng    | Thấp       | Kém        | Trung bình |
| Chỉ clipping  | Trung bình | Trung bình | Tốt        |
| Chỉ scheduler | Tốt        | Tốt        | Tốt        |
| Kết hợp       | Rất tốt    | Cao        | Rất tốt    |

---

## 5. Thảo luận

### 5.1 Lợi ích của Gradient Clipping

Theo phân tích từ :

* Ngăn gradient explosion
* Ổn định số học
* Phù hợp mô hình lớn

Tuy nhiên, làm mất thông tin về độ lớn gradient.

---

### 5.2 Vai trò của Learning Rate Scheduler

Scheduler giúp:

* Tránh cập nhật quá mạnh ban đầu
* Tinh chỉnh ở giai đoạn cuối
* Cải thiện khả năng hội tụ

Đặc biệt hiệu quả với Transformer và LLM.

---

### 5.3 Hạn chế

* Cần tinh chỉnh siêu tham số
* Không phù hợp mô hình nhỏ
* Có thể làm chậm huấn luyện

Do đó cần lựa chọn phù hợp với bài toán.

---

## 6. Kết luận

Bài viết đã trình bày cơ sở lý thuyết và thực nghiệm của Gradient Clipping và Learning Rate Scheduler trong huấn luyện học sâu.

Kết quả cho thấy:

* Gradient Clipping giúp ổn định quá trình tối ưu
* Scheduler cải thiện hội tụ
* Kết hợp hai phương pháp cho hiệu quả cao nhất

Các kỹ thuật này đặc biệt quan trọng trong huấn luyện mô hình lớn và hệ thống AI hiện đại.

---

## Tài liệu tham khảo

1. Gradient Clipping and Learning Rate Scheduler Tutorial 
2. Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep Learning. MIT Press.
3. Loshchilov, I., Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts.

