
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [07 Fine tune pretrained models](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Fine-tuning Hiệu Quả Tham Số (Parameter-Efficient Fine-Tuning – PEFT) Trong Mô Hình Ngôn Ngữ Lớn

## Tóm tắt

Bài viết này trình bày tổng quan về phương pháp **Parameter-Efficient Fine-Tuning (PEFT)** – một nhóm kỹ thuật fine-tuning giúp giảm số lượng tham số cần huấn luyện trong các mô hình ngôn ngữ lớn (LLMs). Dựa trên tài liệu bài giảng , nghiên cứu phân tích các phương pháp tiêu biểu như Adapter, Low-Rank Adaptation, Prefix Tuning và Bias Tuning. Các công thức toán học được bổ sung nhằm làm rõ cơ sở lý thuyết. Kết quả cho thấy PEFT là giải pháp phù hợp cho môi trường hạn chế tài nguyên tính toán.

---

## 1. Giới thiệu

Các mô hình ngôn ngữ lớn hiện đại thường chứa hàng chục đến hàng trăm tỷ tham số. Việc fine-tuning toàn bộ mô hình đòi hỏi:

* Tài nguyên GPU lớn,
* Thời gian huấn luyện dài,
* Chi phí triển khai cao.

Theo tài liệu , PEFT được đề xuất nhằm giải quyết bài toán này bằng cách:

> Đóng băng phần lớn tham số và chỉ huấn luyện một tập con nhỏ.

Mục tiêu của PEFT là:

* Giảm chi phí huấn luyện,
* Duy trì hiệu quả học,
* Phù hợp với bài toán chuyên biệt.

---

## 2. Cơ sở lý thuyết

### 2.1. Mô hình ngôn ngữ tự hồi quy

Cho chuỗi token:

[
X=(x_1,x_2,\dots,x_n)
]

Xác suất sinh:

[
P(X)=\prod_{i=1}^{n}P(x_i|x_1,\dots,x_{i-1};\theta)
]

Trong đó (\theta) là tập tham số của mô hình.

---

### 2.2. Fine-tuning truyền thống

Với hàm mất mát cross-entropy:

[
\mathcal{L}(\theta)
===================

-\frac{1}{N}\sum_{i=1}^{N}
\log P(y_i|x_i;\theta)
]

Cập nhật bằng gradient descent:

[
\theta_{t+1}
============

\theta_t-\eta\nabla_\theta\mathcal{L}
]

Toàn bộ tham số đều được cập nhật.

---

### 2.3. Fine-tuning hiệu quả tham số

Trong PEFT, tham số được chia:

[
\theta = (\theta_f, \theta_t)
]

với:

* (\theta_f): tham số đóng băng,
* (\theta_t): tham số huấn luyện.

Điều kiện:

[
\nabla_{\theta_f}\mathcal{L}=0
]

Chỉ (\theta_t) được cập nhật.

---

## 3. Tổng quan về PEFT

Theo tài liệu , PEFT là một **họ phương pháp**, không phải một kỹ thuật đơn lẻ. Các phương pháp chính gồm:

1. Adapter
2. Low-Rank Adaptation (LoRA/DoRA)
3. Prefix Tuning
4. Bias Tuning

Các phương pháp này thường được triển khai thông qua thư viện của Hugging Face.

---

## 4. Các phương pháp PEFT tiêu biểu

### 4.1. Adapter

#### 4.1.1. Nguyên lý

Adapter chèn các mô-đun nhỏ vào giữa các lớp Transformer:

[
h' = h + W_{up}\sigma(W_{down}h)
]

Trong đó:

* (W_{down}\in\mathbb{R}^{d\times r}),
* (W_{up}\in\mathbb{R}^{r\times d}),
* (r \ll d).

Cấu trúc giống autoencoder nén–giải nén.

---

#### 4.1.2. Số tham số

Số tham số adapter:

[
P_{adapter}=2dr
]

So với:

[
P_{full}=d^2
]

⇒ (P_{adapter}\ll P_{full})

---

### 4.2. Low-Rank Adaptation (LoRA)

#### 4.2.1. Phân rã ma trận

Cho trọng số gốc:

[
W\in\mathbb{R}^{m\times n}
]

LoRA biểu diễn:

[
W' = W + BA
]

với:

[
B\in\mathbb{R}^{m\times r},\quad
A\in\mathbb{R}^{r\times n}
]

và (r\ll \min(m,n)).

---

#### 4.2.2. Giảm tham số

Số tham số:

[
P_{LoRA}=r(m+n)
]

So với:

[
P_{full}=mn
]

Ví dụ:

* (m=n=1000),
* (r=100):

[
P_{full}=10^6,\quad
P_{LoRA}=2\times10^5
]

---

### 4.3. Prefix Tuning

#### 4.3.1. Cơ chế

Thêm vector tiền tố (P):

[
X' = [P; X]
]

với:

[
P\in\mathbb{R}^{k\times d}
]

Đầu vào attention:

[
Q,K,V = (X'W_Q,X'W_K,X'W_V)
]

Chỉ (P) được huấn luyện.

---

#### 4.3.2. Số tham số

[
P_{prefix}=kd
]

Rất nhỏ so với toàn mô hình.

---

### 4.4. Bias Tuning

#### 4.4.1. Nguyên lý

Chỉ huấn luyện bias:

[
y = Wx + b
]

Cập nhật:

[
b_{t+1}=b_t-\eta\nabla_b\mathcal{L}
]

Giữ nguyên (W).

---

#### 4.4.2. Đặc điểm

Bias chủ yếu dịch chuyển phân phối:

[
P'(y|x)=P(y-b|x)
]

Ảnh hưởng yếu đến cấu trúc biểu diễn.

---

## 5. Phân tích hiệu quả PEFT

### 5.1. Chi phí tính toán

Gọi:

* (P_{full}): tham số đầy đủ,
* (P_{peft}): tham số PEFT.

Tỷ lệ:

[
r=\frac{P_{peft}}{P_{full}}\ll 1
]

Thời gian huấn luyện:

[
T_{peft}\approx rT_{full}
]

---

### 5.2. Khả năng tổng quát hóa

Khi số tham số giảm:

[
P\downarrow \Rightarrow Var(\theta)\downarrow
]

⇒ giảm overfitting.

Tuy nhiên:

[
Bias(\theta)\uparrow
]

⇒ mô hình kém linh hoạt.

---

### 5.3. Đánh đổi hiệu năng

Giả sử:

[
Acc_{full},\quad Acc_{peft}
]

Thông thường:

[
Acc_{peft}\le Acc_{full}
]

nhưng:

[
\frac{Acc_{peft}}{Cost_{peft}}

>

\frac{Acc_{full}}{Cost_{full}}
]

⇒ PEFT hiệu quả về chi phí.

---

## 6. Thảo luận

### 6.1. Ưu điểm

1. Giảm tài nguyên GPU.
2. Huấn luyện nhanh.
3. Lưu trữ gọn nhẹ.
4. Dễ triển khai.

---

### 6.2. Hạn chế

Theo tài liệu :

* Hiệu năng thấp hơn fine-tuning đầy đủ,
* Khó tổng quát hóa đa nhiệm,
* Phụ thuộc bài toán.

---

### 6.3. Khi nào nên dùng PEFT?

PEFT phù hợp khi:

[
N_{data}\ll P_{model}
]

và:

* Bài toán phân loại,
* Gán nhãn,
* Chatbot chuyên ngành,
* Tác vụ hẹp.

Không phù hợp cho tác vụ mở, sáng tạo.

---

## 7. Ứng dụng thực tiễn

PEFT được ứng dụng trong:

* NLP doanh nghiệp,
* Chatbot nội bộ,
* Phân tích văn bản y tế,
* Pháp lý,
* Tài chính.

Kết hợp với thư viện của Hugging Face giúp triển khai nhanh trong môi trường sản xuất.

---

## 8. Kết luận

Bài viết đã trình bày tổng quan về Parameter-Efficient Fine-Tuning dựa trên tài liệu bài giảng và phân tích toán học. Các kết luận chính:

1. PEFT giúp giảm mạnh chi phí huấn luyện.
2. Hiệu quả tốt cho tác vụ chuyên biệt.
3. Không thay thế hoàn toàn fine-tuning đầy đủ.
4. Là giải pháp thực tiễn cho hệ thống hạn chế tài nguyên.

Trong tương lai, việc kết hợp PEFT với các phương pháp thích nghi động và học đa nhiệm là hướng nghiên cứu tiềm năng.

---

## Tài liệu tham khảo

1. Parameter-Efficient Fine-Tuning (PEFT) – Lecture Notes 
2. Vaswani et al. (2017). Attention Is All You Need.
3. Hu et al. (2022). LoRA: Low-Rank Adaptation of LLMs.
4. He et al. (2022). Towards Parameter-Efficient Transfer Learning.
5. Goodfellow et al. (2016). *Deep Learning*.

---
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
