# Instruction Tuning với GPT-2 trong Huấn luyện Mô hình Ngôn ngữ

## Tóm tắt

Bài viết này trình bày phương pháp Instruction Tuning áp dụng cho mô hình GPT-2 dựa trên tài liệu đính kèm. Nội dung tập trung vào quy trình xây dựng dữ liệu, kỹ thuật tinh chỉnh mô hình, cơ sở toán học và đánh giá hiệu năng. Ngoài ra, bài viết bổ sung các nguồn tham khảo học thuật nhằm làm rõ vai trò của Instruction Tuning trong phát triển chatbot và trợ lý ảo hiện đại.

---

## 1. Giới thiệu

Mô hình GPT-2 là một trong những mô hình ngôn ngữ nền tảng đặt nền móng cho sự phát triển của các Large Language Models (LLMs). Tuy nhiên, mô hình gốc chủ yếu được huấn luyện trên dữ liệu văn bản thuần, chưa tối ưu cho việc làm theo yêu cầu của người dùng. Instruction Tuning ra đời nhằm khắc phục hạn chế này, giúp mô hình phản hồi chính xác và phù hợp hơn với ngữ cảnh.

Tài liệu đính kèm "Instruction Tuning with GPT-2" cho thấy quá trình tinh chỉnh GPT-2 có thể thực hiện hiệu quả ngay cả với tài nguyên tính toán hạn chế.

---

## 2. Tổng quan về GPT-2

### 2.1. Kiến trúc Transformer

GPT-2 được xây dựng dựa trên kiến trúc Transformer với cơ chế Self-Attention. Mỗi lớp Transformer bao gồm:

* Multi-Head Attention
* Feed-Forward Network
* Layer Normalization
* Residual Connection

### 2.2. Biểu diễn chuỗi đầu vào

Chuỗi đầu vào được mã hóa thành các token:

[
X = (x_1, x_2, ..., x_T)
]

và được ánh xạ thành vector nhúng:

[
e_t = E(x_t)
]

Trong đó (E) là ma trận embedding.

---

## 3. Instruction Tuning với GPT-2

### 3.1. Cấu trúc dữ liệu huấn luyện

Dữ liệu được tổ chức dưới dạng:

[
D = {(I_i, Y_i)}_{i=1}^{N}
]

Trong đó:

* (I_i): câu lệnh
* (Y_i): phản hồi mong muốn
* (N): số lượng mẫu

Ví dụ:

```
Instruction: Tóm tắt đoạn văn sau
Response: ...
```

---

### 3.2. Chuẩn hóa dữ liệu đầu vào

Mỗi mẫu dữ liệu được chuyển thành chuỗi:

[
S_i = [BOS, I_i, SEP, Y_i, EOS]
]

Trong đó BOS, SEP, EOS là các token đặc biệt.

---

## 4. Cơ sở toán học của quá trình huấn luyện

### 4.1. Mô hình xác suất ngôn ngữ

GPT-2 mô hình hóa xác suất chuỗi:

[
P(X) = \prod_{t=1}^{T} P(x_t | x_{<t})
]

---

### 4.2. Hàm mất mát Cross-Entropy

Hàm mất mát được sử dụng là:

[
\mathcal{L} = - \frac{1}{T} \sum_{t=1}^{T} y_t \log(\hat{y}_t)
]

Trong đó:

* (y_t): nhãn thật
* (\hat{y}_t): xác suất dự đoán

---

### 4.3. Thuật toán tối ưu Adam

GPT-2 thường được huấn luyện với Adam:

[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
]

[
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
]

[
\theta_t = \theta_{t-1} - \eta \frac{m_t}{\sqrt{v_t}+\epsilon}
]

---

## 5. Quy trình Instruction Tuning với GPT-2

Quy trình gồm các bước:

1. Thu thập dữ liệu instruction
2. Làm sạch và tiền xử lý
3. Chuẩn hóa dữ liệu theo template
4. Fine-tune GPT-2
5. Đánh giá và tinh chỉnh

Sơ đồ tổng quát:

```
Dữ liệu → Tokenizer → GPT-2 → Loss → Adam → Cập nhật tham số
```

---

## 6. Đánh giá mô hình

### 6.1. Chỉ số Perplexity

[
PP = \exp(\mathcal{L})
]

### 6.2. Độ chính xác theo nhiệm vụ

Mô hình được đánh giá trên các tập kiểm thử instruction.

---

## 7. Thực nghiệm minh họa

Giả sử tập huấn luyện gồm (N=10.000) mẫu, sau 5 epoch huấn luyện, hàm mất mát hội tụ:

[
\mathcal{L}_{final} \approx 1.95
]

Tương ứng:

[
PP \approx e^{1.95} \approx 7.03
]

---

## 8. Hạn chế

* Hiệu năng phụ thuộc mạnh vào dữ liệu
* Khó mở rộng với dữ liệu lớn
* Dễ overfitting nếu dữ liệu nhỏ

---

## 9. Hướng phát triển

* Kết hợp RLHF
* Instruction đa ngôn ngữ
* Huấn luyện phân tán
* Tối ưu mô hình nhẹ

---

## 10. Kết luận

Instruction Tuning giúp GPT-2 chuyển từ mô hình sinh văn bản tổng quát sang mô hình có khả năng tuân thủ yêu cầu người dùng. Việc kết hợp dữ liệu có cấu trúc và tối ưu hóa toán học đóng vai trò then chốt trong nâng cao chất lượng chatbot.

---

## Tài liệu tham khảo

1. Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners.
2. Brown, T. et al. (2020). Language Models are Few-Shot Learners.
3. Wei, J. et al. (2022). Finetuned Language Models Are Zero-Shot Learners.
4. Ouyang, L. et al. (2022). Training Language Models with Human Feedback.
5. Video: Instruction Tuning with GPT-2 (File đính kèm).
