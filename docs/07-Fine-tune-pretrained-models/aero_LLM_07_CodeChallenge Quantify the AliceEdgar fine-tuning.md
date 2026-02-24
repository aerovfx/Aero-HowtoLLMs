
# Định Lượng Hiệu Quả Tinh Chỉnh Phong Cách Văn Học: Thử Thách Alice và Edgar

## Tóm tắt

Bài viết này tập trung vào phương pháp định lượng sự thay đổi phong cách văn bản sau quá trình fine-tuning mô hình ngôn ngữ GPT-Neo. Sử dụng bộ phân loại BERT như một công cụ đo lường khách quan, nghiên cứu này đánh giá mức độ hội tụ của mô hình sinh văn bản về phía hai tác giả mục tiêu: Lewis Carroll (Alice) và Edgar Allan Poe (Edgar). Kết quả cho thấy các chỉ số định lượng như độ chính xác phân loại (Classification Accuracy) và hàm mất mát (Loss) là những công cụ phản ánh chính xác tiến trình học tập của mô hình.

---

## 1. Giới thiệu

Việc đánh giá tính sáng tạo và phong cách của các mô hình ngôn ngữ sau tinh chỉnh thường mang tính định tính và cảm tính. Để đưa ra những đánh giá khoa học hơn, chúng ta cần các phương pháp định lượng.

Theo tài liệu , thử thách "Quantify the Alice-Edgar fine-tuning" được thiết kế để đo lường xem mô hình sinh văn bản đã học được bao nhiêu tri thức về phong cách văn học mục tiêu thông qua một bộ phân loại độc lập.

Mục tiêu nghiên cứu:
* Xây dựng hệ thống đo lường hiệu quả tinh chỉnh.
* Phân tích sự thay đổi của độ chính xác phân loại theo thời gian.
* Đánh giá mối tương quan giữa sự hội tụ của mô hình sinh và mô hình phân loại.

---

## 2. Cơ sở lý thuyết

### 2.1. Đo lường sự khác biệt phân phối

Quá trình tinh chỉnh nhằm mục đích đưa phân phối xác suất của mô hình sinh ($P_{model}$) tiến gần đến phân phối xác suất của dữ liệu mục tiêu ($P_{data}$):

[
D_{KL}(P_{data} \parallel P_{model}) \rightarrow 0
]

Trong bài toán này, chúng ta sử dụng một bộ phân loại $C$ để ước lượng xác suất hậu nghiệm:

[
\hat{y} = C(x) = P(\text{Style} \mid x)
]

---

### 2.2. Chỉ số định lượng

Hai chỉ số chính được sử dụng để đánh giá:

1. **Độ chính xác phân loại (Accuracy):**
[
\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\arg\max C(x_i) = y_i)
]

2. **Hàm mất mát Cross-Entropy (Log-Loss):**
[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i)
]

---

## 3. Quy trình thực nghiệm

### 3.1. Thiết lập mô hình

* **Mô hình sinh:** GPT-Neo 125M được tinh chỉnh trên hai tập dữ liệu khác nhau.
* **Bộ phân loại:** BERT (base) đã được huấn luyện trước trên dữ liệu văn học Alice và Edgar.
* **Tập dữ liệu kiểm tra:** 121 đoạn văn bản chưa được sử dụng trong quá trình huấn luyện.

---

### 3.2. Chu kỳ đánh giá

Theo , việc đánh giá không thực hiện liên tục để tiết kiệm tài nguyên. Thay vào đó, sau mỗi 10 batch huấn luyện, mô hình sinh sẽ tạo ra các đoạn văn bản mẫu và bộ phân loại BERT sẽ tiến hành gán nhãn.

Tiến trình:
[
t = \{10, 20, 30, \dots, T\}
]

---

## 4. Phân tích kết quả

### 4.1. Sự tăng trưởng của độ chính xác

Tại giai đoạn đầu huấn luyện ($t=0$), bộ phân loại BERT gặp khó khăn trong việc phân biệt văn bản sinh từ hai mô hình, độ chính xác dao động quanh mức ngẫu nhiên:

[
\text{Acc}_{t=0} \approx 0.5
]

Khi quá trình tinh chỉnh tiến triển, văn bản sinh bắt đầu mang các đặc trưng phong cách rõ rệt hơn, dẫn đến độ chính xác tăng nhanh:

[
\text{Acc}_{t \rightarrow T} \rightarrow 0.9
]

---

### 4.2. Biểu đồ hội tụ

Quan hệ giữa Loss của mô hình phân loại trên văn bản sinh và số bước huấn luyện:

[
\frac{\partial \mathcal{L}_{cls}}{\partial t} < 0
]

Điều này xác nhận rằng mô hình sinh đang thực sự "di chuyển" trong không gian đặc trưng về phía vùng dữ liệu của Alice hoặc Edgar.

---

## 5. Thảo luận

### 5.1. Ưu điểm của phương pháp định lượng

* **Khách quan:** Loại bỏ yếu tố thiên kiến của con người trong đánh giá văn bản.
* **Thời gian thực:** Cho phép giám sát quá trình huấn luyện và dừng sớm (early stopping) khi đạt yêu cầu.
* **Tính quy mô:** Có thể áp dụng để đánh giá hàng nghìn mẫu văn bản trong thời gian ngắn.

---

### 5.2. Các yếu tố gây nhiễu

* **Sự sai lệch của Tokenizer:** Việc ánh xạ token giữa GPT-Neo và BERT có thể gây mất mát thông tin.
* **Chất lượng bộ phân loại:** Nếu BERT chưa được huấn luyện tốt, kết quả định lượng sẽ không còn tin cậy.

---

## 6. Kết luận

Thử thách định lượng quá trình tinh chỉnh Alice và Edgar đã chứng minh tính hiệu quả của việc sử dụng mô hình AI để đánh giá mô hình AI. Việc kết hợp giữa các chỉ số toán học và mô hình phân loại sâu cung cấp một cái nhìn toàn diện và chính xác về khả năng học phong cách của các LLM hiện đại.

---

## Tài liệu tham khảo

1. Tài liệu hướng dẫn: CodeChallenge Quantify the AliceEdgar fine-tuning.
2. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*.
3. Chen et al. (2021). *Evaluating Large Language Models for Code*.
4. Goodfellow et al. (2016). *Deep Learning*.

---
