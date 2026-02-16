
Dưới đây là **bài viết khoa học** được xây dựng dựa trên tài liệu **“Should You Output Logits or Log-Softmax(Logits)”**, có bổ sung phân tích học thuật và nguồn trích dẫn, trình bày dưới dạng **Markdown**.

---

# **So Sánh Đầu Ra Logits và Log-Softmax Trong Mô Hình Ngôn Ngữ: Tác Động Đến Huấn Luyện và Sinh Văn Bản**

---

## Abstract

Trong các mô hình ngôn ngữ dựa trên Transformer, đầu ra của mạng thường được biểu diễn dưới dạng logits hoặc log-softmax của logits. Việc lựa chọn dạng biểu diễn này ảnh hưởng trực tiếp đến quá trình huấn luyện, tính linh hoạt trong suy luận, và khả năng điều chỉnh nhiệt độ (temperature) khi sinh văn bản. Bài viết này phân tích sự khác biệt giữa hai cách tiếp cận, đánh giá ưu và nhược điểm của từng phương pháp, và làm rõ vai trò của chúng trong huấn luyện và triển khai mô hình ngôn ngữ lớn (LLMs). Kết quả cho thấy việc xuất logits mang lại tính linh hoạt cao hơn trong sinh văn bản, trong khi log-softmax thuận tiện hơn cho huấn luyện và các tác vụ phân loại. 

---

## 1. Introduction

Trong mô hình ngôn ngữ hiện đại, đầu ra của mạng nơ-ron thường là một vector có kích thước bằng số lượng token trong từ vựng. Vector này có thể được biểu diễn dưới hai dạng chính:

* Logits (giá trị thô, chưa chuẩn hóa),
* Log-softmax của logits (logarit của phân phối xác suất).

Theo tài liệu hướng dẫn, cả hai cách tiếp cận đều cho kết quả tương đương về mặt lý thuyết, nhưng lại có những hệ quả thực tiễn khác nhau trong quá trình sinh văn bản và huấn luyện. 

Mục tiêu của bài viết là:

* Phân tích cơ sở toán học của logits và log-softmax,
* Đánh giá ảnh hưởng đến temperature sampling,
* So sánh tác động đến loss function,
* Thảo luận ứng dụng thực tế trong LLMs.

---

## 2. Theoretical Background

### 2.1. Logits trong Mô Hình Ngôn Ngữ

Giả sử mô hình sinh ra vector đầu ra:

[
z = (z_1, z_2, \dots, z_V)
]

trong đó (V) là kích thước từ vựng. Vector (z) được gọi là logits, đại diện cho độ tin cậy chưa chuẩn hóa của từng token.

Logits có thể mang giá trị bất kỳ trong tập số thực và chưa có ý nghĩa xác suất.

---

### 2.2. Softmax và Log-Softmax

Phân phối xác suất được tính bằng hàm softmax:

[
P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
]

Log-softmax được định nghĩa:

[
\log P_i = z_i - \log \sum_j e^{z_j}
]

Hàm log-softmax giúp tăng độ ổn định số học và thường được dùng trực tiếp trong loss function.

---

### 2.3. Cross-Entropy và Negative Log-Likelihood

Trong huấn luyện mô hình ngôn ngữ, loss thường được tính bằng:

[
\mathcal{L} = - \log P_{target}
]

PyTorch thường kết hợp `LogSoftmax` và `NLLLoss` thành `CrossEntropyLoss`, cho phép truyền trực tiếp logits vào hàm loss.

Do đó, việc sử dụng logits hay log-softmax ảnh hưởng đến cách xây dựng pipeline huấn luyện.

---

## 3. Outputting Log-Softmax Inside the Model

### 3.1. Mô Hình Xuất Log-Softmax

Trong cách tiếp cận này, mô hình thực hiện:

[
\text{Output} = \log(\text{Softmax}(z))
]

ngay trong hàm `forward`.

Theo tài liệu, cách này thường được dùng trong các mô hình tập trung vào phân loại văn bản. 

---

### 3.2. Ưu Điểm

* Tương thích trực tiếp với `NLLLoss`,
* Ổn định số học,
* Đơn giản hóa code huấn luyện,
* Phù hợp cho classification.

---

### 3.3. Hạn Chế

Một khi log-softmax đã được tính, quá trình softmax không thể đảo ngược hoàn toàn. Do đó:

* Không thể điều chỉnh temperature,
* Phân phối xác suất bị “đóng băng”,
* Giảm tính linh hoạt trong sinh văn bản.

Theo tài liệu, khi log-softmax được tính sẵn, temperature mặc định bị cố định ở giá trị 1. 

---

## 4. Outputting Raw Logits

### 4.1. Mô Hình Xuất Logits

Trong cách tiếp cận này, mô hình trả về trực tiếp vector (z) mà không áp dụng softmax.

Việc chuẩn hóa được thực hiện bên ngoài mô hình, tùy theo mục đích sử dụng.

---

### 4.2. Kết Hợp Với Loss Function

Khi sử dụng logits, cần áp dụng:

```python
loss = nn.CrossEntropyLoss()(logits, targets)
```

Hàm này tự động thực hiện log-softmax bên trong.

Theo tài liệu, việc quên bước này có thể dẫn đến lỗi huấn luyện nghiêm trọng. 

---

### 4.3. Linh Hoạt Trong Sinh Văn Bản

Khi có logits, phân phối xác suất có thể được điều chỉnh bằng temperature:

[
P_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
]

Trong đó:

* (T < 1): phân phối sắc nét hơn,
* (T > 1): phân phối đa dạng hơn.

Cách này cho phép kiểm soát mức độ sáng tạo của mô hình khi sinh văn bản.

---

## 5. Temperature and Text Generation

### 5.1. Vai Trò Của Temperature

Temperature là tham số quan trọng trong sampling:

* Ảnh hưởng đến entropy,
* Quyết định độ ngẫu nhiên,
* Điều chỉnh phong cách văn bản.

Theo tài liệu, temperature chỉ có thể thay đổi khi logits chưa bị biến đổi thành log-softmax. 

---

### 5.2. Giới Hạn Của Log-Softmax

Nếu đầu ra là log-softmax:

* Có thể lấy lại xác suất bằng hàm exp,
* Nhưng không thể khôi phục logits ban đầu,
* Không thể áp dụng temperature mới.

Do đó, khả năng kiểm soát sinh văn bản bị hạn chế.

---

## 6. Experimental Observations

### 6.1. Ảnh Hưởng Đến Training

Thực nghiệm cho thấy:

* Log-softmax giúp training ổn định,
* Giảm nguy cơ overflow,
* Phù hợp cho supervised learning.

Ngược lại, logits yêu cầu kiểm soát cẩn thận hơn nhưng không làm giảm chất lượng huấn luyện nếu được xử lý đúng.

---

### 6.2. Ảnh Hưởng Đến Inference

Trong inference:

* Logits cho phép sampling linh hoạt,
* Log-softmax giới hạn khả năng điều chỉnh.

Theo tài liệu, các mô hình sinh văn bản chuyên dụng thường ưu tiên logits. 

---

## 7. Discussion

### 7.1. Lựa Chọn Phụ Thuộc Mục Tiêu

| Mục tiêu           | Logits     | Log-Softmax |
| ------------------ | ---------- | ----------- |
| Classification     | Trung bình | Tốt         |
| Text generation    | Rất tốt    | Hạn chế     |
| Temperature tuning | Có         | Không       |
| Code simplicity    | Trung bình | Cao         |

Việc lựa chọn phụ thuộc vào mục đích sử dụng mô hình.

---

### 7.2. Tác Động Đến Thiết Kế Pipeline

Quyết định xuất logits hay log-softmax ảnh hưởng đến:

* Cách viết loss,
* Cách triển khai generate(),
* Khả năng mở rộng ứng dụng,
* Khả năng debug.

Do đó, lập trình viên cần nắm rõ dạng dữ liệu đầu ra.

---

### 7.3. Liên Hệ Với LLMs Hiện Đại

Các mô hình như GPT-style hiện nay thường:

* Xuất logits,
* Chuẩn hóa bên ngoài,
* Áp dụng temperature, top-k, top-p sampling.

Cách tiếp cận này giúp tối ưu tính linh hoạt trong triển khai sản phẩm.

---

## 8. Limitations

Nghiên cứu này có một số hạn chế:

* Chủ yếu dựa trên phân tích lý thuyết,
* Không đánh giá trên nhiều kiến trúc khác nhau,
* Không đo hiệu suất trên downstream tasks.

Các nghiên cứu thực nghiệm quy mô lớn hơn là cần thiết để tổng quát hóa kết luận.

---

## 9. Conclusion

Bài viết đã phân tích sự khác biệt giữa việc xuất logits và log-softmax trong mô hình ngôn ngữ. Các kết luận chính gồm:

1. Log-softmax thuận tiện cho huấn luyện và phân loại.
2. Logits mang lại tính linh hoạt cao trong sinh văn bản.
3. Temperature chỉ có thể điều chỉnh khi sử dụng logits.
4. Log-softmax làm “đóng băng” phân phối xác suất.
5. Lựa chọn phụ thuộc vào mục tiêu ứng dụng.

Nhìn chung, việc xuất logits được xem là lựa chọn ưu tiên cho các hệ thống sinh văn bản hiện đại, trong khi log-softmax phù hợp hơn cho các bài toán phân loại.

---

## References

1. CodeChallenge: Should You Output Logits or Log-Softmax(Logits). Lecture Transcript.

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
4. Holtzman, A., et al. (2020). The Curious Case of Neural Text Degeneration. *ICLR*.

---
