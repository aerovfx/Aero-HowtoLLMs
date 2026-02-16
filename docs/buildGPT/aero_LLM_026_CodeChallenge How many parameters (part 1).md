Dưới đây là bài viết khoa học được xây dựng dựa trên tài liệu **“CodeChallenge: How Many Parameters (Part 1)”**, có bổ sung trích dẫn và trình bày theo định dạng **Markdown**.

---

# Phân Tích Số Lượng Tham Số Trong Mô Hình GPT-2: Phương Pháp Định Lượng và Ý Nghĩa Kiến Trúc

## Tóm tắt (Abstract)

Số lượng tham số là một trong những yếu tố quan trọng quyết định năng lực biểu diễn và hiệu suất của mô hình ngôn ngữ lớn. Bài viết này trình bày phương pháp đếm và phân tích tham số trong các biến thể GPT-2 thông qua bài tập lập trình. Nghiên cứu tập trung vào việc so sánh quy mô mô hình, xác minh cơ chế chia sẻ trọng số giữa embedding và unembedding, cũng như đánh giá tỷ lệ giữa trọng số và bias. Kết quả cho thấy phần lớn tham số của GPT-2 nằm ở các ma trận trọng số, trong khi bias chiếm tỷ lệ rất nhỏ, phản ánh đặc điểm thiết kế của các mô hình Transformer hiện đại.

---

## 1. Giới thiệu

Các mô hình ngôn ngữ lớn (Large Language Models – LLMs) được xây dựng dựa trên hàng triệu đến hàng tỷ tham số. Việc hiểu rõ cấu trúc và phân bố các tham số giúp người nghiên cứu:

* Đánh giá độ phức tạp của mô hình,
* So sánh các phiên bản khác nhau,
* Hiểu sâu cơ chế học biểu diễn.

Tài liệu “CodeChallenge: How Many Parameters (Part 1)” được thiết kế nhằm giúp người học phát triển kỹ năng truy vấn và phân tích tham số trong GPT-2. 

---

## 2. Mục tiêu nghiên cứu

Bài viết này hướng tới ba mục tiêu chính:

1. Đếm tổng số tham số huấn luyện của các phiên bản GPT-2.
2. Xác minh cơ chế chia sẻ trọng số giữa embedding và unembedding.
3. Phân tích tỷ lệ giữa trọng số (weights) và độ lệch (biases).

Các mục tiêu này giúp xây dựng nền tảng cho việc đánh giá và diễn giải kiến trúc LLM. 

---

## 3. Phương pháp nghiên cứu

### 3.1. Tổ chức mô hình bằng Dictionary

Các mô hình GPT-2 được lưu trữ trong một dictionary Python, trong đó:

* Key: tên rút gọn (small, medium, large, xl),
* Value: mô hình tương ứng.

Cách tổ chức này cho phép lặp qua các mô hình một cách hệ thống. 

---

### 3.2. Đếm tham số bằng PyTorch

Tổng số tham số được tính bằng cách lặp qua `model.parameters()`:

```python
total = sum(p.numel() for p in model.parameters())
```

Phương pháp này cho phép đếm chính xác toàn bộ tham số có thể huấn luyện. 

---

### 3.3. Tối ưu thời gian thực thi

Các phiên bản GPT-2 lớn (Large, XL) có số lượng tham số rất lớn, khiến thời gian lặp tăng đáng kể. Vì vậy, quy trình được đề xuất là:

1. Phát triển và kiểm thử trên GPT-2 Small,
2. Sau đó mở rộng sang các phiên bản lớn hơn.

Cách tiếp cận này giúp giảm thời gian chờ và tăng hiệu quả lập trình. 

---

## 4. Kết quả đếm tham số

### 4.1. Tổng số tham số của GPT-2

Kết quả cho thấy:

| Phiên bản | Số tham số (xấp xỉ) |
| --------- | ------------------- |
| Small     | 124 triệu           |
| Medium    | ~355 triệu          |
| Large     | ~774 triệu          |
| XL        | ~1.5 tỷ             |

Trong đó, GPT-2 Small có khoảng 124 triệu tham số, được xem là nhỏ so với các LLM hiện đại. 

---

### 4.2. So sánh với mô hình tự xây dựng

Bài tập cho thấy mô hình “Model 5” tự xây dựng có khoảng 163 triệu tham số. Sau khi trừ đi lớp unembedding (~38 triệu), số còn lại trùng khớp với GPT-2 Small:

[
163M - 38M \approx 124M
]

Kết quả này chứng minh rằng hai mô hình có kiến trúc tương đương. 

---

## 5. Cơ chế chia sẻ trọng số (Weight Tying)

### 5.1. Embedding và Unembedding

Trong GPT-2, ma trận embedding đầu vào và ma trận unembedding đầu ra được chia sẻ:

[
W_{embed} = W_{unembed}^T
]

Điều này giúp:

* Giảm số lượng tham số,
* Cải thiện khả năng tổng quát hóa,
* Tăng tính ổn định huấn luyện.



---

### 5.2. Xác minh bằng tương quan

Việc trích xuất và so sánh hai ma trận cho thấy hệ số tương quan xấp xỉ 1, chứng minh chúng gần như giống hệt nhau. Đây là bằng chứng thực nghiệm cho cơ chế weight tying. 

---

## 6. Phân tích Trọng số và Bias

### 6.1. Định nghĩa

Trong một lớp tuyến tính:

[
y = Wx + b
]

Trong đó:

* (W): trọng số (weights),
* (b): độ lệch (bias).

Weights quyết định mức độ ảnh hưởng của đầu vào, trong khi bias cho phép dịch chuyển phân phối. 

---

### 6.2. Kết quả thống kê

Kết quả phân tích cho thấy:

| Loại tham số | Tỷ lệ  |
| ------------ | ------ |
| Weights      | ~99.9% |
| Biases       | <0.1%  |

Bias chỉ chiếm một phần rất nhỏ trong tổng tham số mô hình. 

---

### 6.3. Ý nghĩa

Tỷ lệ này cho thấy:

* Trọng số là yếu tố quyết định chính đến năng lực mô hình.
* Bias có ảnh hưởng tương đối nhỏ.
* Việc tối ưu và khởi tạo weights quan trọng hơn bias.

Ngoài ra, layer normalization cũng làm giảm vai trò của bias trong mô hình. 

---

## 7. Thảo luận

### 7.1. Giá trị giáo dục

Bài tập đếm tham số giúp người học:

* Hiểu rõ cấu trúc nội tại của LLM,
* Rèn luyện kỹ năng phân tích mô hình,
* Liên kết lý thuyết và thực hành.



---

### 7.2. Ý nghĩa đối với thiết kế mô hình

Kết quả cho thấy thiết kế GPT-2 ưu tiên:

* Ma trận trọng số lớn,
* Chia sẻ tham số,
* Hạn chế bias dư thừa.

Cách tiếp cận này giúp mô hình mở rộng hiệu quả về quy mô. 

---

### 7.3. Hạn chế

Một số hạn chế của phương pháp:

* Chỉ phân tích số lượng, chưa đánh giá chất lượng tham số,
* Không xem xét sự phân bố giá trị,
* Chưa gắn với hiệu năng thực tế.

Do đó, cần kết hợp với phân tích thực nghiệm trong các nghiên cứu tiếp theo.

---

## 8. Kết luận

Bài viết đã trình bày phương pháp đếm và phân tích tham số trong các mô hình GPT-2. Các kết quả chính bao gồm:

1. GPT-2 Small có khoảng 124 triệu tham số.
2. Embedding và unembedding được chia sẻ trọng số.
3. Weights chiếm khoảng 99.9% tổng tham số.
4. Bias đóng vai trò thứ yếu trong kiến trúc.

Những phát hiện này giúp làm rõ cách thức thiết kế của các mô hình ngôn ngữ hiện đại và cung cấp nền tảng cho nghiên cứu tối ưu hóa và diễn giải LLM.

---

## Tài liệu tham khảo

[1] CodeChallenge: How Many Parameters (Part 1), Lecture Transcript. 

---