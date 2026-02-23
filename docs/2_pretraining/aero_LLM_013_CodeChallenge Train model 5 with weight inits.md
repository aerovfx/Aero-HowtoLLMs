Dưới đây là **bài viết khoa học** được xây dựng dựa trên tài liệu *“CodeChallenge: Train Model 5 with Weight Initializations”*, có bổ sung phân tích học thuật và nguồn tham khảo, trình bày theo định dạng **Markdown**.

---

# **Phân Tích Ảnh Hưởng Của Khởi Tạo Trọng Số Và Sự Tiến Hóa Phân Phối Tham Số Trong Quá Trình Huấn Luyện Mô Hình Transformer**

---

## Abstract

Khởi tạo trọng số là một yếu tố quan trọng ảnh hưởng đến tính ổn định và hiệu quả huấn luyện của các mô hình học sâu. Trong các mô hình Transformer, đặc biệt là các mô hình ngôn ngữ lớn (LLMs), việc thiết lập phân phối ban đầu của tham số có ảnh hưởng trực tiếp đến sự lan truyền gradient và động học học tập. Bài viết này phân tích phương pháp khởi tạo trọng số trong mô hình GPT-style, cơ chế áp dụng tự động trong PyTorch, và sự thay đổi phân phối trọng số attention trong quá trình huấn luyện. Kết quả cho thấy các ma trận trọng số dần mở rộng phân phối theo thời gian, phản ánh khả năng biểu diễn ngày càng phong phú của mô hình. 

---

## 1. Introduction

Các mô hình Transformer hiện đại sử dụng hàng trăm triệu đến hàng tỷ tham số, khiến việc kiểm soát hành vi số học trong quá trình huấn luyện trở nên đặc biệt quan trọng. Một trong những yếu tố nền tảng ảnh hưởng đến quá trình này là khởi tạo trọng số ban đầu.

Theo tài liệu được cung cấp, việc áp dụng khởi tạo trọng số thủ công cho từng lớp trong LLM là không khả thi do số lượng module lớn. Thay vào đó, PyTorch cung cấp cơ chế `self.apply()` để áp dụng một hàm khởi tạo cho toàn bộ mô hình một cách tự động. 

Bài viết này tập trung nghiên cứu:

* Phương pháp khởi tạo trọng số tự động,
* Sự liên kết giữa embedding và unembedding,
* Sự thay đổi phân phối attention weights trong huấn luyện,
* Hàm ý đối với interpretability.

---

## 2. Background

### 2.1. Weight Initialization trong Deep Learning

Trong mạng nơ-ron sâu, khởi tạo trọng số ảnh hưởng đến:

* Biên độ kích hoạt (activation),
* Độ lớn gradient,
* Tốc độ hội tụ,
* Khả năng tránh gradient vanishing/exploding.

Nếu trọng số được khởi tạo không phù hợp, mô hình có thể rơi vào trạng thái học kém hiệu quả.

### 2.2. Transformer và Cấu Trúc Tham Số

Một mô hình GPT-style điển hình bao gồm:

* Token embeddings (WTE),
* Positional embeddings (WPE),
* Các khối Transformer,
* Attention QKV matrices,
* MLP layers,
* Output head (unembedding).

Mỗi thành phần có vai trò riêng trong quá trình biểu diễn ngôn ngữ.

---

## 3. Methodology

### 3.1. Áp Dụng Hàm Khởi Tạo Tự Động

Tài liệu mô tả việc xây dựng một hàm `weightInits` và áp dụng bằng:

```python
self.apply(self.weightInits)
```

Hàm này được áp dụng tuần tự lên mọi module trong mô hình. 

Các quy tắc khởi tạo bao gồm:

| Loại Module  | Phương pháp khởi tạo |
| ------------ | -------------------- |
| nn.Linear    | Normal(0, 0.02)      |
| Bias         | Zero initialization  |
| nn.Embedding | Xavier Normal        |



---

### 3.2. Kiểm Tra Phân Phối Ban Đầu

Sau khi khởi tạo, các đại lượng sau được kiểm tra:

* Vector bias,
* Độ lệch chuẩn của MLP weights,
* Độ lệch chuẩn của WTE và WPE.

Việc kiểm tra này giúp xác nhận tính đúng đắn của quá trình khởi tạo. 

---

### 3.3. Hiện Tượng Weight Tying

Một điểm quan trọng được chỉ ra là:

[
W_{embedding} = W_{unembedding}
]

Trong GPT-style models, trọng số embedding được gán trực tiếp cho output head, dẫn đến việc embedding thực chất bị chi phối bởi `nn.Linear`. 

Điều này giải thích vì sao độ lệch chuẩn của token embeddings không tuân theo Xavier mà gần với 0.02.

---

### 3.4. Theo Dõi Attention Weights Trong Huấn Luyện

Trong bài tập 2, tác giả yêu cầu:

* Trích xuất QKV matrices,
* Tính histogram,
* Lưu phân phối mỗi 50 epochs,
* Tính standard deviation cho từng layer.

Dữ liệu được trích xuất bằng:

```python
weights = model.blocks[i].attn.qkv.weight.detach().cpu()
```



---

## 4. Experimental Results

### 4.1. Phân Phối Trọng Số Ban Đầu

Kết quả cho thấy:

* Bias vectors = 0,
* Linear weights: std ≈ 0.02,
* Position embeddings: std ≈ 0.044,
* Token embeddings: std ≈ 0.02.

Sự khác biệt này được giải thích bởi weight tying. 

---

### 4.2. Sự Mở Rộng Phân Phối Khi Huấn Luyện

Theo quan sát:

* Ban đầu: phân phối hẹp, tập trung quanh 0,
* Sau huấn luyện: phân phối rộng hơn, đuôi dài hơn.

Hiện tượng này cho thấy mô hình dần sử dụng không gian tham số lớn hơn để mã hóa thông tin. 

---

### 4.3. Khác Biệt Giữa Các Layer

Phân tích standard deviation cho thấy:

* Các layer đầu mở rộng nhanh hơn,
* Các layer sau mở rộng chậm hơn,
* Tồn tại gradient theo chiều sâu.

Đặc biệt, layer gần embedding có mức tăng độ lệch chuẩn cao nhất. 

---

## 5. Discussion

### 5.1. Động Học Học Tập Của Trọng Số

Sự gia tăng độ lệch chuẩn phản ánh:

* Gia tăng độ phức tạp biểu diễn,
* Mở rộng không gian tìm kiếm,
* Học các mẫu tinh vi hơn.

Điều này phù hợp với lý thuyết về capacity expansion trong deep networks.

---

### 5.2. Liên Hệ Với Mechanistic Interpretability

Việc theo dõi phân phối trọng số là một kỹ thuật nền tảng trong lĩnh vực interpretability.

Theo tài liệu, phương pháp này giúp:

* Phát hiện hành vi bất thường,
* Đánh giá quá trình hình thành biểu diễn,
* Hỗ trợ kiểm soát rủi ro AI. 

---

### 5.3. Vai Trò Của Khởi Tạo Đối Với Attention

Attention matrices ban đầu có phân phối hẹp, giúp:

* Ổn định Softmax,
* Tránh saturation,
* Tăng khả năng học sớm.

Sau đó, phân phối mở rộng khi mô hình đã học được cấu trúc dữ liệu.

---

## 6. Limitations

Nghiên cứu còn tồn tại các hạn chế:

* Quy mô mô hình nhỏ,
* Thời gian huấn luyện ngắn,
* Dữ liệu hạn chế,
* Chỉ khảo sát một cấu hình.

Do đó, kết quả mang tính minh họa hơn là khái quát.

---

## 7. Implications for Large Language Models

Đối với LLMs quy mô lớn, kết quả này gợi ý rằng:

* Khởi tạo ảnh hưởng đến quỹ đạo học tập dài hạn,
* Weight tying làm thay đổi hành vi embedding,
* Các layer sớm đóng vai trò đặc biệt quan trọng,
* Theo dõi phân phối tham số là cần thiết cho an toàn AI.

Các pipeline huấn luyện hiện đại nên tích hợp công cụ phân tích này.

---

## 8. Conclusion

Bài viết đã phân tích phương pháp khởi tạo trọng số và sự tiến hóa của phân phối attention trong mô hình Transformer. Các kết luận chính gồm:

1. `self.apply()` cho phép khởi tạo đồng bộ toàn mô hình.
2. Linear layers được khởi tạo với Normal(0, 0.02).
3. Embedding chịu ảnh hưởng của weight tying.
4. Trọng số attention mở rộng theo thời gian.
5. Layer đầu học nhanh hơn layer sau.
6. Phân tích phân phối hỗ trợ interpretability.

Những kết quả này khẳng định vai trò trung tâm của weight initialization trong huấn luyện LLM.

---

## References

1. CodeChallenge: Train Model 5 with Weight Initializations. Lecture Transcript.

2. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. AISTATS.
3. He, K., et al. (2015). Delving Deep into Rectifiers. ICCV.
4. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
5. Olah, C., et al. (2020). Zoom In: An Introduction to Circuits. Distill.

---
