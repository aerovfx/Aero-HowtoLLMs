
# Phân Bố Tham Số Trong GPT-2: So Sánh Attention, MLP và Layer Normalization

## Tóm tắt (Abstract)

Trong các mô hình ngôn ngữ lớn dựa trên Transformer, việc phân bố tham số giữa các thành phần kiến trúc có ảnh hưởng trực tiếp đến năng lực biểu diễn và hiệu suất tính toán. Bài viết này trình bày phương pháp định lượng và so sánh số lượng tham số trong các lớp Attention, MLP và Layer Normalization của các phiên bản GPT-2. Thông qua phân tích thống kê và trực quan hóa bằng biểu đồ, nghiên cứu cho thấy các lớp MLP chiếm tỷ lệ tham số cao gấp khoảng hai lần Attention, trong khi Layer Normalization chỉ chiếm một phần rất nhỏ. Kết quả này phản ánh chiến lược thiết kế tối ưu của GPT-2 trong việc mở rộng quy mô mô hình.

---

## 1. Giới thiệu

Các mô hình GPT-2 được xây dựng dựa trên kiến trúc Transformer gồm nhiều khối lặp lại, mỗi khối bao gồm Attention, MLP và Layer Normalization. Mặc dù tổng số tham số thường được dùng để đánh giá quy mô mô hình, nhưng việc phân tích chi tiết sự phân bố tham số giữa các thành phần giúp hiểu rõ hơn về vai trò của từng khối chức năng.

Tài liệu “CodeChallenge: How Many Parameters (Part 2)” tiếp nối phần trước, tập trung vào việc so sánh số lượng tham số giữa Attention và MLP, cũng như phân tích các tham số trong Layer Normalization 

---

## 2. Mục tiêu nghiên cứu

Nghiên cứu này hướng tới các mục tiêu chính:

1. So sánh số lượng tham số giữa Attention và MLP.
2. Trực quan hóa tỷ lệ phần trăm tham số trong từng thành phần.
3. Phân tích tham số của Layer Normalization.
4. Giải thích nguyên nhân kiến trúc dẫn đến sự phân bố này.

Các mục tiêu trên góp phần làm rõ cơ chế thiết kế của GPT-2 

---

## 3. Phương pháp nghiên cứu

### 3.1. Trích xuất tham số theo tên

Mỗi tham số trong mô hình PyTorch đều có tên định danh. Phương pháp nghiên cứu dựa trên việc lọc các tham số có chứa chuỗi:

* `"attn"` hoặc `"attention"` → Attention
* `"mlp"` hoặc `"fc"` → MLP
* `"ln"` → Layer Normalization

Cách tiếp cận này cho phép phân loại chính xác các ma trận trọng số theo chức năng 

---

### 3.2. Đếm số lượng tham số

Số tham số trong mỗi tensor được tính bằng:

```python
p.numel()
```

Tổng số tham số cho từng nhóm được cộng dồn trong quá trình lặp qua `named_parameters()` 

---

### 3.3. Trực quan hóa bằng biểu đồ

Do số lượng tham số tuyệt đối giữa các mô hình khác nhau rất lớn, nghiên cứu sử dụng tỷ lệ phần trăm để biểu diễn:

[
\text{Percentage} = \frac{\text{Parameters of sublayer}}{\text{Total parameters}} \times 100%
]

Kết quả được thể hiện bằng biểu đồ cột (bar plot) để so sánh trực quan 

---

## 4. Phân tích Tham số Attention và MLP

### 4.1. Thành phần của Attention

Trong GPT-2, lớp Attention bao gồm:

* Ma trận QKV gộp (C_attn),
* Ma trận chiếu đầu ra (C_proj).

Các ma trận này chịu trách nhiệm học quan hệ phụ thuộc ngữ cảnh giữa các token 

---

### 4.2. Thành phần của MLP

MLP bao gồm hai lớp tuyến tính:

1. Lớp mở rộng chiều (FC / W1),
2. Lớp thu hẹp chiều (Projection / W2).

Cấu trúc này tạo ra sự mở rộng không gian đặc trưng, dẫn đến số lượng tham số lớn 

---

### 4.3. Kết quả so sánh

Kết quả thống kê cho thấy:

| Phiên bản | Attention (%) | MLP (%) |
| --------- | ------------- | ------- |
| Small     | ~22           | ~45     |
| Medium    | ~28           | ~56     |
| Large     | ~30           | ~60     |
| XL        | ~31           | ~63     |

MLP có số tham số xấp xỉ gấp đôi Attention trong tất cả các phiên bản 

---

### 4.4. Giải thích kiến trúc

Sự chênh lệch này xuất phát từ:

* MLP sử dụng tầng mở rộng chiều lớn,
* Attention giới hạn trong không gian embedding ban đầu.

Do đó, MLP trở thành thành phần tiêu thụ tham số lớn nhất trong GPT-2 

---

## 5. Xu hướng Tăng Tỷ lệ Theo Quy Mô Mô Hình

### 5.1. Vai trò của Embedding

Các ma trận embedding (token và position) chỉ xuất hiện một lần ở đầu mô hình và có kích thước cố định cho mọi phiên bản GPT-2.

Ngược lại, số lượng Transformer block tăng theo quy mô mô hình 

---

### 5.2. Ảnh hưởng đến tỷ lệ tham số

Khi số block tăng:

* Tham số Attention và MLP tăng tuyến tính,
* Tham số embedding giữ nguyên.

Do đó, tỷ lệ phần trăm của Attention và MLP trong tổng mô hình ngày càng lớn 

---

## 6. Phân Tích Tham Số Layer Normalization

### 6.1. Cấu trúc Layer Norm

Layer Normalization sử dụng hai tham số chính:

* Tham số scale ((\gamma)),
* Tham số shift ((\beta)).

Công thức:

[
y = \gamma \frac{x - \mu}{\sigma} + \beta
]



---

### 6.2. Kết quả thống kê

Kết quả cho thấy:

* Số tham số Layer Norm: vài chục nghìn đến ~100.000,
* Tỷ lệ: < 0.01% tổng mô hình.

Con số này rất nhỏ so với hàng trăm triệu hoặc hàng tỷ tham số tổng thể 

---

### 6.3. So sánh Weight và Bias

Khác với toàn mô hình, trong Layer Norm:

* Số weight = số bias.

Nguyên nhân là mỗi chiều embedding có đúng một tham số scale và một tham số shift 

---

## 7. Thảo luận

### 7.1. Ý nghĩa đối với thiết kế mô hình

Kết quả cho thấy GPT-2 ưu tiên:

* Dung lượng lớn cho MLP,
* Duy trì Attention ở mức vừa phải,
* Tối giản Layer Norm.

Cách thiết kế này giúp cân bằng giữa khả năng biểu diễn và chi phí tính toán.

---

### 7.2. Giá trị đối với người học

Bài tập giúp người học:

* Hiểu cấu trúc tham số thực tế,
* Rèn luyện kỹ năng đọc mô hình,
* Chuẩn bị cho nghiên cứu interpretability.



---

### 7.3. Hạn chế

Một số hạn chế của nghiên cứu:

* Chỉ phân tích số lượng, chưa xét phân bố giá trị,
* Không liên hệ trực tiếp với hiệu năng,
* Chưa mở rộng sang các mô hình mới hơn.

Do đó, cần kết hợp với phân tích thực nghiệm trong tương lai.

---

## 8. Kết luận

Bài viết đã phân tích sự phân bố tham số trong GPT-2 theo ba thành phần chính: Attention, MLP và Layer Normalization. Các kết luận chính bao gồm:

1. MLP chiếm khoảng gấp đôi số tham số của Attention.
2. Tỷ lệ Attention và MLP tăng theo quy mô mô hình.
3. Layer Normalization chỉ chiếm tỷ lệ rất nhỏ.
4. Weight và bias trong Layer Norm có số lượng bằng nhau.

Những kết quả này phản ánh chiến lược thiết kế tối ưu của GPT-2 và cung cấp nền tảng cho nghiên cứu sâu hơn về kiến trúc LLM.

---

## Tài liệu tham khảo

[1] CodeChallenge: How Many Parameters (Part 2), Lecture Transcript. 

-