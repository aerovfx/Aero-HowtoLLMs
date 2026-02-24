

# Khảo Sát Mô Hình GPT-2 Tiền Huấn Luyện của OpenAI: Kiến Trúc, Tham Số và Cơ Chế Sinh Văn Bản

## Tóm tắt (Abstract)

GPT-2 là một trong những mô hình ngôn ngữ dựa trên kiến trúc Transformer có ảnh hưởng lớn trong lĩnh vực xử lý ngôn ngữ tự nhiên. Bài viết này trình bày quá trình tải, phân tích và khai thác mô hình GPT-2 tiền huấn luyện thông qua thư viện Transformers. Nội dung tập trung vào cấu trúc bên trong của mô hình, cơ chế gộp QKV, phân tích tham số, siêu tham số cấu hình và phương pháp sinh văn bản với các tham số ngẫu nhiên như temperature. Kết quả cho thấy việc khảo sát mô hình tiền huấn luyện giúp hiểu rõ hơn về cơ chế hoạt động của các hệ thống ngôn ngữ lớn và đặt nền tảng cho nghiên cứu diễn giải mô hình (interpretability).

---

## 1. Giới thiệu

Các mô hình ngôn ngữ lớn (Large Language Models – LLMs) đóng vai trò trung tâm trong nhiều ứng dụng hiện đại như chatbot, dịch máy và sinh văn bản tự động. GPT-2, do OpenAI phát triển, là một trong những mô hình tiên phong thể hiện tiềm năng của Transformer trong lĩnh vực này.

Tài liệu tham khảo hướng dẫn cách tải và khảo sát GPT-2 tiền huấn luyện thông qua nền tảng Hugging Face, cho phép người học tiếp cận trực tiếp với mô hình đã được huấn luyện quy mô lớn 

Mục tiêu của bài viết là:

* Phân tích kiến trúc nội tại của GPT-2.
* Khảo sát các tham số và siêu tham số.
* Đánh giá cơ chế sinh văn bản.
* Thảo luận ý nghĩa đối với nghiên cứu AI.

---

## 2. Nguồn mô hình và môi trường triển khai

### 2.1. Nền tảng Hugging Face

GPT-2 được cung cấp thông qua thư viện Transformers của Hugging Face, nơi lưu trữ nhiều mô hình học sâu và bộ dữ liệu mở.

Việc sử dụng nền tảng này cho phép:

* Truy cập nhanh mô hình tiền huấn luyện.
* Chuẩn hóa giao diện lập trình.
* Dễ dàng mở rộng sang các mô hình khác.



---

### 2.2. Tải mô hình và tokenizer

Mô hình GPT-2 Small được tải bằng lệnh:

```python
AutoModelForCausalLM.from_pretrained("gpt2")
```

Phiên bản này có khoảng 124 triệu tham số và tương ứng với GPT-2 Small, trùng với cấu hình của Model 5 trong các bài thực hành trước đó 

---

## 3. Kiến trúc tổng thể của GPT-2

### 3.1. Cấu trúc chính

GPT-2 bao gồm các thành phần:

1. Word Token Embedding (WTE)
2. Word Position Embedding (WPE)
3. Dãy các khối Transformer
4. Layer Normalization cuối
5. Lớp LM Head (Unembedding)

Các thành phần này tạo thành một pipeline khép kín cho mô hình ngôn ngữ tự hồi quy 

---

### 3.2. Các khối Transformer

Mô hình GPT-2 Small gồm 12 khối Transformer, được lưu trữ trong danh sách `H`.

Mỗi khối bao gồm:

* Layer Normalization
* Self-Attention
* Projection Layer
* Feed-Forward Network (MLP)
* Residual Connections

Cấu trúc này giúp duy trì ổn định gradient và tăng khả năng biểu diễn.



---

## 4. Cơ chế Multi-Head Attention và QKV

### 4.1. Gộp ma trận QKV

Trong GPT-2, ba ma trận Query, Key và Value không được lưu riêng lẻ mà được gộp trong một ma trận duy nhất có kích thước:

[
768 \times 2304 = 768 \times (3 \times 768)
]

Cách thiết kế này giúp:

* Giảm số phép truy cập bộ nhớ.
* Tối ưu thực thi trên GPU.
* Đơn giản hóa kiến trúc.



---

### 4.2. Ma trận chiếu (Projection Matrix)

Sau khi tính attention, kết quả được nhân với ma trận chiếu (W_0) kích thước:

[
768 \times 768
]

Ma trận này giúp tổng hợp thông tin từ các head attention khác nhau.



---

## 5. Phân tích tham số và cấu hình

### 5.1. Tham số huấn luyện

Thông qua việc liệt kê `named_parameters`, có thể quan sát:

* Tên từng lớp
* Kích thước trọng số
* Số lượng tham số

Kết quả cho thấy:

* Tổng tham số: ~163 triệu
* Tham số thực (sau weight tying): ~124 triệu



---

### 5.2. Siêu tham số cấu hình

Đối tượng `config` cung cấp thông tin meta:

| Tham số         | Giá trị |
| --------------- | ------- |
| Context length  | 1024    |
| Embedding dim   | 768     |
| Attention heads | 12      |
| Layers          | 12      |

Các giá trị này trùng khớp với cấu hình GPT-2 Small tiêu chuẩn 

---

## 6. Cơ chế sinh văn bản

### 6.1. Quy trình sinh

Quá trình sinh văn bản gồm:

1. Token hóa chuỗi đầu vào.
2. Chuyển thành tensor.
3. Đưa vào mô hình.
4. Lấy mẫu token tiếp theo.
5. Giải mã thành văn bản.

Hàm `generate()` được sử dụng để tự động hóa quy trình này 

---

### 6.2. Vai trò của Temperature

Tham số `temperature` điều chỉnh mức độ ngẫu nhiên khi lấy mẫu:

* Temperature thấp (≈0.1): văn bản lặp, ít sáng tạo.
* Temperature trung bình (≈1): cân bằng.
* Temperature cao (≥10): văn bản hỗn loạn, mất mạch lạc.

Ví dụ, temperature quá thấp khiến mô hình lặp lại cùng một câu, trong khi temperature cao làm giảm tính logic của văn bản sinh ra 

---

### 6.3. Tính ngẫu nhiên và giới hạn mạch lạc

Do bản chất xác suất, GPT-2 có xu hướng:

* Duy trì mạch lạc cục bộ.
* Suy giảm coherence ở chuỗi dài.

Điều này phản ánh hạn chế của mô hình trong việc nắm bắt ngữ cảnh dài hạn.



---

## 7. Thảo luận

### 7.1. Ý nghĩa đối với nghiên cứu mô hình

Việc khảo sát GPT-2 tiền huấn luyện cho thấy:

* Cấu trúc mô hình có tính mô-đun cao.
* Có thể truy cập và phân tích chi tiết từng thành phần.
* Phù hợp cho nghiên cứu diễn giải (mechanistic interpretability).



---

### 7.2. Ứng dụng thực tiễn

Các mô hình như GPT-2 cho phép:

* Thử nghiệm nhanh ý tưởng NLP.
* Xây dựng prototype hệ thống sinh văn bản.
* Học tập kiến trúc LLM.

Hugging Face đóng vai trò trung gian quan trọng trong việc phổ cập các mô hình này.

---

### 7.3. Hạn chế

Một số hạn chế:

* GPT-2 đã lỗi thời so với các LLM mới.
* Khả năng suy luận dài hạn còn yếu.
* Chưa tích hợp cơ chế kiểm soát nội dung.

Các hạn chế này mở ra hướng nghiên cứu cho thế hệ mô hình tiếp theo.

---

## 8. Kết luận

Bài viết đã trình bày quá trình khảo sát mô hình GPT-2 tiền huấn luyện, bao gồm:

* Nguồn gốc và cách tải mô hình.
* Phân tích kiến trúc Transformer.
* Cơ chế gộp QKV.
* Thống kê tham số và cấu hình.
* Cơ chế sinh văn bản.

Kết quả cho thấy việc nghiên cứu mô hình tiền huấn luyện là bước quan trọng để hiểu sâu các hệ thống ngôn ngữ lớn, đồng thời hỗ trợ phát triển các phương pháp diễn giải và tối ưu trong tương lai.

---

## Tài liệu tham khảo

[1] Inspecting OpenAI’s GPT-2, Lecture Transcript. 

---

