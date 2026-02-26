
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [04 buildGPT](../index.md)

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
# Triển Khai Mô Hình GPT-2 Hoàn Chỉnh Trên GPU: Kiến Trúc, Tối Ưu Hóa và Đánh Giá Hiệu Năng

## Tóm tắt (Abstract)

Sự phát triển của các mô hình ngôn ngữ lớn đã thúc đẩy nhu cầu triển khai hiệu quả trên phần cứng tăng tốc như GPU. Bài viết này trình bày quá trình xây dựng một mô hình GPT-2 hoàn chỉnh với trọng số ngẫu nhiên, tập trung vào kiến trúc Multi-Head Attention, tối ưu hóa tham số, tổ chức mã nguồn theo hướng mô-đun, và triển khai trên GPU bằng PyTorch. Ngoài ra, bài viết phân tích phương pháp gộp ma trận truy vấn, khóa và giá trị (QKV), chiến lược chia sẻ tham số, cũng như đánh giá số lượng tham số huấn luyện. Kết quả cho thấy việc tối ưu hóa kiến trúc và quản lý thiết bị tính toán có vai trò quan trọng trong việc nâng cao hiệu suất mô hình.

---

## 1. Giới thiệu

Các mô hình Transformer, đặc biệt là GPT-2, đã trở thành nền tảng cho nhiều hệ thống xử lý ngôn ngữ tự nhiên hiện đại. Tuy nhiên, việc xây dựng và triển khai một mô hình hoàn chỉnh đòi hỏi sự hiểu biết sâu về kiến trúc, tối ưu bộ nhớ và quản lý tài nguyên tính toán.

Tài liệu tham khảo mô tả quá trình xây dựng một mô hình GPT-2 đầy đủ, chạy trên GPU với trọng số ngẫu nhiên, nhằm minh họa cách kết hợp các thành phần đã học thành một hệ thống hoàn chỉnh 

---

## 2. Tổng quan kiến trúc GPT-2

### 2.1. Cấu trúc tổng thể

Mô hình GPT-2 trong nghiên cứu này bao gồm ba thành phần chính:

1. Lớp nhúng từ và vị trí (Token & Position Embeddings)
2. Các khối Transformer
3. Lớp giải mã đầu ra (Unembedding Layer)

Quá trình truyền xuôi (forward pass) được chia thành ba giai đoạn tương ứng, giúp mã nguồn dễ đọc và bảo trì hơn 

---

### 2.2. Thông số siêu tham số

Mô hình được xây dựng theo cấu hình GPT-2 Small:

| Tham số              | Giá trị |
| -------------------- | ------- |
| Số khối Transformer  | 12      |
| Số Attention Head    | 12      |
| Kích thước embedding | 768     |
| Độ dài chuỗi         | 1024    |
| Kích thước từ vựng   | ~50,000 |

Cấu hình này tạo nên một mô hình có quy mô trung bình, phù hợp cho việc thử nghiệm trên GPU 

---

## 3. Tối ưu hóa Multi-Head Attention

### 3.1. Gộp ma trận QKV

Trong mô hình truyền thống, ba ma trận trọng số riêng biệt được sử dụng cho:

* Query (Q)
* Key (K)
* Value (V)

Nghiên cứu này sử dụng chiến lược gộp ba ma trận thành một ma trận duy nhất có kích thước:

[
E \times 3E
]

với (E) là số chiều embedding.

Cách tiếp cận này giúp:

* Giảm số phép toán cấp phát bộ nhớ
* Tăng hiệu quả truyền dữ liệu
* Đơn giản hóa cấu trúc mô hình



---

### 3.2. Tách QKV trong quá trình tính toán

Sau khi nhân dữ liệu đầu vào với ma trận gộp, PyTorch sử dụng hàm chia tensor để tách lại thành ba thành phần Q, K, V riêng biệt.

Quy trình này tương đương với việc sử dụng ba ma trận độc lập, nhưng có hiệu suất cao hơn trong thực thi song song trên GPU 

---

## 4. Thiết kế khối Transformer

### 4.1. Cấu trúc khối

Mỗi khối Transformer bao gồm:

1. Layer Normalization
2. Multi-Head Attention
3. Residual Connection
4. Feed-Forward Network
5. Residual Connection thứ hai

Dạng tổng quát:

[
X_{out} = X + \text{Attention}(\text{LN}(X))
]
[
Y = X_{out} + \text{MLP}(\text{LN}(X_{out}))
]

Cấu trúc này giúp ổn định quá trình huấn luyện và hạn chế hiện tượng gradient biến mất 

---

### 4.2. Quản lý biến trung gian

Tác giả sử dụng các biến trung gian riêng biệt như:

* `X_at`
* `X_ff`

Thay vì ghi đè trực tiếp lên biến gốc, giúp:

* Dễ theo dõi luồng dữ liệu
* Giảm lỗi logic
* Tăng khả năng mở rộng mã nguồn



---

## 5. Mô hình ngôn ngữ hoàn chỉnh

### 5.1. Embedding và Weight Tying

Mô hình sử dụng:

* WTE (Word Token Embedding)
* WPE (Word Position Embedding)

Lớp embedding đầu vào và lớp unembedding đầu ra được chia sẻ trọng số (weight tying), giúp:

* Giảm số tham số
* Cải thiện khả năng tổng quát hóa



---

### 5.2. Dòng xử lý Forward

Forward pass gồm ba giai đoạn:

1. Cộng embedding từ và vị trí
2. Truyền qua 12 khối Transformer
3. Chuẩn hóa và giải mã logits

Mỗi giá trị logit biểu diễn xác suất tiềm năng của token tiếp theo trong chuỗi 

---

## 6. Triển khai trên GPU

### 6.1. Quản lý thiết bị

Mô hình sử dụng biến `device` để điều phối việc chạy trên GPU:

```python
device = torch.device("cuda")
```

Việc đảm bảo tất cả tensor và mô hình nằm trên cùng thiết bị là điều kiện bắt buộc để tránh lỗi thực thi 

---

### 6.2. Lỗi không đồng bộ thiết bị

Một lỗi phổ biến:

> Expected all tensors to be on the same device

Nguyên nhân xuất phát từ việc tensor tạo bằng `torch.arange` mặc định nằm trên CPU.

Giải pháp:

```python
torch.arange(..., device=device)
```



---

## 7. Phân tích tham số mô hình

### 7.1. Đếm tham số bằng torchinfo

Công cụ `torchinfo.summary` được sử dụng để thống kê:

* Kích thước tensor
* Số tham số
* Luồng dữ liệu

Kết quả ban đầu cho thấy mô hình có khoảng:

* 163 triệu tham số



---

### 7.2. Hiệu chỉnh do Weight Tying

Do embedding và unembedding dùng chung trọng số, số tham số thực tế được điều chỉnh:

[
163M - 38M \approx 124M
]

Do đó, mô hình có khoảng 124 triệu tham số huấn luyện thực sự 

---

## 8. Thảo luận

### 8.1. Ý nghĩa thực tiễn

Mô hình minh họa cho thấy:

* Kiến trúc GPT-2 có thể được xây dựng từ các thành phần cơ bản
* Việc tối ưu QKV và weight tying giúp tiết kiệm tài nguyên
* GPU là yếu tố then chốt cho khả năng mở rộng

Các mô hình thương mại hiện đại chủ yếu mở rộng quy mô từ cấu trúc này 

---

### 8.2. Hạn chế

Một số hạn chế:

* Trọng số ngẫu nhiên, chưa huấn luyện
* Chưa đánh giá chất lượng sinh văn bản
* Chưa xét đến phân tán đa GPU

Đây là nền tảng cho các nghiên cứu huấn luyện quy mô lớn hơn.

---

## 9. Kết luận

Bài viết đã trình bày:

* Cách xây dựng GPT-2 hoàn chỉnh
* Tối ưu hóa Multi-Head Attention bằng gộp QKV
* Thiết kế khối Transformer
* Triển khai và xử lý lỗi GPU
* Phân tích tham số mô hình

Kết quả cho thấy việc thiết kế kiến trúc hợp lý và quản lý tài nguyên GPU hiệu quả là yếu tố then chốt trong phát triển mô hình ngôn ngữ lớn hiện đại.

---

## Tài liệu tham khảo

[1] Model 5 – Complete GPT-2 on the GPU, Lecture Transcript. 

--
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
