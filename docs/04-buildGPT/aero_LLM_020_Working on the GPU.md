
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
# Tối Ưu Hóa Huấn Luyện Mô Hình Học Sâu Bằng GPU: Nguyên Lý và Thực Hành

## Tóm tắt (Abstract)

Sự phát triển của các mô hình học sâu và mô hình ngôn ngữ lớn đòi hỏi năng lực tính toán ngày càng cao. Trong bối cảnh đó, Graphics Processing Unit (GPU) trở thành công cụ quan trọng giúp tăng tốc quá trình huấn luyện và suy luận. Bài viết này trình bày sự khác biệt giữa CPU và GPU, cơ chế làm việc của GPU trong học sâu, quy trình chuyển dữ liệu giữa các thiết bị, cũng như các vấn đề thực tiễn khi triển khai bằng PyTorch. Qua đó, bài viết làm rõ vai trò của GPU trong việc nâng cao hiệu suất tính toán cho các mô hình hiện đại.

---

## 1. Giới thiệu

Trong giai đoạn đầu của quá trình học máy, các mô hình và tập dữ liệu thường có kích thước vừa phải, có thể xử lý hiệu quả trên CPU. Tuy nhiên, khi quy mô dữ liệu và độ phức tạp của mô hình tăng lên, việc sử dụng GPU trở nên cần thiết để đảm bảo thời gian huấn luyện hợp lý 

GPU được thiết kế chuyên biệt cho các phép toán song song, đặc biệt là nhân ma trận, vốn là nền tảng của học sâu. Do đó, việc khai thác GPU giúp tăng đáng kể tốc độ xử lý so với CPU truyền thống.

---

## 2. Kiến trúc CPU và GPU

### 2.1. Đặc điểm của CPU

CPU (Central Processing Unit) là bộ xử lý đa năng, được tối ưu cho:

* Xử lý tuần tự.
* Điều khiển luồng chương trình.
* Tương tác thời gian thực.
* Quản lý bộ nhớ và cache.

CPU có số lượng lõi hạn chế nhưng linh hoạt, phù hợp với các tác vụ điều khiển và xử lý logic phức tạp 

---

### 2.2. Đặc điểm của GPU

GPU (Graphics Processing Unit) được thiết kế cho:

* Xử lý song song quy mô lớn.
* Tính toán ma trận.
* Thực hiện nhiều phép toán đơn giản đồng thời.

Phần lớn cấu trúc GPU bao gồm các đơn vị ALU (Arithmetic Logical Unit), giúp thực hiện nhanh các phép toán số học và logic 

GPU có thể được xem là “sức mạnh tính toán” (brawn), trong khi CPU là “bộ não điều khiển” (brains) của hệ thống.

---

### 2.3. So sánh CPU và GPU

| Tiêu chí       | CPU               | GPU              |
| -------------- | ----------------- | ---------------- |
| Cách xử lý     | Tuần tự           | Song song        |
| Số lõi         | Ít                | Rất nhiều        |
| Tính linh hoạt | Cao               | Thấp             |
| Tối ưu cho     | Điều khiển, logic | Ma trận, học sâu |

Trong học sâu, CPU đảm nhiệm việc thiết lập mô hình và xử lý dữ liệu, trong khi GPU thực hiện phần lớn phép toán huấn luyện.

---

## 3. Mô hình xử lý CPU–GPU trong học sâu

### 3.1. Quy trình tổng quát

Quy trình sử dụng GPU trong huấn luyện gồm các bước:

1. Khởi tạo mô hình và dữ liệu trên CPU.
2. Chuyển mô hình và dữ liệu sang GPU.
3. Thực hiện huấn luyện trên GPU.
4. Chuyển kết quả về CPU để xử lý tiếp.

GPU không thể xử lý dữ liệu nằm trên CPU, và ngược lại, việc truy cập dữ liệu GPU phải thông qua cơ chế chuyển đổi 

---

### 3.2. Chi phí truyền dữ liệu

Việc truyền dữ liệu giữa CPU và GPU gây ra:

* Tăng độ phức tạp mã nguồn.
* Tăng chi phí thời gian.
* Nguy cơ phát sinh lỗi.

Tuy chi phí này không lớn, nhưng cần được cân nhắc trong các hệ thống lớn 

---

## 4. Triển khai GPU trong PyTorch

### 4.1. Phát hiện thiết bị

Trong PyTorch, thiết bị thường được xác định như sau:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Cách tiếp cận này cho phép mã chạy linh hoạt trên cả CPU và GPU.

---

### 4.2. Chuyển mô hình và dữ liệu

Việc chuyển dữ liệu và mô hình sang GPU sử dụng phương thức `.to()`:

```python
model = model.to(device)
data = data.to(device)
```

Phương thức này chỉ áp dụng cho đối tượng PyTorch, không dùng cho list hay NumPy array 

---

### 4.3. Tạo dữ liệu trực tiếp trên GPU

Ngoài việc chuyển từ CPU, dữ liệu có thể được tạo trực tiếp trên GPU:

```python
tensor = torch.randn(100, device=device)
```

Cách này giúp giảm chi phí truyền dữ liệu.

---

## 5. Xử lý lỗi phổ biến

### 5.1. Lỗi không đồng bộ thiết bị

Một lỗi thường gặp:

> Expected all tensors to be on the same device.

Nguyên nhân là dữ liệu và mô hình nằm trên hai thiết bị khác nhau (CPU và GPU) 

Giải pháp là đảm bảo mọi tensor và mô hình cùng nằm trên một thiết bị.

---

### 5.2. Chuyển dữ liệu từ GPU về CPU

Để xử lý bằng NumPy hoặc Matplotlib, tensor phải được chuyển về CPU:

```python
output = output.detach().cpu()
```

Với tensor vô hướng, có thể dùng:

```python
value = tensor.item()
```



---

## 6. Đánh giá hiệu năng CPU và GPU

### 6.1. Đồng bộ thời gian

Khi đo thời gian trên GPU, cần đồng bộ hóa:

```python
torch.cuda.synchronize()
```

Việc này đảm bảo độ chính xác khi đo thời gian thực thi 

---

### 6.2. So sánh tốc độ xử lý

Thực nghiệm cho thấy:

* GPU nhanh hơn CPU từ 2 đến 5 lần với các phép toán nhỏ.
* Với ma trận lớn, chênh lệch có thể cao hơn nhiều.

Ngay cả với mô hình nhỏ như GPT-2, GPU có thể giảm thời gian từ hàng chục phút xuống còn vài giây 

---

### 6.3. Ảnh hưởng của truyền dữ liệu

Nếu dữ liệu liên tục được chuyển giữa CPU và GPU, hiệu năng có thể giảm.

Trường hợp lý tưởng là:

* Dữ liệu và mô hình nằm lâu dài trên GPU.
* Hạn chế tối đa việc chuyển đổi thiết bị.



---

## 7. Thảo luận

### 7.1. Khi nào nên dùng GPU?

GPU phù hợp khi:

* Mô hình lớn.
* Dữ liệu nhiều.
* Huấn luyện kéo dài.

CPU phù hợp khi:

* Mô hình nhỏ.
* Thử nghiệm nhanh.
* Phát triển ban đầu.

Không phải mọi tác vụ đều cần GPU.

---

### 7.2. Xu hướng phát triển

Công nghệ GPU đang phát triển nhanh chóng nhờ nhu cầu từ các mô hình ngôn ngữ lớn. Trong tương lai:

* GPU rẻ hơn.
* Hiệu suất cao hơn.
* Dễ tiếp cận hơn.

Điều này giúp mở rộng khả năng nghiên cứu và ứng dụng AI 

---

## 8. Kết luận

Bài viết đã trình bày:

* Sự khác biệt giữa CPU và GPU.
* Mô hình xử lý CPU–GPU trong học sâu.
* Quy trình triển khai GPU với PyTorch.
* Các vấn đề thực tiễn và hiệu năng.

GPU đóng vai trò trung tâm trong huấn luyện mô hình học sâu hiện đại. Việc hiểu rõ cách sử dụng GPU giúp tối ưu thời gian, tài nguyên và độ ổn định của hệ thống.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
