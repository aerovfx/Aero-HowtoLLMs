
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
# Đánh Giá Hiệu Năng GPT-2 Trên CPU và GPU: Thực Nghiệm Thời Gian Khởi Tạo, Suy Luận và Huấn Luyện

## Tóm tắt (Abstract)

Hiệu năng tính toán là yếu tố then chốt trong việc triển khai và huấn luyện các mô hình ngôn ngữ lớn. Bài viết này trình bày một nghiên cứu thực nghiệm nhằm so sánh thời gian thực thi mô hình GPT-2 (Model 5) trên CPU và GPU thông qua ba tác vụ chính: khởi tạo mô hình, suy luận (forward pass) và huấn luyện bằng lan truyền ngược (backpropagation). Kết quả cho thấy GPU mang lại lợi thế vượt trội về hiệu năng, đặc biệt trong các phép tính ma trận quy mô lớn, với mức cải thiện lên tới nhiều bậc độ lớn so với CPU.

---

## 1. Giới thiệu

Sự phát triển của các mô hình ngôn ngữ lớn (Large Language Models – LLMs) đã làm gia tăng nhu cầu về tài nguyên tính toán hiệu năng cao. Trong khi CPU phù hợp cho các tác vụ điều khiển và thử nghiệm ban đầu, GPU được tối ưu cho xử lý song song và các phép toán ma trận, vốn là nền tảng của học sâu.

Tài liệu tham khảo mô tả một bài toán thực hành nhằm đo lường thời gian thực thi của Model 5 trên CPU và GPU, tập trung vào ba giai đoạn: khởi tạo mô hình, suy luận và huấn luyện 

Mục tiêu của bài viết là:

* Đánh giá định lượng sự khác biệt hiệu năng giữa CPU và GPU.
* Phân tích nguyên nhân của sự chênh lệch.
* Thảo luận ý nghĩa thực tiễn đối với phát triển LLM.

---

## 2. Thiết lập thực nghiệm

### 2.1. Môi trường thực thi

Thực nghiệm được thực hiện trong môi trường có hỗ trợ GPU (ví dụ: NVIDIA A100), sử dụng thư viện PyTorch để xây dựng và triển khai mô hình.

Thiết bị được xác định thông qua biến `device`, cho phép tạo hai phiên bản mô hình:

* Một phiên bản trên CPU.
* Một phiên bản trên GPU.

Cách tiếp cận này giúp đảm bảo tính công bằng trong so sánh hiệu năng 

---

### 2.2. Điều chỉnh mã nguồn

Lớp mô hình được mở rộng thêm tham số `device` để đảm bảo các tensor được tạo đúng trên thiết bị tương ứng. Việc này nhằm tránh lỗi do tensor nằm trên CPU trong khi mô hình nằm trên GPU 

Ví dụ:

```python
self.device = device
tensor = torch.arange(..., device=self.device)
```

Cách thiết kế này giúp mã nguồn linh hoạt và ổn định hơn khi chuyển đổi giữa các thiết bị.

---

## 3. Thực nghiệm 1: Thời gian khởi tạo mô hình

### 3.1. Phương pháp

Trong thí nghiệm đầu tiên, thời gian được đo cho quá trình:

* Khởi tạo mô hình trên GPU.
* Khởi tạo mô hình trên CPU.

Không thực hiện forward pass hay huấn luyện, chỉ đánh giá chi phí tạo mô hình.

Quá trình được bao quanh bởi bộ đếm thời gian (clock timer) 

---

### 3.2. Kết quả

Kết quả điển hình:

* GPU: ~1.5 giây
* CPU: ~1.2 giây

Sự chênh lệch khoảng 300 ms là không đáng kể trong thực tế 

---

### 3.3. Phân tích

Do khởi tạo mô hình chỉ diễn ra một lần trong toàn bộ vòng đời hệ thống, nên sự khác biệt nhỏ này không ảnh hưởng nhiều đến hiệu suất tổng thể. Vì vậy, yếu tố quyết định không nằm ở giai đoạn khởi tạo.

---

## 4. Thực nghiệm 2: Đánh giá Forward Pass

### 4.1. Phương pháp

Trong thí nghiệm thứ hai, mô hình thực hiện suy luận trên dữ liệu giả:

* Batch size: 8
* Sequence length: 1024

Quy trình gồm:

1. Sinh tensor token ngẫu nhiên.
2. Chuyển sang thiết bị tương ứng.
3. Thực hiện forward pass.
4. Lặp lại 5 lần.

Trước khi đo thời gian, GPU được đồng bộ với CPU để đảm bảo độ chính xác 

---

### 4.2. Kết quả

Kết quả thực nghiệm:

* CPU: ~20 giây
* GPU: ~0.03 giây (30 ms)

GPU nhanh hơn CPU khoảng 4 bậc độ lớn 

---

### 4.3. Phân tích

Sự khác biệt lớn xuất phát từ:

* Khả năng xử lý song song của GPU.
* Tối ưu hóa phần cứng cho phép nhân ma trận.
* Băng thông bộ nhớ cao.

Trong bối cảnh sinh token liên tục, việc chờ 20 giây cho mỗi lượt suy luận là không khả thi, khiến CPU không phù hợp cho các hệ thống LLM thực tế.

---

## 5. Thực nghiệm 3: Đánh giá Backpropagation

### 5.1. Phương pháp

Thí nghiệm thứ ba đo thời gian huấn luyện thông qua lan truyền ngược:

* Xây dựng hàm mất mát (loss function).
* Khởi tạo bộ tối ưu (optimizer).
* Thực hiện 5 vòng backpropagation.

Quy trình được thực hiện riêng cho CPU và GPU 

---

### 5.2. Kết quả

Kết quả quan sát:

* GPU: ~1.6 giây
* CPU: > 60 giây

Sự chênh lệch vượt quá một phút cho cùng khối lượng tính toán 

---

### 5.3. Phân tích

Backpropagation yêu cầu:

* Nhiều phép nhân ma trận.
* Tính gradient quy mô lớn.
* Cập nhật tham số liên tục.

Các tác vụ này được GPU xử lý hiệu quả hơn nhiều so với CPU. Khi quy mô mô hình tăng (GPT-2 Medium, Large), khoảng cách này tiếp tục mở rộng.

---

## 6. Thảo luận

### 6.1. Ý nghĩa đối với phát triển LLM

Kết quả cho thấy:

* CPU chỉ phù hợp cho học tập và thử nghiệm nhỏ.
* GPU là điều kiện cần cho huấn luyện và triển khai LLM.
* Hiệu năng ảnh hưởng trực tiếp đến khả năng mở rộng mô hình.

Ngay cả với GPT-2 Small, việc thiếu GPU khiến mô hình gần như không khả thi trong ứng dụng thực tế 

---

### 6.2. Khía cạnh kinh tế và chính sách

Tài liệu cũng nhấn mạnh rằng:

* GPU hiệu năng cao là tài nguyên chiến lược.
* Các quốc gia và tập đoàn lớn cần lượng lớn GPU để phát triển AI.
* Việc kiểm soát xuất khẩu GPU là một biện pháp quản lý rủi ro AI.

Điều này cho thấy mối liên hệ chặt chẽ giữa công nghệ, kinh tế và an ninh trong kỷ nguyên AI 

---

### 6.3. Hạn chế của nghiên cứu

Một số hạn chế bao gồm:

* Chỉ thử nghiệm trên GPT-2 Small.
* Dữ liệu đầu vào là dữ liệu giả.
* Chưa xét đến huấn luyện phân tán đa GPU.

Các nghiên cứu tiếp theo có thể mở rộng sang mô hình lớn hơn và môi trường phân tán.

---

## 7. Kết luận

Bài viết đã trình bày một nghiên cứu thực nghiệm về hiệu năng của GPT-2 trên CPU và GPU, tập trung vào ba giai đoạn chính: khởi tạo, suy luận và huấn luyện.

Các kết quả chính gồm:

* Khởi tạo mô hình: khác biệt không đáng kể.
* Forward pass: GPU nhanh hơn CPU ~10⁴ lần.
* Backpropagation: GPU nhanh hơn CPU hàng chục lần.

Những kết quả này khẳng định GPU là nền tảng không thể thiếu cho việc phát triển và ứng dụng mô hình ngôn ngữ lớn hiện đại.

---

## Tài liệu tham khảo

[1] CodeChallenge: Time Model 5 on CPU and GPU, Lecture Transcript. 

---
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
