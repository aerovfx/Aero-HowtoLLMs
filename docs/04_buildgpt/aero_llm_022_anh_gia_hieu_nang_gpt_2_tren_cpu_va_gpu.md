
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [04 buildgpt](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
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

$$

$$

self.device = device

$$

$$

$$
tensor = torch.arange(..., device=self.device)
$$
