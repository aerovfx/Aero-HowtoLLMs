
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [09 quantitative evaluations](index.md)

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
# Phân Phối Của Các Kích Hoạt Trạng Thái Ẩn Trong Mô Hình Ngôn Ngữ

## Tóm tắt

Ngay cả khi có quyền truy cập vào toàn bộ các thông số nội bộ của một mô hình ngôn ngữ lớn (LLM), bản chất phi tuyến tính và phức tạp của chúng làm cho việc hiểu cách mô hình nhận thức và xử lý thông tin trở nên rất khó khăn. Bài viết này khám phá các phương pháp trích xuất mẫu kích hoạt nội tại từ các lớp `transformer` và trực quan hóa phân phối của chúng thông qua biểu đồ phân tán (scatter plots), ma trận hiệp phương sai (covariance matrix), và biểu đồ tần suất (histograms).

---

## 1. Cơ sở về Trạng Thái Ẩn (Hidden States)

Trong một LLM như GPT-2, văn bản đầu vào được mã hóa thành các chỉ số token, sau đó được ánh xạ thành các **vectors nhúng** (embedding vectors). Tại mỗi khối `transformer`, các vector này lại được biến đổi, quay, co giãn, để rồi hình thành nên biểu diễn cuối cùng cho việc dự đoán token.

Bằng cách chạy một lượt lan truyền xuôi (forward pass), ta có thể kích hoạt tùy chọn xuất trạng thái ẩn:

`output_hidden_states = True`

Trong GPT-2 nhỏ, tính toán này sẽ trả ra 13 ten-xơ (tensors), bao gồm:
1 đầu ra từ Lớp Nhúng (Embeddings layer).
12 đầu ra tương ứng từ 12 khối transformer.
Mỗi mạng lưới có cấu hình kích thước dạng `[Batch Size, Sequences, Embedding Dimension]`. Trong GPT, thiết lập này thường là `[1, 62, 768]`.

---

## 2. Các Công Cụ Trực Quan Hóa 

### 2.1 Biểu Đồ Phân Tán (Scatter Plots)

Với biểu đồ phân tán, ta đối chiếu các chỉ số token và chiều biểu diễn (embedding dimensions) với giá trị kích hoạt.

Điểm quan trọng rút ra là **yếu tố nhiễu của token đầu tiên**. Trong tự nhiên, việc xử lý token đầu tiên là phi chuẩn vì không có context (ngữ cảnh) đứng trước nó. Để việc quan sát không bị sai lệch, thông thường token này cần bị loại trừ (sử dụng token có chỉ số 1 trở lên).

### 2.2 Ma Trận Hiệp Phương Sai và $R^2$ (Covariance & $R^2$ Matrix)

Để hiểu được các phép tính ẩn liên đới như thế nào qua từng lớp, ta sử dụng ma trận **Hiệp phương sai** (Covariance) và ma trận tương quan được bình phương ($R^2$, giải thích lượng phương sai được chia sẻ).

R^2 = \text{Corr}(X, Y)^2

Hai đại lượng X và Y hoàn toàn không tương quan sẽ có R^2 \approx 0. Ngược lại, nếu chúng giống hệt, kết quả trả về 1 (hoặc 100%).